from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


MAX_CURRICULUM_STAGE = 6
DEFAULT_ROLLING_WINDOW = 4096
DEFAULT_MIN_EPISODES = 4096
DEFAULT_MIN_STAGE_STEPS = 7_200
REQUIRED_CONSECUTIVE_PASSES = 3

METRIC_KEYS = (
    "episode_success",
    "lap_success",
    "correct_pass",
    "wrong_pass",
    "missed_pass",
    "pillar_collision",
    "wall_collision",
    "no_progress",
    "red_present",
    "green_present",
    "red_correct",
    "green_correct",
)


def _curriculum_step(env: ManagerBasedRLEnv) -> int | None:
    override = os.environ.get("FUTURE_ENGINEER_CURRICULUM_STEP_OVERRIDE")
    if override is not None:
        try:
            return int(override)
        except ValueError:
            pass

    step = getattr(env, "common_step_counter", None)
    if step is None:
        return None
    if isinstance(step, torch.Tensor):
        return int(step.item())
    return int(step)


def _forced_stage() -> int | None:
    override = os.environ.get("FUTURE_ENGINEER_CURRICULUM_STAGE_OVERRIDE")
    if override is None:
        return None

    try:
        return max(1, min(MAX_CURRICULUM_STAGE, int(override)))
    except ValueError:
        return None


def _initial_stage() -> int | None:
    override = os.environ.get("FUTURE_ENGINEER_CURRICULUM_INITIAL_STAGE")
    if override is None:
        return None

    try:
        return max(1, min(MAX_CURRICULUM_STAGE, int(override)))
    except ValueError:
        return None


def curriculum_stage_from_step(step: int | None) -> int:
    initial_stage = _initial_stage()
    if initial_stage is not None:
        return initial_stage
    return 1


def curriculum_stage(env: ManagerBasedRLEnv) -> int:
    forced_stage = _forced_stage()
    if forced_stage is not None:
        return forced_stage

    state = getattr(env, "_future_engineer_curriculum_state", None)
    if state is not None:
        return int(state["stage"])

    return curriculum_stage_from_step(_curriculum_step(env))


def _make_state(
    env: ManagerBasedRLEnv,
    rolling_window: int,
    device: torch.device,
) -> dict[str, object]:
    stage = curriculum_stage_from_step(_curriculum_step(env))
    step = _curriculum_step(env)
    if step is None:
        step = 0

    return {
        "stage": stage,
        "stage_start_step": int(step),
        "condition_streak": 0,
        "histories": {key: torch.zeros(rolling_window, device=device) for key in METRIC_KEYS},
        "history_index": torch.tensor(0, device=device, dtype=torch.long),
        "history_filled": torch.tensor(0, device=device, dtype=torch.long),
    }


def _curriculum_state(
    env: ManagerBasedRLEnv,
    rolling_window: int,
    device: torch.device,
) -> dict[str, object]:
    state = getattr(env, "_future_engineer_curriculum_state", None)
    if (
        state is None
        or "histories" not in state
        or state["histories"]["episode_success"].shape[0] != rolling_window
        or state["histories"]["episode_success"].device != device
    ):
        state = _make_state(env, rolling_window, device)
        env._future_engineer_curriculum_state = state
    return state


def _reset_stage_history(
    state: dict[str, object],
    stage: int,
    step: int | None,
) -> None:
    state["stage"] = stage
    state["stage_start_step"] = 0 if step is None else int(step)
    state["condition_streak"] = 0
    for history in state["histories"].values():
        history.zero_()
    state["history_index"].zero_()
    state["history_filled"].zero_()


def _term_value(
    env: ManagerBasedRLEnv,
    term_name: str,
    env_ids: torch.Tensor,
) -> torch.Tensor:
    try:
        return env.termination_manager.get_term(term_name)[env_ids].to(device=env_ids.device).bool()
    except Exception:
        return torch.zeros(env_ids.shape[0], device=env_ids.device, dtype=torch.bool)


def _finished_mask(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> torch.Tensor:
    finished = torch.zeros(env_ids.shape[0], device=env_ids.device, dtype=torch.bool)
    for term_name in (
        "task_success",
        "time_out",
        "pillar_collision",
        "wall_collision",
        "wrong_pillar_pass",
        "missed_pillar_pass",
        "no_progress",
    ):
        finished |= _term_value(env, term_name, env_ids)

    episode_length_buf = getattr(env, "episode_length_buf", None)
    if episode_length_buf is not None:
        episode_length = episode_length_buf[env_ids.to(device=episode_length_buf.device)]
        finished &= episode_length.to(device=finished.device) > 1

    return finished


def _env_vector(
    env: ManagerBasedRLEnv,
    attr_name: str,
    env_ids: torch.Tensor,
    default_value: float = 0.0,
) -> torch.Tensor:
    values = getattr(env, attr_name, None)
    if values is None or values.shape[0] != env.scene.num_envs:
        return torch.full((env_ids.shape[0],), default_value, device=env_ids.device)
    return values[env_ids.to(device=values.device)].to(device=env_ids.device, dtype=torch.float32)


def _metric_state_values(
    env: ManagerBasedRLEnv,
    key: str,
    env_ids: torch.Tensor,
) -> torch.Tensor:
    metric_state = getattr(env, "_future_engineer_episode_metric_state", None)
    if metric_state is None or key not in metric_state:
        return torch.zeros(env_ids.shape[0], device=env_ids.device)
    values = metric_state[key]
    return values[env_ids.to(device=values.device)].to(device=env_ids.device, dtype=torch.float32)


def _episode_stage(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> torch.Tensor:
    stage = getattr(env, "_future_engineer_episode_stage", None)
    if stage is None or stage.shape[0] != env.scene.num_envs:
        current_stage = curriculum_stage(env)
        return torch.full((env_ids.shape[0],), current_stage, device=env_ids.device, dtype=torch.long)
    return stage[env_ids.to(device=stage.device)].to(device=env_ids.device, dtype=torch.long)


def _record_finished_episodes(
    env: ManagerBasedRLEnv,
    env_ids,
    state: dict[str, object],
) -> int:
    device = state["histories"]["episode_success"].device
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=device)
    elif not isinstance(env_ids, torch.Tensor):
        env_ids = torch.as_tensor(env_ids, device=device, dtype=torch.long)
    else:
        env_ids = env_ids.to(device=device, dtype=torch.long)

    env_ids = env_ids[_finished_mask(env, env_ids)]
    if env_ids.numel() == 0:
        return 0

    recorded = getattr(env, "_future_engineer_curriculum_recorded", None)
    if (
        recorded is None
        or recorded.shape[0] != env.scene.num_envs
        or recorded.device != device
    ):
        recorded = torch.zeros(env.scene.num_envs, device=device, dtype=torch.bool)
        env._future_engineer_curriculum_recorded = recorded

    current_stage = int(state["stage"])
    same_stage = _episode_stage(env, env_ids) == current_stage
    env_ids = env_ids[(~recorded[env_ids]) & same_stage]
    if env_ids.numel() == 0:
        return 0

    required_total = _env_vector(env, "_future_engineer_required_pillar_pass_count", env_ids)
    required_red = _env_vector(env, "_future_engineer_required_red_pillar_pass_count", env_ids)
    required_green = _env_vector(env, "_future_engineer_required_green_pillar_pass_count", env_ids)

    valid_pass = _metric_state_values(env, "valid_pillar_pass_count", env_ids)
    red_pass = _metric_state_values(env, "red_pillar_pass_count", env_ids)
    green_pass = _metric_state_values(env, "green_pillar_pass_count", env_ids)
    lap_count = _metric_state_values(env, "lap_count", env_ids)

    samples = {
        "episode_success": _term_value(env, "task_success", env_ids).float(),
        "lap_success": (lap_count >= 1.0).float(),
        "correct_pass": (valid_pass >= required_total).float(),
        "wrong_pass": _term_value(env, "wrong_pillar_pass", env_ids).float(),
        "missed_pass": _term_value(env, "missed_pillar_pass", env_ids).float(),
        "pillar_collision": _term_value(env, "pillar_collision", env_ids).float(),
        "wall_collision": _term_value(env, "wall_collision", env_ids).float(),
        "no_progress": _term_value(env, "no_progress", env_ids).float(),
        "red_present": (required_red > 0.0).float(),
        "green_present": (required_green > 0.0).float(),
        "red_correct": ((required_red <= 0.0) | (red_pass >= required_red)).float(),
        "green_correct": ((required_green <= 0.0) | (green_pass >= required_green)).float(),
    }

    count = env_ids.numel()
    window = state["histories"]["episode_success"].shape[0]
    positions = (torch.arange(count, device=device, dtype=torch.long) + state["history_index"]) % window

    for key, values in samples.items():
        state["histories"][key][positions] = values.to(device=device)

    state["history_index"] = (state["history_index"] + count) % window
    state["history_filled"] = torch.clamp(state["history_filled"] + count, max=window)
    recorded[env_ids] = True
    return count


def _rate(state: dict[str, object], key: str) -> float:
    filled = int(state["history_filled"].item())
    if filled <= 0:
        return 0.0
    return float(state["histories"][key][:filled].mean().item())


def _conditional_rate(state: dict[str, object], value_key: str, present_key: str) -> float:
    filled = int(state["history_filled"].item())
    if filled <= 0:
        return 0.0
    present = state["histories"][present_key][:filled] > 0.5
    if not torch.any(present):
        return 0.0
    return float(state["histories"][value_key][:filled][present].mean().item())


def _stage_conditions_met(state: dict[str, object]) -> bool:
    stage = int(state["stage"])
    wrong_pass = _rate(state, "wrong_pass")
    missed_pass = _rate(state, "missed_pass")
    pillar_collision = _rate(state, "pillar_collision")
    wall_collision = _rate(state, "wall_collision")
    no_progress = _rate(state, "no_progress")

    if stage == 1:
        return _rate(state, "lap_success") >= 0.90 and wall_collision <= 0.05 and no_progress <= 0.05

    if stage == 2:
        return (
            _rate(state, "episode_success") >= 0.85
            and _rate(state, "correct_pass") >= 0.85
            and wrong_pass <= 0.05
            and missed_pass <= 0.05
            and pillar_collision <= 0.05
            and wall_collision <= 0.10
            and no_progress <= 0.05
            and _conditional_rate(state, "red_correct", "red_present") >= 0.80
            and _conditional_rate(state, "green_correct", "green_present") >= 0.80
        )

    if stage == 3:
        return (
            _rate(state, "episode_success") >= 0.85
            and _rate(state, "correct_pass") >= 0.85
            and wrong_pass <= 0.05
            and missed_pass <= 0.05
            and pillar_collision <= 0.05
            and wall_collision <= 0.10
            and no_progress <= 0.05
            and _conditional_rate(state, "red_correct", "red_present") >= 0.80
            and _conditional_rate(state, "green_correct", "green_present") >= 0.80
        )

    if stage == 4:
        return (
            _rate(state, "episode_success") >= 0.80
            and wrong_pass <= 0.05
            and missed_pass <= 0.05
            and pillar_collision <= 0.05
            and wall_collision <= 0.10
            and no_progress <= 0.05
        )

    if stage == 5:
        return (
            _rate(state, "episode_success") >= 0.85
            and wrong_pass <= 0.05
            and missed_pass <= 0.05
            and pillar_collision <= 0.05
            and wall_collision <= 0.05
            and no_progress <= 0.05
            and _conditional_rate(state, "red_correct", "red_present") >= 0.85
            and _conditional_rate(state, "green_correct", "green_present") >= 0.85
        )

    return False


def _maybe_advance_stage(
    env: ManagerBasedRLEnv,
    state: dict[str, object],
    min_stage_steps: int,
    min_episodes: int,
    required_consecutive_passes: int,
) -> bool:
    stage = int(state["stage"])
    if stage >= MAX_CURRICULUM_STAGE:
        return False

    step = _curriculum_step(env)
    if step is None:
        return False

    stage_steps = int(step) - int(state["stage_start_step"])
    if stage_steps < min_stage_steps or int(state["history_filled"].item()) < min_episodes:
        state["condition_streak"] = 0
        return False

    if _stage_conditions_met(state):
        state["condition_streak"] += 1
    else:
        state["condition_streak"] = 0

    if state["condition_streak"] < required_consecutive_passes:
        return False

    _reset_stage_history(state, stage + 1, step)
    return True


def reward_weight_curriculum(
    env: ManagerBasedRLEnv,
    env_ids,
    rolling_window: int = DEFAULT_ROLLING_WINDOW,
    min_episodes: int = DEFAULT_MIN_EPISODES,
    min_stage_steps: int = DEFAULT_MIN_STAGE_STEPS,
    required_consecutive_passes: int = REQUIRED_CONSECUTIVE_PASSES,
) -> dict[str, float]:
    device = torch.device(getattr(env, "device", "cpu"))
    state = _curriculum_state(env, rolling_window, device)

    forced_stage = _forced_stage()
    if forced_stage is not None:
        if int(state["stage"]) != forced_stage:
            _reset_stage_history(state, forced_stage, _curriculum_step(env))
        recorded_episodes = 0
        advanced = False
    else:
        recorded_episodes = _record_finished_episodes(env, env_ids, state)
        advanced = _maybe_advance_stage(
            env,
            state,
            min_stage_steps,
            min_episodes,
            required_consecutive_passes,
        )

    stage = int(state["stage"])
    env._future_engineer_curriculum_stage = stage

    step = _curriculum_step(env)
    stage_steps = 0.0 if step is None else float(int(step) - int(state["stage_start_step"]))

    return {
        "stage": float(stage),
        "episode_success_rate": _rate(state, "episode_success"),
        "lap_success_rate": _rate(state, "lap_success"),
        "correct_pass_rate": _rate(state, "correct_pass"),
        "wrong_pass_rate": _rate(state, "wrong_pass"),
        "missed_pass_rate": _rate(state, "missed_pass"),
        "pillar_collision_rate": _rate(state, "pillar_collision"),
        "wall_collision_rate": _rate(state, "wall_collision"),
        "no_progress_rate": _rate(state, "no_progress"),
        "red_correct_rate": _conditional_rate(state, "red_correct", "red_present"),
        "green_correct_rate": _conditional_rate(state, "green_correct", "green_present"),
        "history_size": float(state["history_filled"].item()),
        "recorded_episodes": float(recorded_episodes),
        "stage_steps": stage_steps,
        "condition_streak": float(state["condition_streak"]),
        "advanced": 1.0 if advanced else 0.0,
    }
