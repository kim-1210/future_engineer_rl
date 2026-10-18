from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.utils.math import combine_frame_transforms
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.assets import Articulation

from isaaclab.utils.math import quat_apply

from .rewards import (
    TRACK_LENGTH,
    _track_direction,
    _local_xy,
    _robot,
    _track_progress,
    _episode_metric_state,
    required_pillar_pass_count,
    sync_pillar_pass_metrics,
    pillar_collision as _pillar_collision_reward,
    wall_collision as _wall_collision_reward,
    wrong_pillar_pass as _wrong_pillar_pass,
    missed_pillar_pass as _missed_pillar_pass,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def obstacle_collision_termination(
    env: ManagerBasedRLEnv,
    obstacle_half_size: float = 0.025,
    car_half_length: float = 0.12,
    car_half_width: float = 0.085,
) -> torch.Tensor:
    """Reset when the car footprint overlaps a red/green pillar."""
    return _pillar_collision_reward(
        env,
        obstacle_half_size=obstacle_half_size,
        car_half_length=car_half_length,
        car_half_width=car_half_width,
    ).bool()


def wall_collision_termination(
    env: ManagerBasedRLEnv,
    car_half_length: float = 0.12,
    car_half_width: float = 0.085,
) -> torch.Tensor:
    """Reset when the car footprint overlaps the inner/outer wall area."""
    return _wall_collision_reward(
        env,
        car_half_length=car_half_length,
        car_half_width=car_half_width,
    ).bool()


def wrong_pillar_pass_termination(
    env: ManagerBasedRLEnv,
    pass_margin: float = 0.02,
    side_margin: float = 0.13,
    max_lateral_distance: float = 0.45,
    red_correct_left_side: bool = False,
    green_correct_left_side: bool = True,
) -> torch.Tensor:
    return _wrong_pillar_pass(
        env,
        pass_margin=pass_margin,
        side_margin=side_margin,
        max_lateral_distance=max_lateral_distance,
        red_correct_left_side=red_correct_left_side,
        green_correct_left_side=green_correct_left_side,
    ).bool()


def missed_pillar_pass_termination(
    env: ManagerBasedRLEnv,
    pass_margin: float = 0.02,
    side_margin: float = 0.13,
    max_lateral_distance: float = 0.45,
    red_correct_left_side: bool = False,
    green_correct_left_side: bool = True,
) -> torch.Tensor:
    return _missed_pillar_pass(
        env,
        pass_margin=pass_margin,
        side_margin=side_margin,
        max_lateral_distance=max_lateral_distance,
        red_correct_left_side=red_correct_left_side,
        green_correct_left_side=green_correct_left_side,
    ).bool()


def collision_termination(
    env: ManagerBasedRLEnv,
    obstacle_half_size: float = 0.025,
    car_half_length: float = 0.12,
    car_half_width: float = 0.085,
) -> torch.Tensor:
    """Reset when the car hits either a pillar or a wall."""
    obstacle_hit = obstacle_collision_termination(
        env,
        obstacle_half_size=obstacle_half_size,
        car_half_length=car_half_length,
        car_half_width=car_half_width,
    )
    wall_hit = wall_collision_termination(
        env,
        car_half_length=car_half_length,
        car_half_width=car_half_width,
    )

    return obstacle_hit | wall_hit


def contact_collision_termination(
    env: ManagerBasedRLEnv,
    sensor_name: str = "car_contact",
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """Reset when the car physically contacts a filtered wall or obstacle."""
    contact_sensor = env.scene[sensor_name]
    forces = contact_sensor.data.force_matrix_w

    if forces is None:
        forces = contact_sensor.data.net_forces_w

    force_norm = torch.norm(forces, dim=-1)
    reduce_dims = tuple(range(1, force_norm.ndim))

    if not reduce_dims:
        return force_norm > force_threshold

    return torch.amax(force_norm, dim=reduce_dims) > force_threshold


def no_progress_termination(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    min_progress: float = 0.0004,
    min_move_distance: float = 0.0002,
    max_stale_steps: int = 75,
    grace_steps: int = 90,
) -> torch.Tensor:
    """Reset when the car stops making forward progress for too long."""
    robot = _robot(env, asset_cfg)
    direction = _track_direction(env, robot.num_instances, robot.device)
    position_xy = _local_xy(env, robot.data.root_pos_w)
    progress = _track_progress(position_xy, direction)
    num_envs = progress.shape[0]
    device = progress.device

    state = getattr(env, "_future_engineer_no_progress_state", None)
    if (
        state is None
        or state["prev_progress"].shape[0] != num_envs
        or "prev_position" not in state
    ):
        state = {
            "prev_progress": progress.clone(),
            "prev_position": position_xy.clone(),
            "stale_steps": torch.zeros(num_envs, device=device, dtype=torch.long),
        }
        env._future_engineer_no_progress_state = state
        return torch.zeros(num_envs, device=device, dtype=torch.bool)

    delta = progress - state["prev_progress"]
    delta = torch.where(delta < -TRACK_LENGTH * 0.5, delta + TRACK_LENGTH, delta)
    delta = torch.where(delta > TRACK_LENGTH * 0.5, delta - TRACK_LENGTH, delta)

    episode_length_buf = getattr(
        env,
        "episode_length_buf",
        torch.zeros(num_envs, device=device, dtype=torch.long),
    )
    just_reset = episode_length_buf <= 1
    move_distance = torch.norm(position_xy - state["prev_position"], dim=1)
    made_progress = (delta > min_progress) | (move_distance > min_move_distance)

    state["stale_steps"] = torch.where(
        just_reset | made_progress,
        torch.zeros_like(state["stale_steps"]),
        state["stale_steps"] + 1,
    )
    state["prev_progress"] = progress.clone()
    state["prev_position"] = position_xy.clone()

    return (state["stale_steps"] > max_stale_steps) & (episode_length_buf > grace_steps)


def task_success_termination(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    required_laps: int = 1,
    obstacle_half_size: float = 0.025,
    car_half_length: float = 0.12,
    car_half_width: float = 0.085,
    pass_margin: float = 0.02,
    side_margin: float = 0.13,
    max_lateral_distance: float = 0.45,
    red_correct_left_side: bool = False,
    green_correct_left_side: bool = True,
) -> torch.Tensor:
    """End the episode when the WRO-style task is actually completed."""
    robot = _robot(env, asset_cfg)
    direction = _track_direction(env, robot.num_instances, robot.device)
    position_xy = _local_xy(env, robot.data.root_pos_w)
    progress = _track_progress(position_xy, direction)
    num_envs = progress.shape[0]
    device = progress.device

    state = getattr(env, "_future_engineer_task_success_state", None)
    if (
        state is None
        or state["prev_progress"].shape[0] != num_envs
        or "lap_count" not in state
        or "reset_done" not in state
    ):
        state = {
            "prev_progress": progress.clone(),
            "lap_progress": torch.zeros(num_envs, device=device),
            "lap_count": torch.zeros(num_envs, device=device, dtype=torch.long),
            "reset_done": torch.zeros(num_envs, device=device, dtype=torch.bool),
        }
        env._future_engineer_task_success_state = state
        return torch.zeros(num_envs, device=device, dtype=torch.bool)

    episode_length_buf = getattr(
        env,
        "episode_length_buf",
        torch.zeros(num_envs, device=device, dtype=torch.long),
    )
    just_reset = episode_length_buf <= 1
    needs_reset = just_reset & (~state["reset_done"])
    if torch.any(needs_reset):
        state["prev_progress"][needs_reset] = progress[needs_reset]
        state["lap_progress"][needs_reset] = 0.0
        state["lap_count"][needs_reset] = 0
    state["reset_done"] = torch.where(
        just_reset,
        torch.ones_like(state["reset_done"]),
        torch.zeros_like(state["reset_done"]),
    )

    delta = progress - state["prev_progress"]
    delta = torch.where(delta < -TRACK_LENGTH * 0.5, delta + TRACK_LENGTH, delta)
    delta = torch.where(delta > TRACK_LENGTH * 0.5, delta - TRACK_LENGTH, delta)
    delta = torch.where(just_reset, torch.zeros_like(delta), delta)

    lap_progress = torch.clamp(state["lap_progress"] + delta, min=0.0)
    completed = lap_progress >= TRACK_LENGTH
    state["lap_count"] += completed.long()
    state["lap_progress"] = torch.where(completed, lap_progress - TRACK_LENGTH, lap_progress)
    state["prev_progress"] = progress.clone()

    sync_pillar_pass_metrics(
        env,
        pass_margin=pass_margin,
        side_margin=side_margin,
        max_lateral_distance=max_lateral_distance,
        red_correct_left_side=red_correct_left_side,
        green_correct_left_side=green_correct_left_side,
    )
    _pillar_collision_reward(
        env,
        obstacle_half_size=obstacle_half_size,
        car_half_length=car_half_length,
        car_half_width=car_half_width,
    )
    _wall_collision_reward(
        env,
        car_half_length=car_half_length,
        car_half_width=car_half_width,
    )
    metric_state = _episode_metric_state(env, num_envs, device)
    metric_state["lap_count"] = torch.maximum(metric_state["lap_count"], state["lap_count"].float())

    clean_episode = (
        (metric_state["wrong_pillar_pass_count"] <= 0.0)
        & (metric_state["missed_pillar_pass_count"] <= 0.0)
        & (~metric_state["pillar_collision"])
        & (~metric_state["wall_collision"])
    )
    required_pass_count = required_pillar_pass_count(env, num_envs, device)
    episode_stage = getattr(env, "_future_engineer_obstacle_stage", None)
    if episode_stage is None or episode_stage.shape[0] != num_envs:
        global_stage = getattr(env, "_future_engineer_curriculum_stage", 1)
        if isinstance(global_stage, torch.Tensor):
            global_stage = int(global_stage.item())
        episode_stage = torch.full((num_envs,), int(global_stage), device=device, dtype=torch.long)
    else:
        episode_stage = episode_stage.to(device=device, dtype=torch.long)

    required_pass_count = torch.where(
        (episode_stage >= 2) & (episode_stage < 4),
        torch.maximum(required_pass_count, torch.ones_like(required_pass_count)),
        required_pass_count,
    )
    passed_required_pillars = metric_state["valid_pillar_pass_count"] >= required_pass_count
    completed_required_laps = state["lap_count"] >= required_laps

    return completed_required_laps & passed_required_pillars & clean_episode
