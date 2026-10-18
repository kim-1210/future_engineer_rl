from __future__ import annotations

import os
import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.utils.math import combine_frame_transforms
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.assets import Articulation

from isaaclab.utils.math import quat_apply

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


REWARD_WEIGHT_SCHEDULE = (
    (
        0,
        {
            "progress_delta_reward": 10.0,
            "track_following_reward": 4.0,
            "low_speed_penalty": -1.5,
            "wall_close_penalty": -8.0,
            "reverse_penalty": -0.5,
            "pillar_approach_reward": 0.0,
            "correct_pillar_reward": 0.0,
            "wrong_pillar_pass": 0.0,
            "pillar_collision": -500.0,
            "wall_collision": -600.0,
            "lap_complete": 35.0,
            "no_progress_penalty": -500.0,
        },
    ),
    (
        240_000,
        {
            "progress_delta_reward": 10.0,
            "track_following_reward": 4.2,
            "low_speed_penalty": -1.6,
            "wall_close_penalty": -8.5,
            "reverse_penalty": -0.6,
            "pillar_approach_reward": 1.5,
            "correct_pillar_reward": 6.0,
            "wrong_pillar_pass": -25.0,
            "pillar_collision": -550.0,
            "wall_collision": -650.0,
            "lap_complete": 45.0,
            "no_progress_penalty": -550.0,
        },
    ),
    (
        480_000,
        {
            "progress_delta_reward": 9.5,
            "track_following_reward": 4.6,
            "low_speed_penalty": -1.8,
            "wall_close_penalty": -9.5,
            "reverse_penalty": -0.7,
            "pillar_approach_reward": 3.0,
            "correct_pillar_reward": 12.0,
            "wrong_pillar_pass": -50.0,
            "pillar_collision": -650.0,
            "wall_collision": -750.0,
            "lap_complete": 60.0,
            "no_progress_penalty": -650.0,
        },
    ),
    (
        720_000,
        {
            "progress_delta_reward": 9.0,
            "track_following_reward": 5.0,
            "low_speed_penalty": -2.0,
            "wall_close_penalty": -10.5,
            "reverse_penalty": -0.8,
            "pillar_approach_reward": 4.5,
            "correct_pillar_reward": 20.0,
            "wrong_pillar_pass": -80.0,
            "pillar_collision": -750.0,
            "wall_collision": -850.0,
            "lap_complete": 75.0,
            "no_progress_penalty": -750.0,
        },
    ),
    (
        1_000_000,
        {
            "progress_delta_reward": 8.5,
            "track_following_reward": 5.5,
            "low_speed_penalty": -2.3,
            "wall_close_penalty": -12.0,
            "reverse_penalty": -0.9,
            "pillar_approach_reward": 6.5,
            "correct_pillar_reward": 30.0,
            "wrong_pillar_pass": -120.0,
            "pillar_collision": -850.0,
            "wall_collision": -950.0,
            "lap_complete": 90.0,
            "no_progress_penalty": -850.0,
        },
    ),
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


def _interpolate_reward_weights(step: int | None) -> tuple[int, dict[str, float]]:
    if step is None:
        return len(REWARD_WEIGHT_SCHEDULE), dict(REWARD_WEIGHT_SCHEDULE[-1][1])

    if step <= REWARD_WEIGHT_SCHEDULE[0][0]:
        return 1, dict(REWARD_WEIGHT_SCHEDULE[0][1])

    for index in range(1, len(REWARD_WEIGHT_SCHEDULE)):
        start_step, start_weights = REWARD_WEIGHT_SCHEDULE[index - 1]
        end_step, end_weights = REWARD_WEIGHT_SCHEDULE[index]

        if step < end_step:
            alpha = (step - start_step) / max(end_step - start_step, 1)
            weights = {
                term_name: start_weight + (end_weights[term_name] - start_weight) * alpha
                for term_name, start_weight in start_weights.items()
            }
            return index, weights

    return len(REWARD_WEIGHT_SCHEDULE), dict(REWARD_WEIGHT_SCHEDULE[-1][1])


def reward_weight_curriculum(env: ManagerBasedRLEnv, env_ids=None) -> float:
    step = _curriculum_step(env)
    stage, weights = _interpolate_reward_weights(step)

    for term_name, weight in weights.items():
        term_cfg = env.reward_manager.get_term_cfg(term_name)
        if abs(term_cfg.weight - weight) > 1.0e-6:
            term_cfg.weight = weight
            env.reward_manager.set_term_cfg(term_name, term_cfg)

    return float(stage)
