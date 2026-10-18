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
    pillar_collision as _pillar_collision_reward,
    wall_collision as _wall_collision_reward,
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
