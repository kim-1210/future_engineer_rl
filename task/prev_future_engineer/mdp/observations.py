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
    _local_xy,
    _track_direction,
    _track_lookahead_tangent,
    _track_signed_lateral_error,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def tof_distance(env, sensor_name: str, max_distance: float = 1.0):
    sensor = env.scene[sensor_name]

    hit_pos_w = sensor.data.ray_hits_w
    sensor_pos_w = sensor.data.pos_w[:, None, :]

    distance = torch.norm(hit_pos_w - sensor_pos_w, dim=-1)

    distance = torch.where(
        torch.isfinite(distance),
        distance,
        torch.full_like(distance, max_distance),
    )

    distance = torch.clamp(distance, 0.0, max_distance)
    min_distance = torch.min(distance, dim=1, keepdim=True).values

    return min_distance

def pillar_color_features(env, sensor_name: str):
    camera = env.scene[sensor_name]
    rgb = camera.data.output["rgb"].float()

    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]

    red_mask = (r > 150) & (g < 100) & (b < 100)
    green_mask = (g > 150) & (r < 120) & (b < 120)

    num_envs = rgb.shape[0]
    width = rgb.shape[2]

    x_coords = torch.linspace(
        -1.0,
        1.0,
        width,
        device=rgb.device,
    ).view(1, 1, width)

    def mask_features(mask):
        mask_f = mask.float()
        count = mask_f.sum(dim=(1, 2))

        visible = (count > 10).float()
        size = count / (mask.shape[1] * mask.shape[2])

        x_sum = (mask_f * x_coords).sum(dim=(1, 2))
        x = torch.where(
            count > 0,
            x_sum / torch.clamp(count, min=1.0),
            torch.zeros_like(count),
        )

        return visible, x, size

    red_visible, red_x, red_size = mask_features(red_mask)
    green_visible, green_x, green_size = mask_features(green_mask)

    return torch.stack(
        [
            red_visible,
            red_x,
            red_size,
            green_visible,
            green_x,
            green_size,
        ],
        dim=1,
    )

def camera_rgb(env, sensor_name: str):
    camera = env.scene[sensor_name]

    rgb = camera.data.output["rgb"]

    return rgb.float().flatten(start_dim=1) / 255.0

def yaw_rate(env, sensor_name: str):
    imu = env.scene[sensor_name]

    return imu.data.ang_vel_b[:, 2:3]

def speed(env):
    robot = env.scene["robot"]

    forward_axis_b = torch.zeros_like(robot.data.root_pos_w)
    forward_axis_b[:, 0] = 1.0
    forward_axis_w = quat_apply(robot.data.root_quat_w, forward_axis_b)[:, :2]
    forward_axis_w = torch.nn.functional.normalize(forward_axis_w, dim=1)

    forward_speed = torch.sum(robot.data.root_lin_vel_w[:, :2] * forward_axis_w, dim=1, keepdim=True)

    return forward_speed

def track_direction(env):
    robot = env.scene["robot"]

    direction = getattr(
        env,
        "_future_engineer_track_direction",
        None,
    )

    if direction is None:
        return torch.ones(
            (robot.num_instances, 1),
            device=robot.device,
        )

    return direction.to(
        device=robot.device,
        dtype=torch.float32,
    ).unsqueeze(1)


def track_lateral_error(env):
    robot = env.scene["robot"]
    direction = _track_direction(env, robot.num_instances, robot.device)
    local_xy = _local_xy(env, robot.data.root_pos_w)
    signed_error = _track_signed_lateral_error(local_xy, direction)

    return signed_error.unsqueeze(1)


def heading_alignment(env):
    robot = env.scene["robot"]
    direction = _track_direction(env, robot.num_instances, robot.device)
    local_xy = _local_xy(env, robot.data.root_pos_w)
    tangent = _track_lookahead_tangent(local_xy, direction)

    forward_axis_b = torch.zeros_like(robot.data.root_pos_w)
    forward_axis_b[:, 0] = 1.0
    forward_axis_w = quat_apply(robot.data.root_quat_w, forward_axis_b)[:, :2]
    forward_axis_w = torch.nn.functional.normalize(forward_axis_w, dim=1)

    return torch.sum(forward_axis_w * tangent, dim=1, keepdim=True)
