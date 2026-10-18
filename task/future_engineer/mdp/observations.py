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


def tof_distance_bins(env, sensor_name: str, max_distance: float = 1.0, num_bins: int = 1):
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

    num_rays = distance.shape[1]
    binned_distances = []
    for index in range(num_bins):
        start = index * num_rays // num_bins
        end = (index + 1) * num_rays // num_bins
        if start == end:
            binned_distances.append(torch.full((distance.shape[0],), max_distance, device=distance.device))
        else:
            binned_distances.append(torch.min(distance[:, start:end], dim=1).values)

    return torch.stack(binned_distances, dim=1)


def front_tof_bins(env, sensor_name: str = "front_tof", max_distance: float = 0.6):
    return tof_distance_bins(env, sensor_name=sensor_name, max_distance=max_distance, num_bins=5)


def side_tof_bins(env, sensor_name: str, max_distance: float = 0.5):
    return tof_distance_bins(env, sensor_name=sensor_name, max_distance=max_distance, num_bins=3)


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


def current_steering_angle(
    env,
    left_joint_name: str = "front_left_steer_joint",
    right_joint_name: str = "front_right_steer_joint",
    max_steering_angle: float = 0.5236,
):
    robot = env.scene["robot"]
    state_name = "_future_engineer_steering_joint_ids"
    joint_ids = getattr(env, state_name, None)

    if joint_ids is None:
        left_ids, _ = robot.find_joints([left_joint_name])
        right_ids, _ = robot.find_joints([right_joint_name])
        joint_ids = [left_ids[0], right_ids[0]]
        setattr(env, state_name, joint_ids)

    steering = torch.mean(robot.data.joint_pos[:, joint_ids], dim=1, keepdim=True)
    return torch.clamp(steering / max(max_steering_angle, 1.0e-4), -1.0, 1.0)


def previous_actions(env, action_name: str = "drive", action_indices: tuple[int, int] = (0, 1)):
    robot = env.scene["robot"]
    action_term = None
    if hasattr(env.action_manager, "get_term"):
        try:
            action_term = env.action_manager.get_term(action_name)
        except Exception:
            action_term = None
    elif hasattr(env.action_manager, "_terms"):
        action_term = env.action_manager._terms.get(action_name)

    clipped_actions = getattr(action_term, "clipped_actions", None)
    if clipped_actions is not None:
        return clipped_actions[:, list(action_indices)]

    prev_action = getattr(env.action_manager, "prev_action", None)
    if prev_action is None:
        return torch.zeros((robot.num_instances, len(action_indices)), device=robot.device)

    return torch.clamp(prev_action[:, list(action_indices)], -1.0, 1.0)


def nearest_pillar_state(env, max_forward_distance: float = 1.2):
    robot = env.scene["robot"]
    num_envs = robot.num_instances
    device = robot.device

    forward_axis_b = torch.zeros((num_envs, 3), device=device)
    left_axis_b = torch.zeros_like(forward_axis_b)
    forward_axis_b[:, 0] = 1.0
    left_axis_b[:, 1] = 1.0
    forward_axis_w = quat_apply(robot.data.root_quat_w, forward_axis_b)[:, :2]
    left_axis_w = quat_apply(robot.data.root_quat_w, left_axis_b)[:, :2]
    forward_axis_w = torch.nn.functional.normalize(forward_axis_w, dim=1)
    left_axis_w = torch.nn.functional.normalize(left_axis_w, dim=1)

    rel_forward_values = []
    rel_left_values = []
    color_values = []
    distance_values = []

    for collection_name, color_value in (("red_obstacles", 1.0), ("green_obstacles", -1.0)):
        objects = env.scene[collection_name]
        pillar_pos_w = objects.data.object_pos_w
        local_xy = pillar_pos_w[..., :2] - env.scene.env_origins[:, None, :2]
        visible = (
            (pillar_pos_w[..., 2] > -1.0)
            & (torch.abs(local_xy[..., 0]) < 50.0)
            & (torch.abs(local_xy[..., 1]) < 50.0)
        )

        rel_w = pillar_pos_w[..., :2] - robot.data.root_pos_w[:, None, :2]
        rel_forward = torch.sum(rel_w * forward_axis_w[:, None, :], dim=-1)
        rel_left = torch.sum(rel_w * left_axis_w[:, None, :], dim=-1)
        distance = torch.norm(torch.stack((rel_forward, rel_left), dim=-1), dim=-1)
        valid = visible & (rel_forward > -0.05) & (rel_forward < max_forward_distance)

        rel_forward_values.append(rel_forward)
        rel_left_values.append(rel_left)
        color_values.append(torch.full_like(rel_forward, color_value))
        distance_values.append(torch.where(valid, distance, torch.full_like(distance, float("inf"))))

    rel_forward_all = torch.cat(rel_forward_values, dim=1)
    rel_left_all = torch.cat(rel_left_values, dim=1)
    color_all = torch.cat(color_values, dim=1)
    distance_all = torch.cat(distance_values, dim=1)

    nearest_distance, nearest_index = torch.min(distance_all, dim=1)
    has_pillar = torch.isfinite(nearest_distance)
    batch_index = torch.arange(num_envs, device=device)

    rel_forward = rel_forward_all[batch_index, nearest_index]
    rel_left = rel_left_all[batch_index, nearest_index]
    color = color_all[batch_index, nearest_index]

    rel_forward = torch.where(has_pillar, rel_forward, torch.full_like(rel_forward, max_forward_distance))
    rel_left = torch.where(has_pillar, rel_left, torch.zeros_like(rel_left))
    color = torch.where(has_pillar, color, torch.zeros_like(color))
    visible = has_pillar.to(dtype=rel_forward.dtype)

    return torch.stack((rel_forward, rel_left, color, visible), dim=1)
