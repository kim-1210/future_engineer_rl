from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import Articulation
from isaaclab.utils.math import quat_apply

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


OUTER_LIMIT = 1.55
INNER_LIMIT = 0.60
TRACK_CENTER = (OUTER_LIMIT + INNER_LIMIT) * 0.5
TRACK_LENGTH = TRACK_CENTER * 8.0
CAR_HALF_LENGTH = 0.12
CAR_HALF_WIDTH = 0.085
OBSTACLE_HALF_SIZE = 0.025


def _robot(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> Articulation:
    return env.scene[asset_cfg.name]


def _local_xy(env, pos_w: torch.Tensor) -> torch.Tensor:
    return pos_w[:, :2] - env.scene.env_origins[:, :2]


def _local_body_xy(env, robot: Articulation) -> torch.Tensor:
    return robot.data.body_pos_w[..., :2] - env.scene.env_origins[:, None, :2]


def _track_direction(env, num_envs: int, device: torch.device) -> torch.Tensor:
    direction = getattr(env, "_future_engineer_track_direction", None)
    if direction is None or direction.shape[0] != num_envs:
        return torch.ones(num_envs, device=device)
    return direction.to(device=device, dtype=torch.float32)


def _track_tangent(local_xy: torch.Tensor, direction: torch.Tensor | None = None) -> torch.Tensor:
    tangent = torch.zeros_like(local_xy)
    right_side, top_side, left_side, bottom_side = _track_side_masks(local_xy)

    tangent[right_side, 1] = 1.0
    tangent[left_side, 1] = -1.0
    tangent[top_side, 0] = -1.0
    tangent[bottom_side, 0] = 1.0

    if direction is not None:
        while direction.ndim < tangent.ndim:
            direction = direction.unsqueeze(-1)
        tangent = tangent * direction

    return tangent


def _track_side_masks(local_xy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x = local_xy[:, 0]
    y = local_xy[:, 1]
    c = TRACK_CENTER

    distances = torch.stack(
        (
            torch.abs(x - c),
            torch.abs(y - c),
            torch.abs(x + c),
            torch.abs(y + c),
        ),
        dim=1,
    )
    side = torch.argmin(distances, dim=1)

    right_side = side == 0
    top_side = side == 1
    left_side = side == 2
    bottom_side = side == 3

    return right_side, top_side, left_side, bottom_side


def _track_progress(local_xy: torch.Tensor, direction: torch.Tensor | None = None) -> torch.Tensor:
    x = local_xy[:, 0]
    y = local_xy[:, 1]
    c = TRACK_CENTER

    right_side, top_side, left_side, bottom_side = _track_side_masks(local_xy)

    progress = torch.zeros_like(x)
    progress[right_side] = y[right_side].clamp(-c, c) + c
    progress[top_side] = 2.0 * c + (c - x[top_side].clamp(-c, c))
    progress[left_side] = 4.0 * c + (c - y[left_side].clamp(-c, c))
    progress[bottom_side] = 6.0 * c + (x[bottom_side].clamp(-c, c) + c)

    if direction is not None:
        progress = torch.where(direction < 0.0, torch.remainder(TRACK_LENGTH - progress, TRACK_LENGTH), progress)

    return progress


def _track_centerline_point(progress: torch.Tensor, direction: torch.Tensor | None = None) -> torch.Tensor:
    c = TRACK_CENTER
    if direction is not None:
        progress = torch.where(direction < 0.0, torch.remainder(TRACK_LENGTH - progress, TRACK_LENGTH), progress)
    progress = torch.remainder(progress, TRACK_LENGTH)

    point = torch.zeros((progress.shape[0], 2), device=progress.device, dtype=progress.dtype)
    right_side = progress < 2.0 * c
    top_side = (progress >= 2.0 * c) & (progress < 4.0 * c)
    left_side = (progress >= 4.0 * c) & (progress < 6.0 * c)
    bottom_side = progress >= 6.0 * c

    point[right_side, 0] = c
    point[right_side, 1] = progress[right_side] - c
    point[top_side, 0] = c - (progress[top_side] - 2.0 * c)
    point[top_side, 1] = c
    point[left_side, 0] = -c
    point[left_side, 1] = c - (progress[left_side] - 4.0 * c)
    point[bottom_side, 0] = progress[bottom_side] - 6.0 * c - c
    point[bottom_side, 1] = -c

    return point


def _track_lookahead_tangent(
    local_xy: torch.Tensor,
    direction: torch.Tensor,
    lookahead_distance: float = 0.16,
) -> torch.Tensor:
    progress = _track_progress(local_xy, direction)
    target_progress = torch.remainder(progress + lookahead_distance, TRACK_LENGTH)
    target_xy = _track_centerline_point(target_progress, direction)
    to_target = target_xy - local_xy
    fallback = _track_tangent(local_xy, direction)
    norm = torch.norm(to_target, dim=1, keepdim=True)

    return torch.where(
        norm > 1.0e-4,
        to_target / torch.clamp(norm, min=1.0e-4),
        fallback,
    )


def _track_lateral_error(local_xy: torch.Tensor) -> torch.Tensor:
    x = local_xy[:, 0]
    y = local_xy[:, 1]
    c = TRACK_CENTER

    distances = torch.stack(
        (
            torch.abs(x - c),
            torch.abs(y - c),
            torch.abs(x + c),
            torch.abs(y + c),
        ),
        dim=1,
    )

    return torch.min(distances, dim=1).values


def _track_signed_lateral_error(
    local_xy: torch.Tensor,
    direction: torch.Tensor | None = None,
) -> torch.Tensor:
    x = local_xy[:, 0]
    y = local_xy[:, 1]
    c = TRACK_CENTER

    right_side, top_side, left_side, bottom_side = _track_side_masks(local_xy)

    centerline_xy = torch.zeros_like(local_xy)
    centerline_xy[right_side, 0] = c
    centerline_xy[right_side, 1] = y[right_side].clamp(-c, c)
    centerline_xy[top_side, 0] = x[top_side].clamp(-c, c)
    centerline_xy[top_side, 1] = c
    centerline_xy[left_side, 0] = -c
    centerline_xy[left_side, 1] = y[left_side].clamp(-c, c)
    centerline_xy[bottom_side, 0] = x[bottom_side].clamp(-c, c)
    centerline_xy[bottom_side, 1] = -c

    tangent = _track_tangent(local_xy, direction)
    left_normal = torch.stack((-tangent[:, 1], tangent[:, 0]), dim=1)
    return torch.sum((local_xy - centerline_xy) * left_normal, dim=1)


def _tof_min_distance(env, sensor_name: str, max_distance: float) -> torch.Tensor:
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

    return torch.min(distance, dim=1).values


def _visible_object_pos(env, collection_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    objects = env.scene[collection_name]
    pos_w = objects.data.object_pos_w
    local_xy = pos_w[..., :2] - env.scene.env_origins[:, None, :2]
    visible = (pos_w[..., 2] > -1.0) & (torch.abs(local_xy[..., 0]) < 50.0) & (torch.abs(local_xy[..., 1]) < 50.0)

    return pos_w[..., :2], visible


def _car_axes_xy(robot: Articulation) -> tuple[torch.Tensor, torch.Tensor]:
    num_envs = robot.data.root_pos_w.shape[0]
    x_axis_b = torch.zeros((num_envs, 3), device=robot.device)
    y_axis_b = torch.zeros_like(x_axis_b)
    x_axis_b[:, 0] = 1.0
    y_axis_b[:, 1] = 1.0

    x_axis_w = quat_apply(robot.data.root_quat_w, x_axis_b)[:, :2]
    y_axis_w = quat_apply(robot.data.root_quat_w, y_axis_b)[:, :2]

    x_axis_w = torch.nn.functional.normalize(x_axis_w, dim=1)
    y_axis_w = torch.nn.functional.normalize(y_axis_w, dim=1)

    return x_axis_w, y_axis_w


def _car_footprint_corners(env, robot: Articulation, half_length: float, half_width: float) -> torch.Tensor:
    center_xy = _local_xy(env, robot.data.root_pos_w)
    x_axis, y_axis = _car_axes_xy(robot)

    signs = torch.tensor(
        (
            (1.0, 1.0),
            (1.0, -1.0),
            (-1.0, 1.0),
            (-1.0, -1.0),
        ),
        device=robot.device,
        dtype=center_xy.dtype,
    )

    return (
        center_xy[:, None, :]
        + signs[None, :, 0:1] * half_length * x_axis[:, None, :]
        + signs[None, :, 1:2] * half_width * y_axis[:, None, :]
    )


def _car_aabb_overlap(
    car_center_xy: torch.Tensor,
    car_x_axis: torch.Tensor,
    car_y_axis: torch.Tensor,
    box_center_xy: torch.Tensor,
    box_half_size: float,
    car_half_length: float,
    car_half_width: float,
) -> torch.Tensor:
    """2-D SAT test between car footprint OBB and axis-aligned square boxes."""
    delta = box_center_xy - car_center_xy[:, None, :]

    car_x = car_x_axis[:, None, :]
    car_y = car_y_axis[:, None, :]
    abs_car_x = torch.abs(car_x)
    abs_car_y = torch.abs(car_y)

    proj_car_x = torch.abs(torch.sum(delta * car_x, dim=-1))
    proj_car_y = torch.abs(torch.sum(delta * car_y, dim=-1))

    box_on_car_x = box_half_size * (abs_car_x[..., 0] + abs_car_x[..., 1])
    box_on_car_y = box_half_size * (abs_car_y[..., 0] + abs_car_y[..., 1])

    sep_car_x = proj_car_x > car_half_length + box_on_car_x
    sep_car_y = proj_car_y > car_half_width + box_on_car_y

    proj_world_x = torch.abs(delta[..., 0])
    proj_world_y = torch.abs(delta[..., 1])

    car_on_world_x = car_half_length * abs_car_x[..., 0] + car_half_width * abs_car_y[..., 0]
    car_on_world_y = car_half_length * abs_car_x[..., 1] + car_half_width * abs_car_y[..., 1]

    sep_world_x = proj_world_x > box_half_size + car_on_world_x
    sep_world_y = proj_world_y > box_half_size + car_on_world_y

    return ~(sep_car_x | sep_car_y | sep_world_x | sep_world_y)


def _episode_just_reset(env, num_envs: int, device: torch.device) -> torch.Tensor:
    episode_length_buf = getattr(
        env,
        "episode_length_buf",
        torch.zeros(num_envs, device=device, dtype=torch.long),
    )
    return episode_length_buf <= 1


def _curriculum_scale(env, start_step: int, end_step: int, device: torch.device) -> torch.Tensor:
    step = getattr(env, "common_step_counter", None)
    if step is None:
        return torch.tensor(1.0, device=device)
    if isinstance(step, torch.Tensor):
        step = int(step.item())

    scale = (int(step) - start_step) / max(end_step - start_step, 1)
    return torch.tensor(min(max(scale, 0.0), 1.0), device=device)


def _pillar_pass_state(
    env,
    collection_name: str,
    event_name: str,
    num_envs: int,
    num_objects: int,
    device: torch.device,
    current_longitudinal: torch.Tensor,
) -> dict[str, torch.Tensor]:
    state_name = f"_future_engineer_{collection_name}_{event_name}_pass_state"
    state = getattr(env, state_name, None)
    if (
        state is None
        or state["scored"].shape != (num_envs, num_objects)
        or "reset_done" not in state
        or "prev_longitudinal" not in state
    ):
        state = {
            "scored": torch.zeros(num_envs, num_objects, device=device, dtype=torch.bool),
            "reset_done": torch.zeros(num_envs, device=device, dtype=torch.bool),
            "prev_longitudinal": current_longitudinal.clone(),
        }
        setattr(env, state_name, state)

    just_reset = _episode_just_reset(env, num_envs, device)
    needs_reset = just_reset & (~state["reset_done"])
    if torch.any(needs_reset):
        state["scored"][needs_reset] = False
        state["prev_longitudinal"][needs_reset] = current_longitudinal[needs_reset]
    state["reset_done"] = torch.where(
        just_reset,
        torch.ones_like(state["reset_done"]),
        torch.zeros_like(state["reset_done"]),
    )

    return state


def _pillar_pass_event(
    env,
    collection_name: str,
    correct_left_side: bool,
    pass_margin: float,
    side_margin: float,
    max_lateral_distance: float,
    want_correct: bool,
) -> torch.Tensor:
    robot = _robot(env)
    car_xy = _local_xy(env, robot.data.root_pos_w)

    pillar_xy_w, visible = _visible_object_pos(env, collection_name)
    pillar_xy = pillar_xy_w - env.scene.env_origins[:, None, :2]
    num_envs, num_objects, _ = pillar_xy.shape

    direction = _track_direction(env, num_envs, pillar_xy.device)
    tangent = _track_tangent(
        pillar_xy.reshape(-1, 2),
        direction[:, None].expand(-1, num_objects).reshape(-1),
    ).reshape(num_envs, num_objects, 2)
    left_normal = torch.stack((-tangent[..., 1], tangent[..., 0]), dim=-1)

    car_from_pillar = car_xy[:, None, :] - pillar_xy
    longitudinal = torch.sum(car_from_pillar * tangent, dim=-1)
    lateral = torch.sum(car_from_pillar * left_normal, dim=-1)

    state = _pillar_pass_state(
        env,
        collection_name,
        event_name="correct" if want_correct else "wrong",
        num_envs=num_envs,
        num_objects=num_objects,
        device=longitudinal.device,
        current_longitudinal=longitudinal,
    )

    crossed_forward = (
        (state["prev_longitudinal"] < longitudinal)
        & (state["prev_longitudinal"] <= pass_margin)
        & (longitudinal >= -pass_margin)
    )
    near_line = torch.abs(lateral) < max_lateral_distance

    if correct_left_side:
        correct_side = lateral > side_margin
        wrong_side = lateral < -side_margin
    else:
        correct_side = lateral < -side_margin
        wrong_side = lateral > side_margin

    side_match = correct_side if want_correct else wrong_side
    candidate = (
        visible
        & crossed_forward
        & near_line
        & side_match
        & (~state["scored"])
    )

    event = candidate & (~state["scored"])
    state["scored"] |= event
    state["prev_longitudinal"] = torch.where(
        visible,
        longitudinal,
        state["prev_longitudinal"],
    )

    return event.any(dim=1).float()


def _safe_to_cruise(
    env,
    front_sensor_name: str = "front_tof",
    left_sensor_name: str = "left_tof",
    right_sensor_name: str = "right_tof",
    front_max_distance: float = 0.6,
    side_max_distance: float = 0.5,
    safe_distance: float = 0.34,
) -> torch.Tensor:
    front_distance = _tof_min_distance(env, front_sensor_name, front_max_distance)
    left_distance = _tof_min_distance(env, left_sensor_name, side_max_distance)
    right_distance = _tof_min_distance(env, right_sensor_name, side_max_distance)
    min_distance = torch.minimum(front_distance, torch.minimum(left_distance, right_distance))

    return min_distance > safe_distance


def _forward_speed_on_track(env: ManagerBasedRLEnv, robot: Articulation) -> torch.Tensor:
    local_xy = _local_xy(env, robot.data.root_pos_w)
    direction = _track_direction(env, robot.num_instances, robot.device)
    tangent = _track_lookahead_tangent(local_xy, direction)

    return torch.sum(robot.data.root_lin_vel_w[:, :2] * tangent, dim=1)


def progress_delta_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    safe_target_delta: float = 0.003,
    caution_target_delta: float = 0.0010,
) -> torch.Tensor:
    robot = _robot(env, asset_cfg)
    local_xy = _local_xy(env, robot.data.root_pos_w)
    direction = _track_direction(env, robot.num_instances, robot.device)
    progress = _track_progress(local_xy, direction)
    num_envs = progress.shape[0]
    device = progress.device

    state = getattr(env, "_future_engineer_progress_delta_reward_state", None)
    if state is None or state["prev_progress"].shape[0] != num_envs:
        state = {"prev_progress": progress.clone()}
        env._future_engineer_progress_delta_reward_state = state
        return torch.zeros(num_envs, device=device)

    delta = progress - state["prev_progress"]
    delta = torch.where(delta < -TRACK_LENGTH * 0.5, delta + TRACK_LENGTH, delta)
    delta = torch.where(delta > TRACK_LENGTH * 0.5, delta - TRACK_LENGTH, delta)
    delta = torch.where(_episode_just_reset(env, num_envs, device), torch.zeros_like(delta), delta)
    state["prev_progress"] = progress.clone()

    safe = _safe_to_cruise(env)
    target_delta = torch.where(
        safe,
        torch.full_like(delta, safe_target_delta),
        torch.full_like(delta, caution_target_delta),
    )

    return torch.clamp(delta / torch.clamp(target_delta, min=1.0e-4), 0.0, 1.0)


def centerline_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_lateral_error: float = 0.30,
) -> torch.Tensor:
    robot = _robot(env, asset_cfg)
    lateral_error = torch.abs(_track_signed_lateral_error(_local_xy(env, robot.data.root_pos_w)))
    score = torch.clamp(1.0 - lateral_error / max(max_lateral_error, 1.0e-4), 0.0, 1.0)

    return torch.square(score)


def heading_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot = _robot(env, asset_cfg)
    local_xy = _local_xy(env, robot.data.root_pos_w)
    direction = _track_direction(env, robot.num_instances, robot.device)
    tangent = _track_lookahead_tangent(local_xy, direction)

    forward_axis_b = torch.zeros_like(robot.data.root_pos_w)
    forward_axis_b[:, 0] = 1.0
    forward_axis_w = quat_apply(robot.data.root_quat_w, forward_axis_b)[:, :2]
    forward_axis_w = torch.nn.functional.normalize(forward_axis_w, dim=1)

    return torch.clamp(torch.sum(forward_axis_w * tangent, dim=1), 0.0, 1.0)


def speed_target_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    safe_target_speed: float = 0.11,
    caution_target_speed: float = 0.05,
    safe_tolerance: float = 0.06,
    caution_tolerance: float = 0.04,
) -> torch.Tensor:
    robot = _robot(env, asset_cfg)
    forward_speed = _forward_speed_on_track(env, robot)
    safe = _safe_to_cruise(env)
    target_speed = torch.where(
        safe,
        torch.full_like(forward_speed, safe_target_speed),
        torch.full_like(forward_speed, caution_target_speed),
    )
    tolerance = torch.where(
        safe,
        torch.full_like(forward_speed, safe_tolerance),
        torch.full_like(forward_speed, caution_tolerance),
    )

    return torch.clamp(1.0 - torch.abs(forward_speed - target_speed) / torch.clamp(tolerance, min=1.0e-4), 0.0, 1.0)


def progress_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_speed: float = 0.18,
) -> torch.Tensor:
    robot = _robot(env, asset_cfg)
    local_xy = _local_xy(env, robot.data.root_pos_w)
    direction = _track_direction(env, robot.num_instances, robot.device)
    tangent = _track_lookahead_tangent(local_xy, direction)
    forward_speed = torch.sum(robot.data.root_lin_vel_w[:, :2] * tangent, dim=1)

    return torch.clamp(forward_speed / max_speed, 0.0, 1.0)


def track_following_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_lateral_error: float = 0.30,
    safe_min_forward_speed: float = 0.04,
    safe_target_forward_speed: float = 0.13,
    caution_target_forward_speed: float = 0.05,
    caution_tolerance: float = 0.04,
) -> torch.Tensor:
    robot = _robot(env, asset_cfg)
    local_xy = _local_xy(env, robot.data.root_pos_w)
    direction = _track_direction(env, robot.num_instances, robot.device)
    tangent = _track_lookahead_tangent(local_xy, direction)

    lateral_error = torch.abs(_track_signed_lateral_error(local_xy, direction))
    center_score = torch.clamp(1.0 - lateral_error / max_lateral_error, 0.0, 1.0)

    forward_axis_b = torch.zeros_like(robot.data.root_pos_w)
    forward_axis_b[:, 0] = 1.0
    forward_axis_w = quat_apply(robot.data.root_quat_w, forward_axis_b)[:, :2]
    forward_axis_w = torch.nn.functional.normalize(forward_axis_w, dim=1)
    heading_score = torch.clamp(torch.sum(forward_axis_w * tangent, dim=1), 0.0, 1.0)

    forward_speed = torch.sum(robot.data.root_lin_vel_w[:, :2] * tangent, dim=1)
    safe = _safe_to_cruise(env)
    safe_speed_score = torch.clamp(
        (forward_speed - safe_min_forward_speed) / max(safe_target_forward_speed - safe_min_forward_speed, 1.0e-4),
        0.0,
        1.0,
    )
    caution_speed_score = torch.clamp(
        1.0 - torch.abs(forward_speed - caution_target_forward_speed) / max(caution_tolerance, 1.0e-4),
        0.0,
        1.0,
    )
    speed_score = torch.where(safe, safe_speed_score, caution_speed_score)

    return center_score * heading_score * speed_score


def correct_pillar_reward(
    env: ManagerBasedRLEnv,
    pass_margin: float = 0.02,
    side_margin: float = 0.03,
    max_lateral_distance: float = 0.45,
    red_correct_left_side: bool = False,
    green_correct_left_side: bool = True,
) -> torch.Tensor:
    red_score = _pillar_pass_event(
        env,
        "red_obstacles",
        correct_left_side=red_correct_left_side,
        pass_margin=pass_margin,
        side_margin=side_margin,
        max_lateral_distance=max_lateral_distance,
        want_correct=True,
    )
    green_score = _pillar_pass_event(
        env,
        "green_obstacles",
        correct_left_side=green_correct_left_side,
        pass_margin=pass_margin,
        side_margin=side_margin,
        max_lateral_distance=max_lateral_distance,
        want_correct=True,
    )

    return torch.clamp(red_score + green_score, 0.0, 1.0)


def pillar_approach_reward(
    env: ManagerBasedRLEnv,
    lookahead_distance: float = 0.55,
    pass_margin: float = 0.02,
    side_margin: float = 0.03,
    red_correct_left_side: bool = False,
    green_correct_left_side: bool = True,
) -> torch.Tensor:
    robot = _robot(env)
    car_xy = _local_xy(env, robot.data.root_pos_w)
    num_envs = car_xy.shape[0]
    direction = _track_direction(env, num_envs, car_xy.device)

    rewards = []
    for collection_name, correct_left_side in (
        ("red_obstacles", red_correct_left_side),
        ("green_obstacles", green_correct_left_side),
    ):
        pillar_xy_w, visible = _visible_object_pos(env, collection_name)
        pillar_xy = pillar_xy_w - env.scene.env_origins[:, None, :2]
        _, num_objects, _ = pillar_xy.shape

        tangent = _track_tangent(
            pillar_xy.reshape(-1, 2),
            direction[:, None].expand(-1, num_objects).reshape(-1),
        ).reshape(num_envs, num_objects, 2)
        left_normal = torch.stack((-tangent[..., 1], tangent[..., 0]), dim=-1)

        car_from_pillar = car_xy[:, None, :] - pillar_xy
        longitudinal = torch.sum(car_from_pillar * tangent, dim=-1)
        lateral = torch.sum(car_from_pillar * left_normal, dim=-1)

        approaching = (longitudinal >= -lookahead_distance) & (longitudinal <= pass_margin)
        approach_score = torch.clamp(
            (longitudinal + lookahead_distance) / max(lookahead_distance + pass_margin, 1.0e-4),
            0.0,
            1.0,
        )

        signed_side = lateral if correct_left_side else -lateral
        side_score = torch.clamp(signed_side / max(side_margin, 1.0e-4), 0.0, 1.0)

        score = torch.where(visible & approaching, approach_score * side_score, torch.zeros_like(longitudinal))
        rewards.append(score.max(dim=1).values)

    return torch.maximum(rewards[0], rewards[1])


def wall_close_penalty(
    env: ManagerBasedRLEnv,
    front_sensor_name: str = "front_tof",
    left_sensor_name: str = "left_tof",
    right_sensor_name: str = "right_tof",
    front_max_distance: float = 0.6,
    side_max_distance: float = 0.5,
    close_distance: float = 0.34,
    critical_distance: float = 0.12,
) -> torch.Tensor:
    front_distance = _tof_min_distance(env, front_sensor_name, front_max_distance)
    left_distance = _tof_min_distance(env, left_sensor_name, side_max_distance)
    right_distance = _tof_min_distance(env, right_sensor_name, side_max_distance)
    min_distance = torch.minimum(front_distance, torch.minimum(left_distance, right_distance))

    warning = torch.clamp(
        (close_distance - min_distance) / max(close_distance - critical_distance, 1.0e-4),
        0.0,
        1.0,
    )
    critical = torch.clamp(
        (critical_distance - min_distance) / max(critical_distance, 1.0e-4),
        0.0,
        1.0,
    )

    return warning + critical


def front_wall_speed_penalty(
    env: ManagerBasedRLEnv,
    front_sensor_name: str = "front_tof",
    front_max_distance: float = 0.6,
    danger_distance: float = 0.36,
    max_speed: float = 0.18,
) -> torch.Tensor:
    front_distance = _tof_min_distance(env, front_sensor_name, front_max_distance)
    robot = _robot(env)
    speed = torch.norm(robot.data.root_lin_vel_w[:, :2], dim=1)

    danger = torch.clamp(
        (danger_distance - front_distance) / max(danger_distance, 1.0e-4),
        0.0,
        1.0,
    )
    speed_ratio = torch.clamp(speed / max(max_speed, 1.0e-4), 0.0, 1.0)

    return danger * speed_ratio


def steering_smooth_penalty(
    env: ManagerBasedRLEnv,
    steering_action_index: int = 0,
) -> torch.Tensor:
    delta = (
        env.action_manager.action[:, steering_action_index]
        - env.action_manager.prev_action[:, steering_action_index]
    )

    return torch.square(delta) * _curriculum_scale(env, 60_000, 120_000, delta.device)


def steering_speed_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    steering_action_index: int = 0,
    max_speed: float = 0.18,
) -> torch.Tensor:
    robot = _robot(env, asset_cfg)
    speed = torch.norm(robot.data.root_lin_vel_w[:, :2], dim=1)
    speed_ratio = torch.clamp(speed / max(max_speed, 1.0e-4), 0.0, 1.0)
    steering_ratio = torch.clamp(torch.abs(env.action_manager.action[:, steering_action_index]), 0.0, 1.0)

    return steering_ratio * speed_ratio


def reverse_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_reverse_speed: float = 0.5,
) -> torch.Tensor:
    robot = _robot(env, asset_cfg)
    local_xy = _local_xy(env, robot.data.root_pos_w)
    direction = _track_direction(env, robot.num_instances, robot.device)
    tangent = _track_lookahead_tangent(local_xy, direction)
    forward_speed = torch.sum(robot.data.root_lin_vel_w[:, :2] * tangent, dim=1)

    reverse_speed = torch.clamp(-forward_speed, 0.0, max_reverse_speed)

    return reverse_speed / max_reverse_speed

def timeout_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    return env.termination_manager.get_term("time_out").float()


def no_progress_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    return env.termination_manager.get_term("no_progress").float()


def cruise_speed_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    min_speed: float = 0.06,
    target_min_speed: float = 0.10,
    target_max_speed: float = 0.15,
    too_fast_speed: float = 0.18,
) -> torch.Tensor:
    robot = _robot(env, asset_cfg)
    local_xy = _local_xy(env, robot.data.root_pos_w)
    direction = _track_direction(env, robot.num_instances, robot.device)
    tangent = _track_lookahead_tangent(local_xy, direction)
    forward_speed = torch.sum(robot.data.root_lin_vel_w[:, :2] * tangent, dim=1)
    safe = _safe_to_cruise(env)

    accel_score = torch.clamp(
        (forward_speed - min_speed) / max(target_min_speed - min_speed, 1.0e-4),
        0.0,
        1.0,
    )
    too_fast_score = torch.clamp(
        (too_fast_speed - forward_speed) / max(too_fast_speed - target_max_speed, 1.0e-4),
        0.0,
        1.0,
    )
    speed_score = torch.minimum(accel_score, too_fast_score)

    return torch.where(
        safe,
        speed_score,
        torch.zeros_like(forward_speed),
    )


def low_speed_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    safe_min_speed: float = 0.10,
    caution_min_speed: float = 0.02,
) -> torch.Tensor:
    robot = _robot(env, asset_cfg)
    local_xy = _local_xy(env, robot.data.root_pos_w)
    direction = _track_direction(env, robot.num_instances, robot.device)
    tangent = _track_lookahead_tangent(local_xy, direction)
    forward_speed = torch.sum(robot.data.root_lin_vel_w[:, :2] * tangent, dim=1)
    safe = _safe_to_cruise(env)
    required_speed = torch.where(
        safe,
        torch.full_like(forward_speed, safe_min_speed),
        torch.full_like(forward_speed, caution_min_speed),
    )

    penalty = torch.clamp((required_speed - forward_speed) / torch.clamp(required_speed, min=1.0e-4), 0.0, 1.0)
    return penalty


def pillar_collision(
    env: ManagerBasedRLEnv,
    obstacle_half_size: float = OBSTACLE_HALF_SIZE,
    car_half_length: float = CAR_HALF_LENGTH,
    car_half_width: float = CAR_HALF_WIDTH,
) -> torch.Tensor:
    robot = _robot(env)
    car_center_xy = _local_xy(env, robot.data.root_pos_w)
    car_x_axis, car_y_axis = _car_axes_xy(robot)

    hits = []
    for collection_name in ("red_obstacles", "green_obstacles"):
        pillar_xy_w, visible = _visible_object_pos(env, collection_name)
        pillar_xy = pillar_xy_w - env.scene.env_origins[:, None, :2]
        overlap = _car_aabb_overlap(
            car_center_xy=car_center_xy,
            car_x_axis=car_x_axis,
            car_y_axis=car_y_axis,
            box_center_xy=pillar_xy,
            box_half_size=obstacle_half_size,
            car_half_length=car_half_length,
            car_half_width=car_half_width,
        )
        hits.append(overlap & visible)

    return torch.cat(hits, dim=1).any(dim=1).float()


def wall_collision(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    car_half_length: float = CAR_HALF_LENGTH,
    car_half_width: float = CAR_HALF_WIDTH,
) -> torch.Tensor:
    robot = _robot(env, asset_cfg)
    car_center_xy = _local_xy(env, robot.data.root_pos_w)
    car_x_axis, car_y_axis = _car_axes_xy(robot)
    corners = _car_footprint_corners(env, robot, car_half_length, car_half_width)
    abs_corners = torch.abs(corners)

    outside_outer = (abs_corners[..., 0] > OUTER_LIMIT) | (abs_corners[..., 1] > OUTER_LIMIT)
    inner_overlap = _car_aabb_overlap(
        car_center_xy=car_center_xy,
        car_x_axis=car_x_axis,
        car_y_axis=car_y_axis,
        box_center_xy=torch.zeros((car_center_xy.shape[0], 1, 2), device=car_center_xy.device, dtype=car_center_xy.dtype),
        box_half_size=INNER_LIMIT,
        car_half_length=car_half_length,
        car_half_width=car_half_width,
    )

    return (outside_outer.any(dim=1) | inner_overlap[:, 0]).float()


def wrong_pillar_pass(
    env: ManagerBasedRLEnv,
    pass_margin: float = 0.02,
    side_margin: float = 0.03,
    max_lateral_distance: float = 0.45,
    red_correct_left_side: bool = False,
    green_correct_left_side: bool = True,
) -> torch.Tensor:
    red_score = _pillar_pass_event(
        env,
        "red_obstacles",
        correct_left_side=red_correct_left_side,
        pass_margin=pass_margin,
        side_margin=side_margin,
        max_lateral_distance=max_lateral_distance,
        want_correct=False,
    )
    green_score = _pillar_pass_event(
        env,
        "green_obstacles",
        correct_left_side=green_correct_left_side,
        pass_margin=pass_margin,
        side_margin=side_margin,
        max_lateral_distance=max_lateral_distance,
        want_correct=False,
    )

    return torch.clamp(red_score + green_score, 0.0, 1.0)


def _lap_state(
    env,
    device: torch.device,
    num_envs: int,
    current_progress: torch.Tensor,
) -> dict[str, torch.Tensor]:
    state = getattr(env, "_future_engineer_lap_reward_state", None)
    if (
        state is None
        or state["prev_progress"].shape[0] != num_envs
        or "lap_progress" not in state
        or "reset_done" not in state
    ):
        state = {
            "prev_progress": current_progress.clone(),
            "lap_progress": torch.zeros(num_envs, device=device),
            "lap_count": torch.zeros(num_envs, device=device, dtype=torch.long),
            "reset_done": torch.zeros(num_envs, device=device, dtype=torch.bool),
        }
        env._future_engineer_lap_reward_state = state

    just_reset = _episode_just_reset(env, num_envs, device)
    needs_reset = just_reset & (~state["reset_done"])
    if torch.any(needs_reset):
        state["prev_progress"][needs_reset] = current_progress[needs_reset]
        state["lap_progress"][needs_reset] = 0.0
        state["lap_count"][needs_reset] = 0
    state["reset_done"] = torch.where(
        just_reset,
        torch.ones_like(state["reset_done"]),
        torch.zeros_like(state["reset_done"]),
    )

    return state

# 한바퀴를 문제 없이 돌았는지
def lap_complete(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot = _robot(env, asset_cfg)
    local_xy = _local_xy(env, robot.data.root_pos_w)
    direction = _track_direction(env, robot.num_instances, robot.device)
    progress = _track_progress(local_xy, direction)

    state = _lap_state(env, robot.device, robot.num_instances, progress)
    delta = progress - state["prev_progress"]
    delta = torch.where(delta < -TRACK_LENGTH * 0.5, delta + TRACK_LENGTH, delta)
    delta = torch.where(delta > TRACK_LENGTH * 0.5, delta - TRACK_LENGTH, delta)
    lap_progress = torch.clamp(state["lap_progress"] + delta, min=0.0)
    completed = lap_progress >= TRACK_LENGTH
    state["lap_count"] += completed.long()
    state["lap_progress"] = torch.where(completed, lap_progress - TRACK_LENGTH, lap_progress)
    state["prev_progress"] = progress

    return completed.float()
