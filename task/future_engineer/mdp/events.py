from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.utils.math import combine_frame_transforms
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.assets import Articulation

from isaaclab.utils.math import euler_xyz_from_quat, quat_from_euler_xyz, sample_uniform

from .curriculums import curriculum_stage

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


OUTER_LIMIT = 1.55
INNER_LIMIT = 0.55
TRACK_CENTER = (OUTER_LIMIT + INNER_LIMIT) * 0.5
TRAFFIC_SIGN_SEAT_ALONG = (-0.45, 0.45)
TRAFFIC_SIGN_SEAT_ACROSS = (-0.10, 0.10)
MAX_TRAFFIC_SIGNS_PER_SIDE = 2
MAX_TRAFFIC_SIGNS_TOTAL = 8
MIN_TRAFFIC_SIGN_DISTANCE = 0.90
MIN_START_SIGN_DISTANCE = 0.45
MIN_START_FORWARD_SIGN_DISTANCE = 0.70
START_SIGN_FORWARD_HALF_WIDTH = 0.25
SIDE_TANGENTS = (
    (0.0, 1.0),
    (-1.0, 0.0),
    (0.0, -1.0),
    (1.0, 0.0),
)
TRAFFIC_SIGN_SEATS = (
    tuple((TRACK_CENTER + across, along) for across in TRAFFIC_SIGN_SEAT_ACROSS for along in TRAFFIC_SIGN_SEAT_ALONG),
    tuple((along, TRACK_CENTER + across) for across in TRAFFIC_SIGN_SEAT_ACROSS for along in TRAFFIC_SIGN_SEAT_ALONG),
    tuple((-(TRACK_CENTER + across), along) for across in TRAFFIC_SIGN_SEAT_ACROSS for along in TRAFFIC_SIGN_SEAT_ALONG),
    tuple((along, -(TRACK_CENTER + across)) for across in TRAFFIC_SIGN_SEAT_ACROSS for along in TRAFFIC_SIGN_SEAT_ALONG),
)


def _set_usd_obstacle_transforms(
    env_ids: torch.Tensor,
    object_state_local: torch.Tensor,
    prim_prefix: str,
    enabled: bool,
) -> None:
    """Mirror reset positions to USD so UI rendering shows the randomized layout."""
    if not enabled:
        return

    try:
        import omni.usd
        from pxr import Gf, UsdGeom
    except Exception:
        return

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return

    for env_index, env_id in enumerate(env_ids.tolist()):
        for object_id in range(object_state_local.shape[1]):
            prim_path = f"/World/envs/env_{env_id}/{prim_prefix}_{object_id}"
            prim = stage.GetPrimAtPath(prim_path)
            if not prim.IsValid():
                continue

            pos = object_state_local[env_index, object_id, :3]
            xformable = UsdGeom.Xformable(prim)
            translate_op = None
            for op in xformable.GetOrderedXformOps():
                if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                    translate_op = op
                    break
            if translate_op is None:
                translate_op = xformable.AddTranslateOp()

            translate_op.Set(
                Gf.Vec3d(
                    float(pos[0].item()),
                    float(pos[1].item()),
                    float(pos[2].item()),
                )
            )

def _is_cpu_sim(env) -> bool:
    sim = getattr(env, "sim", None)
    sim_cfg = getattr(sim, "cfg", None)
    return getattr(sim_cfg, "device", None) == "cpu"


def _select_spaced_seat_ids(
    candidate_seats: torch.Tensor,
    candidate_side_ids: torch.Tensor,
    candidate_priorities: torch.Tensor,
    side_max_counts: torch.Tensor,
    target_count: int,
) -> list[int]:
    """Pick traffic sign seats while keeping enough space between all signs."""
    random_score = torch.rand(candidate_seats.shape[0], device=candidate_seats.device)
    order = torch.argsort(candidate_priorities.float() + random_score)
    ordered_seats = candidate_seats[order]
    ordered_side_ids = candidate_side_ids[order]
    min_distance_sq = MIN_TRAFFIC_SIGN_DISTANCE * MIN_TRAFFIC_SIGN_DISTANCE

    pair_distance_sq = torch.sum(
        (ordered_seats[:, None, :] - ordered_seats[None, :, :]) ** 2,
        dim=2,
    )
    pair_ok = pair_distance_sq >= min_distance_sq - 1e-6

    best_selected: list[int] = []
    selected: list[int] = []
    side_counts = [0 for _ in range(side_max_counts.shape[0])]
    side_max_counts_list = [int(value.item()) for value in side_max_counts]

    def search(start_index: int) -> bool:
        nonlocal best_selected
        if len(selected) > len(best_selected):
            best_selected = selected.copy()
        if len(selected) >= target_count:
            return True
        if len(selected) + (ordered_seats.shape[0] - start_index) <= len(best_selected):
            return False

        for candidate_index in range(start_index, ordered_seats.shape[0]):
            if len(selected) + (ordered_seats.shape[0] - candidate_index) <= len(best_selected):
                return False

            side_id = int(ordered_side_ids[candidate_index].item())
            if side_counts[side_id] >= side_max_counts_list[side_id]:
                continue
            if selected and not bool(torch.all(pair_ok[candidate_index, selected]).item()):
                continue

            selected.append(candidate_index)
            side_counts[side_id] += 1
            if search(candidate_index + 1):
                return True
            side_counts[side_id] -= 1
            selected.pop()
        return False

    search(0)
    return [int(order[index].item()) for index in best_selected]


def obj_random_pos(
    scene,
    num_envs: int,
    red_k: int,
    green_k: int,
    red_counts: torch.Tensor,
    green_counts: torch.Tensor,
    car_xy: torch.Tensor | None = None,
    car_side: torch.Tensor | None = None,
    track_direction: torch.Tensor | None = None,
):
    device = scene["red_obstacles"].device
    red_pose = torch.zeros(
        (num_envs, red_k, 7),
        device=device,
    )

    green_pose = torch.zeros(
        (num_envs, green_k, 7),
        device=device,
    )

    # z
    red_pose[:, :, 2] = 0.05
    green_pose[:, :, 2] = 0.05

    # quaternion
    red_pose[:, :, 3] = 1.0
    green_pose[:, :, 3] = 1.0

    seat_xy_by_side = torch.tensor(
        TRAFFIC_SIGN_SEATS,
        device=device,
        dtype=red_pose.dtype,
    )

    # 환경별로 배치
    for env_id in range(num_envs):
        active_red_count = int(red_counts[env_id].item())
        active_green_count = int(green_counts[env_id].item())
        active_total_count = active_red_count + active_green_count
        if active_total_count <= 0:
            continue

        candidate_parts = []
        candidate_side_parts = []
        candidate_priority_parts = []
        side_max_counts = torch.full(
            (seat_xy_by_side.shape[0],),
            MAX_TRAFFIC_SIGNS_PER_SIDE,
            device=device,
            dtype=torch.long,
        )
        for side_id in torch.randperm(seat_xy_by_side.shape[0], device=device).tolist():
            side_seats = seat_xy_by_side[side_id]
            if car_xy is None:
                seat_ids = torch.arange(side_seats.shape[0], device=device, dtype=torch.long)
                priorities = torch.zeros(seat_ids.shape[0], device=device, dtype=torch.long)
            else:
                car_xy_env = car_xy[env_id].to(device=device)
                distances = torch.linalg.norm(side_seats - car_xy_env, dim=1)
                safe_mask = distances >= MIN_START_SIGN_DISTANCE
                directly_ahead = torch.zeros_like(safe_mask)

                if car_side is not None and int(car_side[env_id].item()) == side_id:
                    side_tangent = torch.tensor(
                        SIDE_TANGENTS[side_id],
                        device=device,
                        dtype=side_seats.dtype,
                    )
                    direction = 1.0
                    if track_direction is not None:
                        direction = float(track_direction[env_id].item())
                    forward_tangent = side_tangent * direction
                    left_normal = torch.stack((-forward_tangent[1], forward_tangent[0]))
                    delta = side_seats - car_xy_env
                    forward_distance = torch.sum(delta * forward_tangent, dim=1)
                    lateral_distance = torch.abs(torch.sum(delta * left_normal, dim=1))
                    directly_ahead = (
                        (forward_distance >= 0.0)
                        & (forward_distance <= MIN_START_FORWARD_SIGN_DISTANCE)
                        & (lateral_distance <= START_SIGN_FORWARD_HALF_WIDTH)
                    )
                    safe_mask &= ~directly_ahead

                safe_ids = torch.nonzero(safe_mask, as_tuple=False).squeeze(1)
                fallback_ids = torch.nonzero(~safe_mask & ~directly_ahead, as_tuple=False).squeeze(1)
                seat_ids = torch.cat((safe_ids, fallback_ids), dim=0)
                priorities = torch.cat(
                    (
                        torch.zeros(safe_ids.shape[0], device=device, dtype=torch.long),
                        torch.ones(fallback_ids.shape[0], device=device, dtype=torch.long),
                    ),
                    dim=0,
                )
            if seat_ids.numel() > 0:
                candidate_parts.append(side_seats[seat_ids])
                candidate_side_parts.append(torch.full((seat_ids.numel(),), side_id, device=device, dtype=torch.long))
                candidate_priority_parts.append(priorities)

        candidate_seats = torch.cat(candidate_parts, dim=0)
        candidate_side_ids = torch.cat(candidate_side_parts, dim=0)
        candidate_priorities = torch.cat(candidate_priority_parts, dim=0)
        selected_ids = _select_spaced_seat_ids(
            candidate_seats=candidate_seats,
            candidate_side_ids=candidate_side_ids,
            candidate_priorities=candidate_priorities,
            side_max_counts=side_max_counts,
            target_count=active_total_count,
        )
        active_total_count = min(active_total_count, len(selected_ids))
        active_red_count = min(active_red_count, active_total_count)
        active_green_count = active_total_count - active_red_count
        red_counts[env_id] = active_red_count
        green_counts[env_id] = active_green_count

        selected_seats = candidate_seats[
            torch.tensor(selected_ids[:active_total_count], device=device, dtype=torch.long)
        ]
        selected_seats = selected_seats[torch.randperm(selected_seats.shape[0], device=device)]
        if active_red_count > 0:
            red_pose[env_id, :active_red_count, :2] = selected_seats[:active_red_count]
        if active_green_count > 0:
            green_pose[env_id, :active_green_count, :2] = selected_seats[
                active_red_count:active_total_count
            ]

    return red_pose, green_pose

def _track_direction_curriculum(
    env,
    num_envs: int,
    device: torch.device,
) -> torch.Tensor:
    # _track_tangent 기준으로 +1은 반시계방향, -1은 시계방향이다.
    counter_clockwise = torch.ones(num_envs, device=device)
    clockwise = -torch.ones(num_envs, device=device)
    if curriculum_stage(env) < 6:
        return counter_clockwise

    return torch.where(torch.rand(num_envs, device=device) < 0.5, clockwise, counter_clockwise)

def _obstacle_curriculum_counts(
    env,
    num_envs: int,
    max_red: int,
    max_green: int,
    device: torch.device,
) -> tuple[int, torch.Tensor, torch.Tensor]:
    stage = curriculum_stage(env)

    def zeros() -> torch.Tensor:
        return torch.zeros(num_envs, device=device, dtype=torch.long)

    def ones(max_count: int) -> torch.Tensor:
        if max_count <= 0:
            return zeros()
        return torch.ones(num_envs, device=device, dtype=torch.long)

    def split_total_randomly(total_counts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        red_counts = zeros()
        green_counts = zeros()
        max_total_count = int(total_counts.max().item()) if total_counts.numel() > 0 else 0
        for index in range(max_total_count):
            active = index < total_counts
            prefer_red = torch.rand(num_envs, device=device) < 0.5
            red_available = red_counts < max_red
            green_available = green_counts < max_green
            use_red = active & ((prefer_red & red_available) | (~green_available))
            use_green = active & (~use_red) & green_available
            red_counts += use_red.long()
            green_counts += use_green.long()
        return red_counts, green_counts

    if stage == 1:
        red_counts = zeros()
        green_counts = zeros()
    elif stage == 2:
        choose_green = torch.rand(num_envs, device=device) < 0.5
        red_counts = torch.where(choose_green, zeros(), ones(max_red))
        green_counts = torch.where(choose_green, ones(max_green), zeros())
    elif stage == 3:
        red_counts = ones(max_red)
        green_counts = ones(max_green)
    elif stage == 4:
        max_total = min(3, max_red + max_green)
        if max_total <= 0:
            red_counts = zeros()
            green_counts = zeros()
        else:
            total_counts = torch.randint(1, max_total + 1, (num_envs,), device=device)
            red_counts, green_counts = split_total_randomly(total_counts)
    else:
        max_total = min(MAX_TRAFFIC_SIGNS_TOTAL, max_red + max_green)
        if max_total <= 0:
            red_counts = zeros()
            green_counts = zeros()
        else:
            min_total = min(6, max_total)
            total_counts = torch.randint(min_total, max_total + 1, (num_envs,), device=device)
            red_counts, green_counts = split_total_randomly(total_counts)

    return stage, red_counts, green_counts


def reset_scene(
    env,
    env_ids,
):
    red_objects = env.scene["red_obstacles"]
    green_objects = env.scene["green_obstacles"]
    robot = env.scene["robot"]
    object_device = torch.device(red_objects.device)

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=red_objects.device)
    else:
        env_ids = env_ids.to(device=red_objects.device, dtype=torch.long)
    num_reset_envs = len(env_ids)

    max_red = red_objects.data.default_object_state.shape[1]
    max_green = green_objects.data.default_object_state.shape[1]
    stage, red_counts, green_counts = _obstacle_curriculum_counts(
        env=env,
        num_envs=num_reset_envs,
        max_red=max_red,
        max_green=max_green,
        device=red_objects.device,
    )

    # 기본 state
    red_state = red_objects.data.default_object_state[env_ids].clone()
    green_state = green_objects.data.default_object_state[env_ids].clone()

    # 전부 숨김
    hidden_pos = torch.tensor(
        [100.0, 100.0, -5.0],
        device=red_state.device,
        dtype=red_state.dtype,
    )

    red_state[:, :, :3] = hidden_pos
    green_state[:, :, :3] = hidden_pos

    red_state[:, :, 7:] = 0.0
    green_state[:, :, 7:] = 0.0

    # reset_robot_state에서 정한 시작 위치를 우선 사용한다.
    # root_pos_w는 같은 reset tick 안에서 아직 이전 값일 수 있다.
    start_xy_state = getattr(env, "_future_engineer_start_xy", None)
    if start_xy_state is not None and start_xy_state.shape[0] == env.scene.num_envs:
        car_xy = start_xy_state[env_ids].to(device=red_objects.device)
    else:
        car_xy = robot.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2]

    start_side_state = getattr(env, "_future_engineer_start_side", None)
    if start_side_state is not None and start_side_state.shape[0] == env.scene.num_envs:
        car_side = start_side_state[env_ids].to(device=red_objects.device)
    else:
        car_side = None

    direction_state = getattr(env, "_future_engineer_track_direction", None)
    if direction_state is not None and direction_state.shape[0] == env.scene.num_envs:
        track_direction = direction_state[env_ids].to(device=red_objects.device)
    else:
        track_direction = None

    # 랜덤 위치 생성
    red_pose, green_pose = obj_random_pos(
        env.scene,
        num_envs=num_reset_envs,
        red_k=max_red,
        green_k=max_green,
        red_counts=red_counts,
        green_counts=green_counts,
        car_xy=car_xy,
        car_side=car_side,
        track_direction=track_direction,
    )

    active_red = torch.arange(max_red, device=red_objects.device).unsqueeze(0) < red_counts.unsqueeze(1)
    active_green = torch.arange(max_green, device=green_objects.device).unsqueeze(0) < green_counts.unsqueeze(1)

    required_pass_count = getattr(env, "_future_engineer_required_pillar_pass_count", None)
    if (
        required_pass_count is None
        or required_pass_count.shape[0] != env.scene.num_envs
        or required_pass_count.device != object_device
    ):
        required_pass_count = torch.zeros(env.scene.num_envs, device=object_device)
        env._future_engineer_required_pillar_pass_count = required_pass_count
    required_pass_count[env_ids] = (red_counts + green_counts).float()

    for state_name, counts in (
        ("_future_engineer_required_red_pillar_pass_count", red_counts),
        ("_future_engineer_required_green_pillar_pass_count", green_counts),
    ):
        color_required_count = getattr(env, state_name, None)
        if (
            color_required_count is None
            or color_required_count.shape[0] != env.scene.num_envs
            or color_required_count.device != object_device
        ):
            color_required_count = torch.zeros(env.scene.num_envs, device=object_device)
            setattr(env, state_name, color_required_count)
        color_required_count[env_ids] = counts.float()

    stage_state = getattr(env, "_future_engineer_obstacle_stage", None)
    if (
        stage_state is None
        or stage_state.shape[0] != env.scene.num_envs
        or stage_state.device != object_device
    ):
        stage_state = torch.zeros(env.scene.num_envs, device=object_device, dtype=torch.long)
        env._future_engineer_obstacle_stage = stage_state
    stage_state[env_ids] = stage

    episode_stage = getattr(env, "_future_engineer_episode_stage", None)
    if (
        episode_stage is None
        or episode_stage.shape[0] != env.scene.num_envs
        or episode_stage.device != object_device
    ):
        episode_stage = torch.zeros(env.scene.num_envs, device=object_device, dtype=torch.long)
        env._future_engineer_episode_stage = episode_stage
    episode_stage[env_ids] = stage

    curriculum_recorded = getattr(env, "_future_engineer_curriculum_recorded", None)
    if (
        curriculum_recorded is not None
        and curriculum_recorded.shape[0] == env.scene.num_envs
        and curriculum_recorded.device == object_device
    ):
        curriculum_recorded[env_ids] = False

    from .rewards import reset_episode_mdp_state

    reset_episode_mdp_state(env, env_ids)

    red_state[:, :, :7] = torch.where(active_red[:, :, None], red_pose, red_state[:, :, :7])
    green_state[:, :, :7] = torch.where(active_green[:, :, None], green_pose, green_state[:, :, :7])

    # USD Stage는 env prim 아래 local 좌표를 보고, PhysX는 world 좌표를 본다.
    red_state_local = red_state.clone()
    green_state_local = green_state.clone()

    # local 좌표 -> world 좌표
    red_state[:, :, :3] += env.scene.env_origins[env_ids, None, :]
    green_state[:, :, :3] += env.scene.env_origins[env_ids, None, :]

    _set_usd_obstacle_transforms(
        env_ids=env_ids,
        object_state_local=red_state_local,
        prim_prefix="RedObstacle",
        enabled=_is_cpu_sim(env),
    )
    _set_usd_obstacle_transforms(
        env_ids=env_ids,
        object_state_local=green_state_local,
        prim_prefix="GreenObstacle",
        enabled=_is_cpu_sim(env),
    )

    # PhysX에 적용
    red_objects.write_object_pose_to_sim(
        red_state[:, :, :7],
        env_ids=env_ids,
    )
    red_objects.write_object_velocity_to_sim(
        red_state[:, :, 7:],
        env_ids=env_ids,
    )
    green_objects.write_object_pose_to_sim(
        green_state[:, :, :7],
        env_ids=env_ids,
    )
    green_objects.write_object_velocity_to_sim(
        green_state[:, :, 7:],
        env_ids=env_ids,
    )


def reset_robot_state(
    env,
    env_ids,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    position_noise: tuple[float, float] = (0.02, 0.08),
    yaw_noise: float = 0.12,
):
    robot: Articulation = env.scene[asset_cfg.name]
    robot_device = torch.device(robot.device)

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=robot_device)
    else:
        env_ids = env_ids.to(device=robot_device, dtype=torch.long)
    num_reset_envs = len(env_ids)

    root_state = robot.data.default_root_state[env_ids].clone()

    direction_state = getattr(env, "_future_engineer_track_direction", None)
    if (
        direction_state is None
        or direction_state.shape[0] != env.scene.num_envs
        or direction_state.device != robot_device
    ):
        direction_state = torch.ones(env.scene.num_envs, device=robot_device)
        env._future_engineer_track_direction = direction_state

    start_xy_state = getattr(env, "_future_engineer_start_xy", None)
    if (
        start_xy_state is None
        or start_xy_state.shape != (env.scene.num_envs, 2)
        or start_xy_state.device != robot_device
    ):
        start_xy_state = torch.zeros((env.scene.num_envs, 2), device=robot_device)
        env._future_engineer_start_xy = start_xy_state

    start_side_state = getattr(env, "_future_engineer_start_side", None)
    if (
        start_side_state is None
        or start_side_state.shape[0] != env.scene.num_envs
        or start_side_state.device != robot_device
    ):
        start_side_state = torch.zeros(env.scene.num_envs, device=robot_device, dtype=torch.long)
        env._future_engineer_start_side = start_side_state
    
    track_direction = _track_direction_curriculum(
        env,
        num_reset_envs,
        robot_device,
    )
    direction_state[env_ids] = track_direction

    track_center = TRACK_CENTER
    corner_margin = position_noise[1]
    lateral_noise = position_noise[0]
    side = torch.randint(0, 4, (num_reset_envs,), device=robot_device)
    start_side_state[env_ids] = side
    along = sample_uniform(
        -track_center + corner_margin,
        track_center - corner_margin,
        num_reset_envs,
        robot_device,
    )
    lateral = sample_uniform(-lateral_noise, lateral_noise, num_reset_envs, robot_device)

    start_xy = torch.zeros((num_reset_envs, 2), device=robot_device)
    base_tangent = torch.zeros((num_reset_envs, 2), device=robot_device)

    right_side = side == 0
    top_side = side == 1
    left_side = side == 2
    bottom_side = side == 3

    start_xy[right_side, 0] = track_center
    start_xy[right_side, 1] = along[right_side]
    base_tangent[right_side, 1] = 1.0

    start_xy[top_side, 0] = along[top_side]
    start_xy[top_side, 1] = track_center
    base_tangent[top_side, 0] = -1.0

    start_xy[left_side, 0] = -track_center
    start_xy[left_side, 1] = along[left_side]
    base_tangent[left_side, 1] = -1.0

    start_xy[bottom_side, 0] = along[bottom_side]
    start_xy[bottom_side, 1] = -track_center
    base_tangent[bottom_side, 0] = 1.0

    left_normal = torch.stack((-base_tangent[:, 1], base_tangent[:, 0]), dim=1)
    root_state[:, :2] = start_xy + lateral[:, None] * left_normal
    start_xy_state[env_ids] = root_state[:, :2]

    roll, pitch, yaw = euler_xyz_from_quat(root_state[:, 3:7])
    forward_tangent = base_tangent * track_direction[:, None]
    yaw = torch.atan2(forward_tangent[:, 1], forward_tangent[:, 0])
    yaw = yaw + sample_uniform(-yaw_noise, yaw_noise, num_reset_envs, robot_device)
    root_state[:, 3:7] = quat_from_euler_xyz(roll, pitch, yaw)

    root_state[:, :3] += env.scene.env_origins[env_ids]

    joint_pos = robot.data.default_joint_pos[env_ids].clone()
    joint_vel = robot.data.default_joint_vel[env_ids].clone()

    robot.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids)
    robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids=env_ids)
    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    robot.set_joint_position_target(joint_pos, env_ids=env_ids)
    robot.set_joint_velocity_target(joint_vel, env_ids=env_ids)
