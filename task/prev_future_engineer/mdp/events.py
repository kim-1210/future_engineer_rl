from __future__ import annotations

import os
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

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


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

def obj_random_pos(
    scene,
    num_envs: int,
    red_k: int,
    green_k: int,
    car_xy: torch.Tensor | None = None,
):
    device = scene["red_obstacles"].device

    # 벽 위치 기준값
    OUTER_LIMIT = 1.55
    INNER_LIMIT = 0.6
    
    # 장애물 사이즈
    OBSTACLE_SIZE = 0.05
    OBSTACLE_HALF = OBSTACLE_SIZE / 2

    # 장애물과 벽과의 거리
    WALL_CLEARANCE = 0.3
    # 장애물 끼리의 최소 거리
    OBSTACLE_MIN_DISTANCE = 0.5
    # 차량 중심 기준 장애물 금지 반경: 400mm
    CAR_MIN_DISTANCE = 0.4

    outer_limit = (
        OUTER_LIMIT
        - OBSTACLE_HALF
        - WALL_CLEARANCE
    )

    inner_limit = (
        INNER_LIMIT
        + OBSTACLE_HALF
        + WALL_CLEARANCE
    )

    # 결과
    # red_pose
    # (num_envs, red_k, 7)
    #
    # green_pose
    # (num_envs, green_k, 7)

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

    min_distance_sq = OBSTACLE_MIN_DISTANCE * OBSTACLE_MIN_DISTANCE
    car_min_distance_sq = CAR_MIN_DISTANCE * CAR_MIN_DISTANCE

    # 환경별로 배치
    for env_id in range(num_envs):
        # 현재까지 배치된 모든 장애물 XY
        placed_xy = []
        # 빨간 장애물
        for obstacle_id in range(red_k):
            while True:
                x = (
                    torch.rand(1, device=device).item() * 2 - 1
                ) * outer_limit
                y = (
                    torch.rand(1, device=device).item() * 2 - 1
                ) * outer_limit

                # 내부벽 검사
                inside_inner = (
                    abs(x) < inner_limit
                    and
                    abs(y) < inner_limit
                )

                if inside_inner:
                    continue

                # 차량 주변 400mm 이내 금지
                if car_xy is not None:
                    dx = x - car_xy[env_id, 0].item()
                    dy = y - car_xy[env_id, 1].item()

                    if dx * dx + dy * dy < car_min_distance_sq:
                        continue

                # 기존 장애물과 겹침 검사
                overlap = False
                for other_x, other_y in placed_xy:
                    dx = x - other_x
                    dy = y - other_y

                    if dx * dx + dy * dy < min_distance_sq:
                        overlap = True
                        break
                if overlap:
                    continue

                # 위치 확정
                red_pose[
                    env_id,
                    obstacle_id,
                    0
                ] = x
                red_pose[
                    env_id,
                    obstacle_id,
                    1
                ] = y
                placed_xy.append(
                    (x, y)
                )

                break

        # 초록 장애물
        for obstacle_id in range(green_k):
            while True:
                x = (
                    torch.rand(1, device=device).item() * 2 - 1
                ) * outer_limit

                y = (
                    torch.rand(1, device=device).item() * 2 - 1
                ) * outer_limit

                # 내부벽 검사
                inside_inner = (
                    abs(x) < inner_limit
                    and
                    abs(y) < inner_limit
                )

                if inside_inner:
                    continue

                # 차량 주변 400mm 이내 금지
                if car_xy is not None:
                    dx = x - car_xy[env_id, 0].item()
                    dy = y - car_xy[env_id, 1].item()

                    if dx * dx + dy * dy < car_min_distance_sq:
                        continue

                # 모든 기존 장애물 검사
                overlap = False
                for other_x, other_y in placed_xy:
                    dx = x - other_x
                    dy = y - other_y

                    if dx * dx + dy * dy < min_distance_sq:
                        overlap = True
                        break

                if overlap:
                    continue

                # 위치 확정
                green_pose[
                    env_id,
                    obstacle_id,
                    0
                ] = x

                green_pose[
                    env_id,
                    obstacle_id,
                    1
                ] = y

                placed_xy.append(
                    (x, y)
                )

                break

    return red_pose, green_pose

def _curriculum_step(env) -> int | None:
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

def _track_direction_curriculum(
    env,
    num_envs: int,
    device: torch.device,
) -> torch.Tensor:
    step = _curriculum_step(env)
    if step is None:
        return torch.where(
            torch.rand(num_envs, device=device) < 0.5,
            -torch.ones(num_envs, device=device),
            torch.ones(num_envs, device=device),
        )

    # _track_tangent 기준으로 +1은 반시계방향, -1은 시계방향이다.
    # 기본 운전이 먼저 안정된 뒤 반대 방향을 천천히 섞는다.
    counter_clockwise = torch.ones(num_envs, device=device)
    clockwise = -torch.ones(num_envs, device=device)
    sample = torch.rand(num_envs, device=device)

    if step < 120_000: # 1875
        # 먼저 한 방향에서 벽에 박지 않고 도는 기본 운전을 만든다.
        return counter_clockwise
    if step < 240_000: # 3750
        # 장애물 전에 반대 방향을 천천히 섞는다.
        return torch.where(sample < 0.3, clockwise, counter_clockwise)
    if step < 480_000: # 7500
        # 빨간 기둥이 처음 들어오면 방향 난이도는 다시 낮춘다.
        return torch.where(sample < 0.2, clockwise, counter_clockwise)
    if step < 720_000: # 11250
        return torch.where(sample < 0.4, clockwise, counter_clockwise)

    # 시계방향 50%, 반시계방향 50%
    return torch.where(sample < 0.5, clockwise, counter_clockwise)

def _obstacle_curriculum_counts(
    env,
    num_envs: int,
    max_red: int,
    max_green: int,
    device: torch.device,
) -> tuple[int, torch.Tensor, torch.Tensor]:
    step = _curriculum_step(env)

    # view_scene.py 같은 확인용 환경은 step 카운터가 없으므로 전체 랜덤 배치를 보여준다.
    if step is None:
        stage = 5
    elif step < 240_000: # 3750
        stage = 1
    elif step < 480_000: # 7500
        stage = 2
    elif step < 720_000: # 11250
        stage = 3
    elif step < 1_000_000: # 5000
        stage = 4
    else:
        stage = 5

    def zeros() -> torch.Tensor:
        return torch.zeros(num_envs, device=device, dtype=torch.long)

    def ones(max_count: int) -> torch.Tensor:
        if max_count <= 0:
            return zeros()
        return torch.ones(num_envs, device=device, dtype=torch.long)

    def randint_one_to(max_count: int) -> torch.Tensor:
        if max_count <= 0:
            return zeros()
        return torch.randint(1, max_count + 1, (num_envs,), device=device)

    if stage == 1:
        red_counts = zeros()
        green_counts = zeros()
    elif stage == 2:
        red_counts = ones(max_red)
        green_counts = zeros()
    elif stage == 3:
        red_counts = ones(max_red)
        green_counts = ones(max_green)
    elif stage == 4:
        red_counts = randint_one_to(min(max_red, 2))
        green_counts = randint_one_to(min(max_green, 2))
    else:
        red_counts = randint_one_to(max_red)
        green_counts = randint_one_to(max_green)

    return stage, red_counts, green_counts


def reset_scene(
    env,
    env_ids,
):
    red_objects = env.scene["red_obstacles"]
    green_objects = env.scene["green_obstacles"]
    robot = env.scene["robot"]

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=red_objects.device)
    else:
        env_ids = env_ids.to(device=red_objects.device, dtype=torch.long)
    num_reset_envs = len(env_ids)

    max_red = red_objects.data.default_object_state.shape[1]
    max_green = green_objects.data.default_object_state.shape[1]
    _, red_counts, green_counts = _obstacle_curriculum_counts(
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

    # 랜덤 위치 생성
    red_pose, green_pose = obj_random_pos(
        env.scene,
        num_envs=num_reset_envs,
        red_k=max_red,
        green_k=max_green,
        car_xy=car_xy,
    )

    active_red = torch.arange(max_red, device=red_objects.device).unsqueeze(0) < red_counts.unsqueeze(1)
    active_green = torch.arange(max_green, device=green_objects.device).unsqueeze(0) < green_counts.unsqueeze(1)

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
    
    track_direction = _track_direction_curriculum(
        env,
        num_reset_envs,
        robot_device,
    )
    direction_state[env_ids] = track_direction

    track_center = (1.55 + 0.60) * 0.5
    corner_margin = position_noise[1]
    lateral_noise = position_noise[0]
    side = torch.randint(0, 4, (num_reset_envs,), device=robot_device)
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
