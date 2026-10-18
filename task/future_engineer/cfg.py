"""
Stage 1: 장애물 없음, CCW
Stage 2: 기둥 1개, 빨강/초록 50:50, CCW
Stage 3: 빨강 1개 + 초록 1개, CCW
Stage 4: 총 기둥 1~3개, 빨강/초록 혼합, CCW
Stage 5: 한 변 최대 2개, 총 6~8개 목표(배치 제약에 따라 감소), 빨강/초록 혼합, CCW
Stage 6: 한 변 최대 2개, 총 6~8개 목표(배치 제약에 따라 감소), 빨강/초록 혼합, CW/CCW 50:50
"""

from dataclasses import MISSING

import torch 

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, DeformableObjectCfg, RigidObjectCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg

from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewardTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import SceneEntityCfg

from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg

from isaaclab.sensors import CameraCfg, ImuCfg, MultiMeshRayCasterCfg, patterns
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import (
    FrameTransformerCfg,
    OffsetCfg
)
from isaaclab.scene import InteractiveSceneCfg

from robot.future_engineer_car import FUTURE_ENGINEER_CAR_CFG

from . import mdp
from .scene import *

# 환경 셋팅
@configclass
class FutureGineerEnv(InteractiveSceneCfg):
    # global
    ground = AssetBaseCfg(
        prim_path="/World/Ground",
        spawn=GroundPlaneCfg(
            color=(1.0, 1.0, 1.0)
        )
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0)
    )

    # local
    flatten = wall_entity(
        prim_path= "{ENV_REGEX_NS}/Flatten",
        size= (3.2, 3.2, 0.01),
        init_pos= (0.0, 0.0, 0.0),
        color= (1.0, 1.0, 1.0)
    )

    # 외벽
    out_wall_top = wall_entity(
        prim_path= "{ENV_REGEX_NS}/OutWallTop",
        size= (3.2, 0.05, DEFAULT_Z),
        init_pos= (0.0, 1.575, DEFAULT_Z / 2),
        color= (0.0, 0.0, 0.0)
    )
    out_wall_bottom = wall_entity(
        prim_path= "{ENV_REGEX_NS}/OutWallBottom",
        size= (3.2, 0.05, DEFAULT_Z),
        init_pos= (0.0, -1.575, DEFAULT_Z / 2),
        color= (0.0, 0.0, 0.0)
    )
    out_wall_left = wall_entity(
        prim_path= "{ENV_REGEX_NS}/OutWallLeft",
        size= (0.05, 3.1, DEFAULT_Z),
        init_pos= (1.575, 0.0, DEFAULT_Z / 2),
        color= (0.0, 0.0, 0.0)
    )
    out_wall_right = wall_entity(
        prim_path= "{ENV_REGEX_NS}/OutWallRight",
        size= (0.05, 3.1, DEFAULT_Z),
        init_pos= (-1.575, 0.0, DEFAULT_Z / 2),
        color= (0.0, 0.0, 0.0)
    )

    # 내벽
    in_wall_top = wall_entity(
        prim_path= "{ENV_REGEX_NS}/InWallTop",
        size= (1.1, 0.05, DEFAULT_Z),
        init_pos= (0.0, 0.525, DEFAULT_Z / 2),
        color= (0.0, 0.0, 0.0)
    )
    in_wall_bottom = wall_entity(
        prim_path= "{ENV_REGEX_NS}/InWallBottom",
        size= (1.1, 0.05, DEFAULT_Z),
        init_pos= (0.0, -0.525, DEFAULT_Z / 2),
        color= (0.0, 0.0, 0.0)
    )
    in_wall_left = wall_entity(
        prim_path= "{ENV_REGEX_NS}/InWallLeft",
        size= (0.05, 1.0, DEFAULT_Z),
        init_pos= (0.525, 0.0, DEFAULT_Z / 2),
        color= (0.0, 0.0, 0.0)
    )
    in_wall_right = wall_entity(
        prim_path= "{ENV_REGEX_NS}/InWallRight",
        size= (0.05, 1.0, DEFAULT_Z),
        init_pos= (-0.525, 0.0, DEFAULT_Z / 2),
        color= (0.0, 0.0, 0.0)
    )

    # 코너 라인: 두께 20mm, 안쪽 사각형 모서리를 향하는 대각선
    top_left_orange_line = line_entity(
        prim_path="{ENV_REGEX_NS}/TopLeftOrangeLine",
        start_xy=(-OUTER_LINE_EDGE, LINE_SIDE_Y),
        end_xy=(-INNER_LINE_CORNER, INNER_LINE_CORNER),
        color=ORANGE_LINE_COLOR,
    )
    top_right_orange_line = line_entity(
        prim_path="{ENV_REGEX_NS}/TopRightOrangeLine",
        start_xy=(LINE_TOP_X, OUTER_LINE_EDGE),
        end_xy=(INNER_LINE_CORNER, INNER_LINE_CORNER),
        color=ORANGE_LINE_COLOR,
    )
    bottom_left_orange_line = line_entity(
        prim_path="{ENV_REGEX_NS}/BottomLeftOrangeLine",
        start_xy=(-LINE_TOP_X, -OUTER_LINE_EDGE),
        end_xy=(-INNER_LINE_CORNER, -INNER_LINE_CORNER),
        color=ORANGE_LINE_COLOR,
    )
    bottom_right_orange_line = line_entity(
        prim_path="{ENV_REGEX_NS}/BottomRightOrangeLine",
        start_xy=(OUTER_LINE_EDGE, -LINE_SIDE_Y),
        end_xy=(INNER_LINE_CORNER, -INNER_LINE_CORNER),
        color=ORANGE_LINE_COLOR,
    )
    top_left_blue_line = line_entity(
        prim_path="{ENV_REGEX_NS}/TopLeftBlueLine",
        start_xy=(-LINE_TOP_X, OUTER_LINE_EDGE),
        end_xy=(-INNER_LINE_CORNER, INNER_LINE_CORNER),
        color=BLUE_LINE_COLOR,
    )
    top_right_blue_line = line_entity(
        prim_path="{ENV_REGEX_NS}/TopRightBlueLine",
        start_xy=(OUTER_LINE_EDGE, LINE_SIDE_Y),
        end_xy=(INNER_LINE_CORNER, INNER_LINE_CORNER),
        color=BLUE_LINE_COLOR,
    )
    bottom_left_blue_line = line_entity(
        prim_path="{ENV_REGEX_NS}/BottomLeftBlueLine",
        start_xy=(-OUTER_LINE_EDGE, -LINE_SIDE_Y),
        end_xy=(-INNER_LINE_CORNER, -INNER_LINE_CORNER),
        color=BLUE_LINE_COLOR,
    )
    bottom_right_blue_line = line_entity(
        prim_path="{ENV_REGEX_NS}/BottomRightBlueLine",
        start_xy=(LINE_TOP_X, -OUTER_LINE_EDGE),
        end_xy=(INNER_LINE_CORNER, -INNER_LINE_CORNER),
        color=BLUE_LINE_COLOR,
    )

    # 장애물
    red_obstacles = RigidObjectCollectionCfg(
        rigid_objects={
            f"red_{i}": obstacle_entity(
                prim_path=f"{{ENV_REGEX_NS}}/RedObstacle_{i}",
                color=(
                    238 / 255,
                    39 / 255,
                    55 / 255,
                ),
            )
            for i in range(MAX_OBSTACLE)
        }
    )
    green_obstacles = RigidObjectCollectionCfg(
        rigid_objects={
            f"green_{i}": obstacle_entity(
                prim_path=f"{{ENV_REGEX_NS}}/GreenObstacle_{i}",
                color=(
                    68 / 255,
                    214 / 255,
                    44 / 255,
                ),
            )
            for i in range(MAX_OBSTACLE)
        }
    )

    robot = FUTURE_ENGINEER_CAR_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Car",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(1.075, 0.0, 0.055),
            rot=(0.7071, 0.0, 0.0, 0.7071),
        ),
    )

    front_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Car/base_link/front_camera",
        update_period=0.0,
        height=64,
        width=64,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=12.0,
            clipping_range=(0.01, 2.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.06, 0.0, 0.06),
            rot=(1.0, 0.0, 0.0, 0.0),
            convention="world",
        ),
    )

    imu = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Car/base_link",
        update_period=0.0,
        offset=ImuCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.025),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        gravity_bias=(0.0, 0.0, 9.81),
    )

    front_tof = MultiMeshRayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Car/base_link",
        update_period=0.0,
        max_distance=0.6,
        ray_alignment="base",
        offset=MultiMeshRayCasterCfg.OffsetCfg(
            pos=(0.12, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        pattern_cfg=patterns.GridPatternCfg(
            resolution=0.02,
            size=(0.0, 0.4),
            direction=(1.0, 0.0, 0.0),
        ),
        mesh_prim_paths=avoid_target_obj(),
        debug_vis=False,
    )

    left_tof = MultiMeshRayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Car/base_link",
        update_period=0.0,
        max_distance=0.5,
        ray_alignment="base",
        offset=MultiMeshRayCasterCfg.OffsetCfg(
            pos=(0.02, 0.065, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        pattern_cfg=patterns.GridPatternCfg(
            resolution=0.02,
            size=(0.2, 0.0),
            direction=(0.0, 1.0, 0.0),
        ),
        mesh_prim_paths=avoid_target_obj(),
        debug_vis=False,
    )

    right_tof = MultiMeshRayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Car/base_link",
        update_period=0.0, # 매 step마다 갱신
        max_distance=0.5, # 최대 감지 거리 (1m)
        ray_alignment="base", # ray방향이 따라갈 물체 
        offset=MultiMeshRayCasterCfg.OffsetCfg(
            pos=(0.02, -0.065, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        # ray를 어떻게 쓸지 
        pattern_cfg=patterns.GridPatternCfg(
            resolution=0.02, # ray 사이 간격 (2cm)
            size=(0.2, 0.0), # ray를 cm폭으로 퍼트릴지 (x방향으로 10cm 폭)
            direction=(0.0, -1.0, 0.0), # ray를 쏘는 방향
        ),
        mesh_prim_paths=avoid_target_obj(), # 검사할 물체
        debug_vis=False,
    )

# 환경에서 reset 및 mode관련 수정 
@configclass
class EventCfg:
    # 리셋시 로봇의 상태(위치, 방향 등)
    reset_robot = EventTerm(
        func=mdp.reset_robot_state,
        mode="reset",
    )
    # 리셋시 장애물들의 상태(빨간색, 초록색 -> 개수)
    reset_obstacles = EventTerm(
        func=mdp.reset_scene,
        mode="reset",
    )

@configclass
class CommandCfg:
    MISSING

# stage만 관리한다. reward weight는 stage가 올라가도 바꾸지 않는다.
@configclass
class CurriculumCfg:
    stage = CurrTerm(
        func=mdp.reward_weight_curriculum,
    )

@configclass
class TerminationCfg:
    time_out = DoneTerm(
        func=mdp.time_out,
        time_out=True,
    )

    pillar_collision = DoneTerm(
        func=mdp.obstacle_collision_termination,
        params={
            "obstacle_half_size": 0.025,
            "car_half_length": 0.12,
            "car_half_width": 0.085,
        },
    )

    wall_collision = DoneTerm(
        func=mdp.wall_collision_termination,
        params={
            "car_half_length": 0.12,
            "car_half_width": 0.085,
        },
    )

    wrong_pillar_pass = DoneTerm(
        func=mdp.wrong_pillar_pass_termination,
        params={
            "pass_margin": 0.02,
            "side_margin": 0.13,
            "max_lateral_distance": 0.45,
            "red_correct_left_side": False,
            "green_correct_left_side": True,
        },
    )

    missed_pillar_pass = DoneTerm(
        func=mdp.missed_pillar_pass_termination,
        params={
            "pass_margin": 0.02,
            "side_margin": 0.13,
            "max_lateral_distance": 0.45,
            "red_correct_left_side": False,
            "green_correct_left_side": True,
        },
    )

    no_progress = DoneTerm(
        func=mdp.no_progress_termination,
        params={
            "min_progress": 0.0004,
            "min_move_distance": 0.0002,
            "max_stale_steps": 75,
            "grace_steps": 90,
        },
    )

    task_success = DoneTerm(
        func=mdp.task_success_termination,
        params={
            "required_laps": 1,
            "obstacle_half_size": 0.025,
            "car_half_length": 0.12,
            "car_half_width": 0.085,
            "pass_margin": 0.02,
            "side_margin": 0.13,
            "max_lateral_distance": 0.45,
            "red_correct_left_side": False,
            "green_correct_left_side": True,
        },
    )

# 강화학습시 ai에게 줄 input값
@configclass
class ObservationCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # tof value
        front_distance = ObsTerm(
            func=mdp.front_tof_bins,
            params={
                "sensor_name": "front_tof",
                "max_distance": 0.6,
            },
        )
        left_distance = ObsTerm(
            func=mdp.side_tof_bins,
            params={
                "sensor_name": "left_tof",
                "max_distance": 0.5,
            },
        )
        right_distance = ObsTerm(
            func=mdp.side_tof_bins,
            params={
                "sensor_name": "right_tof",
                "max_distance": 0.5,
            },
        )
        # simulator-provided nearest pillar: car-frame x, car-frame y, color(red=1, green=-1), visible
        nearest_pillar = ObsTerm(
            func=mdp.nearest_pillar_state,
        )
        # imu값
        yaw_rate = ObsTerm(
            func=mdp.yaw_rate,
            params={"sensor_name": "imu"},
        )
        # 차량 속도
        speed = ObsTerm(
            func=mdp.speed,
        )
        direction = ObsTerm(
            func=mdp.track_direction,
        )
        lateral_error = ObsTerm(
            func=mdp.track_lateral_error,
        )
        heading_alignment = ObsTerm(
            func=mdp.heading_alignment,
        )
        current_steering = ObsTerm(
            func=mdp.current_steering_angle,
            params={"max_steering_angle": 0.5236},
        )
        previous_actions = ObsTerm(
            func=mdp.previous_actions,
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
    policy: PolicyCfg = PolicyCfg()

# 로봇의 행동 값
@configclass
class ActionCfg:
    drive = mdp.CarDriveActionCfg(
        asset_name="robot",
        # 조향이름
        left_steering_joint_name="front_left_steer_joint",
        right_steering_joint_name="front_right_steer_joint",
        # 바퀴 이름
        wheel_joint_names=[
            "rear_left_wheel_joint",
            "rear_right_wheel_joint",
        ],
        # 바퀴 속도
        wheel_velocity_signs=[
            1.0,
            1.0,
        ],
        # 정책이 내는 가상 bicycle steering 각도. 25도면 안쪽 실제 조향이 약 30도 안에 들어온다.
        steering_scale=0.437,
        # 실제 steering joint 최대각. 가상 steering보다 크게 둬야 Ackermann 안쪽 바퀴가 잘리지 않는다.
        max_steering_joint_angle=0.5236,
        left_steering_sign=1.0,
        right_steering_sign=1.0,
        # 최대 바퀴 속도(rad/s). 6.0 rad/s는 바퀴 반지름 0.03m 기준 약 0.18m/s.
        speed_scale=6.0,
        # 최소 바퀴 속도(rad/s). 0이면 벽/기둥 앞에서 멈추는 선택지가 생긴다.
        min_speed=0.0,
        use_ackermann=True,
        wheel_base=0.16,
        track_width=0.13,
    )

# 행동을 할 시 보상 값
@configclass
class RewardCfg:
    # 트랙 진행 방향으로 앞으로 가는지
    track_progress_reward = RewardTerm(
        func=mdp.track_progress_reward,
        weight=1.0,
    )

    # 벽에 가까워지는지
    wall_close_penalty = RewardTerm(
        func=mdp.wall_close_penalty,
        weight=0.0,
    )

    # 기둥과 특정 간격을 맞추지 않고, 진행 방향 기준 올바른 쪽에 있는지만 본다.
    correct_side_reward = RewardTerm(
        func=mdp.correct_side_reward,
        weight=1.0,
        params={
            "lookahead_distance": 0.55,
            "pass_margin": 0.02,
            "side_margin": 0.13,
            "red_correct_left_side": False,
            "green_correct_left_side": True,
        },
    )

    # 규칙대로 기둥을 완전히 통과했는지
    correct_pillar_pass = RewardTerm(
        func=mdp.correct_pillar_pass_reward,
        weight=300.0,
        params={
            "pass_margin": 0.02,
            "side_margin": 0.13,
            "max_lateral_distance": 0.45,
            "red_correct_left_side": False,
            "green_correct_left_side": True,
        },
    )

    # 아래 실패 보상은 termination 결과를 그대로 사용한다.
    wrong_pillar_pass = RewardTerm(
        func=mdp.is_terminated_term,
        weight=-300.0,
        params={"term_keys": ["wrong_pillar_pass"]},
    )

    missed_pillar_pass = RewardTerm(
        func=mdp.is_terminated_term,
        weight=-300.0,
        params={"term_keys": ["missed_pillar_pass"]},
    )

    pillar_collision = RewardTerm(
        func=mdp.is_terminated_term,
        weight=-300.0,
        params={"term_keys": ["pillar_collision"]},
    )

    wall_collision = RewardTerm(
        func=mdp.is_terminated_term,
        weight=-300.0,
        params={"term_keys": ["wall_collision"]},
    )

    no_progress = RewardTerm(
        func=mdp.is_terminated_term,
        weight=-150.0,
        params={"term_keys": ["no_progress"]},
    )

# scene과 curriculum을 뺴고는 마지막에 s를 붙임
@configclass
class FutureEngineerCfg(ManagerBasedRLEnvCfg):
    scene:FutureGineerEnv = FutureGineerEnv()
    events:EventCfg = EventCfg()
    terminations:TerminationCfg = TerminationCfg()
    curriculum:CurriculumCfg = CurriculumCfg()
    actions:ActionCfg = ActionCfg()
    observations:ObservationCfg = ObservationCfg()
    rewards:RewardCfg = RewardCfg()
    commands:CommandCfg = CommandCfg()

    def __post_init__(self,):
        super().__post_init__()

        self.decimation = 2
        self.sim.render_interval = self.decimation
        self.sim.use_fabric = True
        # self.episode_length_s = (150.0)
        self.episode_length_s = 200.0
        # self.sim.dt = 0.01
        self.sim.dt = 1.0 / 60.0
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_max_rigid_patch_count = 2**20
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 64 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625

        self.scene.env_spacing = 4.0
        self.observations.policy.enable_corruption = False

class FutureEngineerEnv(ManagerBasedRLEnv):
    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
