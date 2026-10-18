from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg


FUTURE_ENGINEER_CAR_CFG = ArticulationCfg(

    spawn=sim_utils.UsdFileCfg(
        usd_path=str(Path(__file__).resolve().parents[1] / "car_description/urdf/car/car.usd"),

        activate_contact_sensors=False,

        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,

            max_linear_velocity=5.0,
            max_angular_velocity=20.0,

            max_depenetration_velocity=2.0,

            enable_gyroscopic_forces=True,
        ),

        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            # 현재 바퀴와 차체가 상당히 가까우므로 False 추천
            enabled_self_collisions=False,

            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),

    # =========================================================
    # 초기 상태
    # =========================================================

    init_state=ArticulationCfg.InitialStateCfg(

        # 이전 URDF 기준
        # wheel 중심 = base 기준 z -0.02
        # wheel radius = 0.03
        #
        # base z = 0.05이면
        # wheel center = 0.03
        # wheel bottom = 0.0
        pos=(0.0, 0.0, 0.05),

        rot=(1.0, 0.0, 0.0, 0.0),

        joint_pos={
            # 조향
            "front_left_steer_joint": 0.0,
            "front_right_steer_joint": 0.0,

            # 바퀴
            "front_left_wheel_joint": 0.0,
            "front_right_wheel_joint": 0.0,

            "rear_left_wheel_joint": 0.0,
            "rear_right_wheel_joint": 0.0,
        },

        joint_vel={
            ".*": 0.0,
        },
    ),

    # =========================================================
    # Actuator
    # =========================================================

    actuators={

        # -----------------------------------------------------
        # 앞바퀴 조향
        #
        # Position control
        # -----------------------------------------------------

        "front_left_steering": ImplicitActuatorCfg(

            joint_names_expr=[
                "front_left_steer_joint",
            ],

            effort_limit_sim=5.0,
            velocity_limit_sim=3.0,

            # position control
            stiffness=50.0,
            damping=5.0,
        ),

        "front_right_steering": ImplicitActuatorCfg(

            joint_names_expr=[
                "front_right_steer_joint",
            ],

            effort_limit_sim=5.0,
            velocity_limit_sim=3.0,

            # position control
            stiffness=50.0,
            damping=5.0,
        ),


        # -----------------------------------------------------
        # 뒷바퀴 구동
        #
        # Velocity control
        # -----------------------------------------------------

        "rear_drive": ImplicitActuatorCfg(

            joint_names_expr=[
                "rear_left_wheel_joint",
                "rear_right_wheel_joint",
            ],

            effort_limit_sim=20.0,
            velocity_limit_sim=40.0,

            # velocity control이므로 stiffness = 0
            stiffness=0.0,

            # velocity error에 대해 torque 발생
            damping=5.0,
        ),


        # -----------------------------------------------------
        # 앞바퀴 회전
        #
        # 구동하지 않고 자유롭게 굴러야 함
        # -----------------------------------------------------

        "front_wheel_passive": ImplicitActuatorCfg(

            joint_names_expr=[
                "front_left_wheel_joint",
                "front_right_wheel_joint",
            ],

            effort_limit_sim=0.0,
            velocity_limit_sim=50.0,

            stiffness=0.0,
            damping=0.0,
        ),
    },
)
