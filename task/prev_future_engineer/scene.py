import torch
import random
import math

from isaaclab.utils import configclass

import isaaclab.sim as sim_utils
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.assets import (
    AssetBaseCfg,
    RigidObjectCfg,
    RigidObjectCollectionCfg,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import MultiMeshRayCasterCfg

from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

DEFAULT_Z = 0.3
MAX_OBSTACLE = 5
LINE_WIDTH = 0.02
LINE_HEIGHT = 0.002
LINE_Z = 0.006
INNER_LINE_CORNER = 0.60
OUTER_LINE_EDGE = 1.55
LINE_TOP_X = 1.15
LINE_SIDE_Y = 1.05
ORANGE_LINE_COLOR = (255.0/255.0, 130.0/255.0, 32.0/255.0)
BLUE_LINE_COLOR = (0.0, 51.0/255.0, 153.0/255.0)

def avoid_target_obj():
    # TOF센서 검사 물체
    TOF_TARGETS = [
        # 외벽
        "{ENV_REGEX_NS}/OutWallTop",
        "{ENV_REGEX_NS}/OutWallBottom",
        "{ENV_REGEX_NS}/OutWallLeft",
        "{ENV_REGEX_NS}/OutWallRight",

        # 내벽
        "{ENV_REGEX_NS}/InWallTop",
        "{ENV_REGEX_NS}/InWallBottom",
        "{ENV_REGEX_NS}/InWallLeft",
        "{ENV_REGEX_NS}/InWallRight",

        # 빨강/초록 장애물
        MultiMeshRayCasterCfg.RaycastTargetCfg(
            prim_expr="{ENV_REGEX_NS}/RedObstacle_.*",
            track_mesh_transforms=True,
        ),
        MultiMeshRayCasterCfg.RaycastTargetCfg(
            prim_expr="{ENV_REGEX_NS}/GreenObstacle_.*",
            track_mesh_transforms=True,
        ),
    ]
    return TOF_TARGETS

# 벽 obj
def wall_entity(
    prim_path: str, 
    size: tuple, 
    init_pos: tuple, 
    color: tuple= (0.0, 0.0, 0.0)
) -> AssetBaseCfg:
    return AssetBaseCfg(
        prim_path = f"{prim_path}",
        spawn=sim_utils.CuboidCfg(
            size=size,
            # rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            # mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=color
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=init_pos
        ),
    )

# 바닥 라인 obj
def line_entity(
    prim_path: str,
    start_xy: tuple,
    end_xy: tuple,
    color: tuple,
) -> AssetBaseCfg:
    dx = end_xy[0] - start_xy[0]
    dy = end_xy[1] - start_xy[1]
    length = math.sqrt(dx * dx + dy * dy)
    yaw = math.atan2(dy, dx)
    center_pos = (
        (start_xy[0] + end_xy[0]) / 2,
        (start_xy[1] + end_xy[1]) / 2,
        LINE_Z,
    )

    return AssetBaseCfg(
        prim_path=prim_path,
        spawn=sim_utils.CuboidCfg(
            size=(length, LINE_WIDTH, LINE_HEIGHT),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=color
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=center_pos,
            rot=yaw_quat(yaw),
        ),
    )

# 장애물 obj
def obstacle_entity(
    prim_path: str,
    color: tuple,
) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=prim_path,
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.05, 0.10),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(
                mass=0.1
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=color
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, -5.0),
        ),
    )

def yaw_quat(yaw: float) -> tuple:
    return (
        math.cos(yaw / 2),
        0.0,
        0.0,
        math.sin(yaw / 2),
    )
