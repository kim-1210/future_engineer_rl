"""
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
"${ISAACLAB_PATH}/isaaclab.sh" -p launch/view_scene.py --num_envs 4 --randomize_obstacles --reset_interval_s 3 \
--enable_cameras
"""

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="View the Future Engineer scene only.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of scene copies to spawn.")
parser.add_argument("--env_spacing", type=float, default=5.0, help="Spacing between environment copies.")
parser.add_argument(
    "--randomize_obstacles",
    action="store_true",
    default=False,
    help="Run the obstacle reset function once so red/green obstacles are visible.",
)
parser.add_argument(
    "--reset_interval_s",
    type=float,
    default=0.0,
    help="Randomize all obstacles repeatedly every N seconds. Disabled when 0.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import torch

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene

from task.future_engineer.cfg import FutureGineerEnv
from task.future_engineer.mdp.events import reset_scene


class _SceneOnlyEnv:
    def __init__(self, scene, sim):
        self.scene = scene
        self.sim = sim


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 60.0, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=(0.0, -4.5, 3.2), target=(0.0, 0.0, 0.0))

    scene_cfg = FutureGineerEnv(num_envs=args_cli.num_envs, env_spacing=args_cli.env_spacing)
    scene = InteractiveScene(scene_cfg)

    sim.reset()

    if args_cli.randomize_obstacles:
        env_ids = torch.arange(scene.num_envs, device=sim.device)
        reset_scene(_SceneOnlyEnv(scene, sim), env_ids)
        scene.write_data_to_sim()

    elapsed_time = 0.0
    next_reset_time = args_cli.reset_interval_s

    while simulation_app.is_running():
        scene.write_data_to_sim()
        sim.step()
        dt = sim.get_physics_dt()
        scene.update(dt)

        if args_cli.reset_interval_s > 0.0:
            elapsed_time += dt
            if elapsed_time >= next_reset_time:
                env_ids = torch.arange(scene.num_envs, device=sim.device)
                reset_scene(_SceneOnlyEnv(scene, sim), env_ids)
                next_reset_time += args_cli.reset_interval_s


if __name__ == "__main__":
    main()
    simulation_app.close()
