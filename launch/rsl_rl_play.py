"""
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
"${ISAACLAB_PATH}/isaaclab.sh" -p launch/rsl_rl_play.py \
--task future_engine_rsl_rl \
--num_envs 5 \
--seed 420 \
--ui \
--curriculum_stage 4 \
--checkpoint logs/rsl_rl/future_engineer_task/2026-09-01_13-14-40/model_38200.pt
"""

import argparse
from pathlib import Path
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument(
    "--num_envs", type=int, 
    default=None, help="Number of environments to simulate."
)
parser.add_argument(
    "--task", type=str, default=None, help="Name of the task."
)
parser.add_argument(
    "--agent", type=str, 
    default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument(
    "--seed", type=int, default=None, 
    help="Seed used for the environment"
)
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--ui", action="store_true", 
    default=False, help="Show Isaac Sim viewport for visual playback."
)
parser.add_argument(
    "--max_steps", type=int, 
    default=None, help="Stop playback after this many environment steps."
)
parser.add_argument(
    "--debug_play", action="store_true", 
    default=False, help="Print action and robot motion during playback."
)
parser.add_argument(
    "--debug_interval", type=int, 
    default=30, help="Print playback debug info every N steps."
)
parser.add_argument(
    "--curriculum_stage", type=int,
    default=None,
    choices=(1, 2, 3, 4, 5, 6),
    help="Force obstacle curriculum stage for playback. Defaults to the checkpoint iteration.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True
if args_cli.ui:
    args_cli.headless = False

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import inspect
import importlib.metadata as metadata
import os
import time
import torch

from rsl_rl.algorithms import PPO
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
# from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
    handle_deprecated_rsl_rl_cfg,
)

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import logging
logger = logging.getLogger(__name__)

import task
# import FrankaPickPlace.tasks  # noqa: F401

installed_version = metadata.version("rsl-rl-lib")


def _nearest_checkpoint_in_dir(directory: Path, requested_name: str) -> Path | None:
    if not directory.is_dir():
        return None
    matches = sorted(directory.glob("model_*.pt"))
    if not matches:
        return None

    try:
        requested_step = int(requested_name.removeprefix("model_").removesuffix(".pt"))
    except ValueError:
        return matches[-1]

    def step_of(path: Path) -> tuple[int, str]:
        try:
            return int(path.stem.removeprefix("model_")), path.name
        except ValueError:
            return 10**18, path.name

    valid = [path for path in matches if path.stem.removeprefix("model_").isdigit()]
    if not valid:
        return matches[-1]
    return min(valid, key=lambda path: (abs(step_of(path)[0] - requested_step), -step_of(path)[0]))


def _resolve_checkpoint_path(
    checkpoint_arg: str | None,
    log_root_path: str,
    load_run: str | None,
    load_checkpoint: str | None,
) -> str | None:
    if not checkpoint_arg:
        try:
            return get_checkpoint_path(log_root_path, load_run, load_checkpoint)
        except ValueError as exc:
            print(f"[WARN] No checkpoint found. Running with an untrained random policy: {exc}")
            return None

    requested = Path(checkpoint_arg).expanduser()
    if requested.exists():
        return str(retrieve_file_path(str(requested)))

    candidate_dirs: list[Path] = []
    if requested.parent != Path(""):
        candidate_dirs.append(requested.parent)

    # Mirror old absolute paths into the current workspace when the run exists locally.
    parts = requested.parts
    if "logs" in parts:
        logs_idx = parts.index("logs")
        mirrored = Path.cwd().joinpath(*parts[logs_idx:])
        candidate_dirs.append(mirrored.parent)

    for candidate_dir in candidate_dirs:
        nearest = _nearest_checkpoint_in_dir(candidate_dir, requested.name)
        if nearest is not None:
            print(
                f"[WARN] Requested checkpoint was not found: {requested}\n"
                f"[WARN] Falling back to nearest available checkpoint: {nearest}"
            )
            return str(nearest.resolve())

    raise FileNotFoundError(f"Checkpoint not found: {requested}")


def _map_legacy_submodule_state_dict(
    legacy_state_dict: dict[str, torch.Tensor],
    *,
    normalizer_prefix: str,
    module_prefix: str,
    std_key: str | None = None,
) -> dict[str, torch.Tensor]:
    mapped: dict[str, torch.Tensor] = {}
    for key, value in legacy_state_dict.items():
        if key.startswith("obs_normalizer."):
            mapped[f"{normalizer_prefix}{key.removeprefix('obs_normalizer.')}"] = value
        elif key.startswith("mlp."):
            mapped[f"{module_prefix}{key.removeprefix('mlp.')}"] = value
        elif key in ("distribution.std_param", "distribution.log_std_param"):
            if std_key is None:
                raise KeyError(f"Legacy actor checkpoint contains {key} but current policy has no std key.")
            mapped[std_key] = value
        else:
            raise KeyError(f"Unsupported legacy checkpoint key: {key}")
    return mapped


def _find_std_key(state_dict: dict[str, torch.Tensor], prefix: str = "") -> str | None:
    for key in (
        f"{prefix}distribution.std_param",
        f"{prefix}distribution.log_std_param",
        f"{prefix}std",
        f"{prefix}log_std",
    ):
        if key in state_dict:
            return key
    return None


def _load_module_state_dict(module: torch.nn.Module, state_dict: dict[str, torch.Tensor], module_name: str) -> None:
    missing, unexpected = module.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"[WARN] {module_name} checkpoint key mismatch. missing={missing}, unexpected={unexpected}")


def _load_legacy_checkpoint(runner: OnPolicyRunner | DistillationRunner, resume_path: str, device: str) -> dict | None:
    checkpoint = torch.load(resume_path, weights_only=False, map_location=device)

    if "model_state_dict" in checkpoint:
        model_state_dict = checkpoint["model_state_dict"]

        actor_state_dict = {}
        critic_state_dict = {}
        actor_std_key = _find_std_key(runner.alg.actor.state_dict()) if hasattr(runner.alg, "actor") else "distribution.std_param"
        for key, value in model_state_dict.items():
            if key == "std":
                actor_state_dict[actor_std_key or "distribution.std_param"] = value
            elif key.startswith("actor."):
                actor_state_dict[f"mlp.{key.removeprefix('actor.')}"] = value
            elif key.startswith("critic."):
                critic_state_dict[f"mlp.{key.removeprefix('critic.')}"] = value

        _load_module_state_dict(runner.alg.actor, actor_state_dict, "Actor")
        _load_module_state_dict(runner.alg.critic, critic_state_dict, "Critic")
        if "iter" in checkpoint:
            runner.current_learning_iteration = checkpoint["iter"]
        return checkpoint.get("infos")

    if "actor_state_dict" not in checkpoint:
        raise KeyError(
            "Checkpoint must contain either 'model_state_dict' or legacy 'actor_state_dict'/'critic_state_dict'."
        )

    if hasattr(runner.alg, "policy"):
        policy = runner.alg.policy
        current_state_dict = policy.state_dict()

        std_key = _find_std_key(current_state_dict, "actor.") or _find_std_key(current_state_dict)

        remapped_state_dict: dict[str, torch.Tensor] = {}
        remapped_state_dict.update(
            _map_legacy_submodule_state_dict(
                checkpoint["actor_state_dict"],
                normalizer_prefix="actor_obs_normalizer.",
                module_prefix="actor.",
                std_key=std_key,
            )
        )

        if "critic_state_dict" in checkpoint:
            remapped_state_dict.update(
                _map_legacy_submodule_state_dict(
                    checkpoint["critic_state_dict"],
                    normalizer_prefix="critic_obs_normalizer.",
                    module_prefix="critic.",
                )
            )

        merged_state_dict = dict(current_state_dict)
        merged_state_dict.update(remapped_state_dict)
        policy.load_state_dict(merged_state_dict, strict=True)
    elif hasattr(runner.alg, "actor") and hasattr(runner.alg, "critic"):
        actor_state_dict = _map_legacy_submodule_state_dict(
            checkpoint["actor_state_dict"],
            normalizer_prefix="obs_normalizer.",
            module_prefix="mlp.",
            std_key=_find_std_key(runner.alg.actor.state_dict()),
        )
        _load_module_state_dict(runner.alg.actor, actor_state_dict, "Actor")

        if "critic_state_dict" in checkpoint:
            critic_state_dict = _map_legacy_submodule_state_dict(
                checkpoint["critic_state_dict"],
                normalizer_prefix="obs_normalizer.",
                module_prefix="mlp.",
            )
            _load_module_state_dict(runner.alg.critic, critic_state_dict, "Critic")
    else:
        raise AttributeError("Runner algorithm has neither 'policy' nor 'actor'/'critic' modules.")

    if "iter" in checkpoint:
        runner.current_learning_iteration = checkpoint["iter"]

    return checkpoint.get("infos")


def _set_play_curriculum_step(
    resume_path: str | None,
    num_steps_per_env: int,
    curriculum_stage: int | None,
) -> None:
    stage_steps = {
        1: 0,
        2: 240_000,
        3: 480_000,
        4: 720_000,
        5: 1_000_000,
        6: 1_240_000,
    }
    if curriculum_stage is not None:
        curriculum_step = stage_steps[curriculum_stage]
        os.environ["FUTURE_ENGINEER_CURRICULUM_STAGE_OVERRIDE"] = str(curriculum_stage)
        os.environ["FUTURE_ENGINEER_CURRICULUM_STEP_OVERRIDE"] = str(curriculum_step)
        print(
            "[INFO] Play curriculum stage forced: "
            f"stage={curriculum_stage}, step={curriculum_step}"
        )
        return

    if resume_path is None:
        return

    try:
        checkpoint = torch.load(resume_path, weights_only=False, map_location="cpu")
    except Exception as exc:
        print(f"[WARN] Could not inspect checkpoint curriculum step: {exc}")
        return

    checkpoint_iter = checkpoint.get("iter")
    if checkpoint_iter is None:
        return

    curriculum_step = int(checkpoint_iter) * int(num_steps_per_env)
    os.environ["FUTURE_ENGINEER_CURRICULUM_STEP_OVERRIDE"] = str(curriculum_step)
    print(
        "[INFO] Play curriculum step restored from checkpoint: "
        f"iter={checkpoint_iter}, step={curriculum_step}"
    )


def _sanitize_runner_cfg_for_installed_rsl_rl(agent_cfg: RslRlBaseRunnerCfg) -> dict:
    """Drop unsupported PPO kwargs for the installed rsl-rl version."""
    runner_cfg = agent_cfg.to_dict()
    algorithm_cfg = runner_cfg.get("algorithm", {})
    if runner_cfg.get("class_name") != "OnPolicyRunner" or algorithm_cfg.get("class_name") != "PPO":
        return runner_cfg

    ppo_signature = inspect.signature(PPO.__init__)
    allowed_keys = set(ppo_signature.parameters) - {"self", "policy", "device", "multi_gpu_cfg"}
    filtered_algorithm_cfg = {
        key: value
        for key, value in algorithm_cfg.items()
        if key == "class_name" or key in allowed_keys
    }
    dropped_keys = sorted(set(algorithm_cfg) - set(filtered_algorithm_cfg))
    if dropped_keys:
        print(f"[INFO] Dropped unsupported PPO args for installed rsl-rl: {dropped_keys}")

    runner_cfg["algorithm"] = filtered_algorithm_cfg
    return runner_cfg


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.enable_cameras:
        env_cfg.sim.use_fabric = False
        print("[INFO] --enable_cameras enabled: disabling Fabric for Isaac Sim 4.5 camera pose updates")
    if args_cli.ui:
        env_cfg.sim.device = "cpu"
        agent_cfg.device = "cpu"
        print("[INFO] --ui enabled: using CPU simulation so the viewport shows reset/motion updates")

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        print("[ERROR] get_published_pretrained_checkpoint 모듈을 찾을 수 없습니다.")
        return
    else:
        resume_path = _resolve_checkpoint_path(
            args_cli.checkpoint,
            log_root_path,
            agent_cfg.load_run,
            agent_cfg.load_checkpoint,
        )

    log_dir = os.path.dirname(resume_path) if resume_path is not None else os.path.join(log_root_path, "play_random")
    _set_play_curriculum_step(resume_path, agent_cfg.num_steps_per_env, args_cli.curriculum_stage)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner_cfg = _sanitize_runner_cfg_for_installed_rsl_rl(agent_cfg)

    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, runner_cfg, log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, runner_cfg, log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    if resume_path is not None:
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        _load_legacy_checkpoint(runner, resume_path, agent_cfg.device)
    else:
        print("[INFO]: No checkpoint supplied/found. Using a freshly initialized random policy.")

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        if hasattr(runner.alg, "actor"):
            policy_nn = runner.alg.actor
        else:
            # version 2.2 and below
            policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    elif hasattr(policy_nn, "obs_normalizer"):
        normalizer = policy_nn.obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    if resume_path is not None:
        export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
        try:
            export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
            export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")
        except Exception as exc:
            print(f"[WARN] Skipping policy export because the installed rsl-rl exporter rejected this policy: {exc}")

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    timestep = 0
    print("[INFO] Starting policy playback. Close Isaac Sim or press Ctrl+C to stop.", flush=True)
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, dones, _ = env.step(actions)
            if args_cli.debug_play and timestep % max(args_cli.debug_interval, 1) == 0:
                robot = env.unwrapped.scene["robot"]
                processed_actions = env.unwrapped.action_manager.get_term("drive").processed_actions
                root_xy = robot.data.root_pos_w[:, :2]
                speed = torch.linalg.norm(robot.data.root_lin_vel_w[:, :2], dim=1)
                print(
                    "[DEBUG_PLAY] "
                    f"step={timestep} "
                    f"raw_action={actions.detach().mean(dim=0).cpu().tolist()} "
                    f"processed_action={processed_actions.detach().mean(dim=0).cpu().tolist()} "
                    f"root_xy={root_xy.detach().mean(dim=0).cpu().tolist()} "
                    f"speed_mean={speed.mean().item():.4f} "
                    f"dones={dones.detach().cpu().tolist()}"
                )
            # reset recurrent states for episodes that have terminated
            policy_nn.reset(dones)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break
        elif args_cli.max_steps is not None:
            timestep += 1
            if timestep >= args_cli.max_steps:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
