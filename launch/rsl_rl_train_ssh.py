"""
학습 (VRAM 16GB 기준)
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
"${ISAACLAB_PATH}/isaaclab.sh" -p launch/rsl_rl_train.py \
--task future_engine_rsl_rl \
--num_envs 80 \
--seed 41 \
--max_iterations 20000 \
--enable_cameras \
--ui

재학습
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
"${ISAACLAB_PATH}/isaaclab.sh" -p launch/rsl_rl_train.py \
--task future_engine_rsl_rl \
--num_envs 80 \
--resume \
--load_run 2026-08-26_18-39-34 \
--checkpoint model_3700.pt \
--max_iterations 16300 \
--enable_cameras \
--seed 61
"""

import argparse
import os
import socket
import subprocess
import sys


def _parse_ssh_gpu_ids(value: str) -> list[int]:
    gpu_ids = [int(gpu_id.strip()) for gpu_id in value.split(",") if gpu_id.strip()]
    if not gpu_ids:
        raise ValueError("--ssh_gpu_ids must contain at least one GPU id, e.g. --ssh_gpu_ids 2,3")
    return gpu_ids


def _remove_option(argv: list[str], option: str) -> list[str]:
    cleaned = []
    skip_next = False
    for index, arg in enumerate(argv):
        if skip_next:
            skip_next = False
            continue
        if arg == option:
            if index + 1 < len(argv) and not argv[index + 1].startswith("--"):
                skip_next = True
            continue
        if arg.startswith(option + "="):
            continue
        cleaned.append(arg)
    return cleaned


def _find_free_port() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return str(sock.getsockname()[1])


def _maybe_launch_ssh_workers() -> None:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--ssh_gpu_ids", type=str, default=None)
    pre_parser.add_argument("--ssh_child", action="store_true", default=False)
    pre_parser.add_argument("--ssh_master_port", type=str, default=None)
    pre_args, _ = pre_parser.parse_known_args()

    if pre_args.ssh_gpu_ids is None or pre_args.ssh_child:
        return

    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        raise RuntimeError(
            "rsl_rl_train_ssh.py should be run directly, not through torch.distributed.run, "
            "when --ssh_gpu_ids is used."
        )
    if "--distributed" not in sys.argv[1:]:
        raise RuntimeError("Please pass --distributed when using --ssh_gpu_ids.")

    gpu_ids = _parse_ssh_gpu_ids(pre_args.ssh_gpu_ids)
    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
    master_port = pre_args.ssh_master_port or os.environ.get("MASTER_PORT") or _find_free_port()

    child_args = _remove_option(sys.argv[1:], "--ssh_gpu_ids")
    child_args = _remove_option(child_args, "--ssh_master_port")
    child_args = ["--ssh_child", "--ssh_gpu_ids", pre_args.ssh_gpu_ids, *child_args]

    print(f"[SSH TRAIN] launching {len(gpu_ids)} workers on physical GPUs: {gpu_ids}")
    print(f"[SSH TRAIN] MASTER_ADDR={master_addr} MASTER_PORT={master_port}")

    processes = []
    try:
        for rank, gpu_id in enumerate(gpu_ids):
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            env["MASTER_ADDR"] = master_addr
            env["MASTER_PORT"] = master_port
            env["WORLD_SIZE"] = str(len(gpu_ids))
            env["RANK"] = str(rank)
            env["LOCAL_RANK"] = "0"
            env["SSH_PHYSICAL_GPU_ID"] = str(gpu_id)
            env.pop("JAX_RANK", None)
            env.pop("JAX_LOCAL_RANK", None)

            cmd = [sys.executable, os.path.abspath(__file__), *child_args]
            print(f"[SSH TRAIN] rank={rank} physical_gpu={gpu_id} CUDA_VISIBLE_DEVICES={gpu_id}")
            processes.append(subprocess.Popen(cmd, env=env))

        return_codes = [process.wait() for process in processes]
    except KeyboardInterrupt:
        for process in processes:
            process.terminate()
        raise

    failed = [code for code in return_codes if code != 0]
    if failed:
        raise SystemExit(max(failed))
    raise SystemExit(0)


_maybe_launch_ssh_workers()

from isaaclab.app import AppLauncher

# local imports
import cli_args

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="렌더링/카메라 센서를 활성화합니다. Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument(
    "--num_envs", type=int, 
    default=1024, help="Numbe of environments to simulate."
)
parser.add_argument(
    "--task", 
    type=str, 
    default=None, 
    help="Name of the task."
)
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--curriculum_stage",
    type=int,
    default=None,
    choices=(1, 2, 3, 4, 5, 6),
    help="Start training from this obstacle curriculum stage. It can still advance normally.",
)
parser.add_argument( 
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument(
    "--ray-proc-id", "-rid", type=int, default=None, help="Automatically configured by Ray integration, otherwise None."
)
parser.add_argument(
    "--ssh_gpu_ids",
    type=str,
    default=None,
    help="Comma-separated physical GPU ids for SSH distributed training, e.g. 2,3.",
)
parser.add_argument("--ssh_child", action="store_true", default=False, help=argparse.SUPPRESS)
parser.add_argument(
    "--ssh_master_port",
    type=str,
    default=None,
    help="Master port used by the SSH worker launcher.",
)
parser.add_argument(
    "--ui",
    action="store_true",
    default=False,
    help="시각적으로 안보이게 하기"
)

# rsl_rl args 덮붙이기 (객체로 전달되기 때문에 return안해도됨)
cli_args.add_rsl_rl_args(parser)
# AppLauncher에 필요한 인지들을 parser에 추가
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True
if args_cli.ui:
    args_cli.headless = False
else:
    args_cli.headless = True

ssh_physical_gpu_id = os.environ.get("SSH_PHYSICAL_GPU_ID")
if ssh_physical_gpu_id is not None:
    args_cli.device = "cuda:0"
    args_cli.device_explicit = True
    ssh_kit_args = [
        "--/renderer/multiGpu/enabled=false",
        "--/renderer/multiGpu/autoEnable=false",
        "--/renderer/multiGpu/maxGpuCount=1",
        f"--/renderer/activeGpu={ssh_physical_gpu_id}",
        "--/physics/cudaDevice=0",
    ]
    args_cli.kit_args = " ".join([args_cli.kit_args, *ssh_kit_args]).strip()
    print(
        "[SSH TRAIN] worker uses "
        f"physical_gpu={ssh_physical_gpu_id}, CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}, "
        "training_device=cuda:0"
    )
# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata
import platform

from packaging import version

# check minimum supported rsl-rl version
RSL_RL_VERSION = "3.0.1"
installed_version = metadata.version("rsl-rl-lib")
if version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = [os.path.join(os.environ.get("ISAACLAB_PATH", "/path/to/IsaacLab"), "isaaclab.sh"), "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""Rest everything follows."""

import gymnasium as gym
import logging
import os
import torch
from datetime import datetime

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# import logger
logger = logging.getLogger(__name__)

import task

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

PROGRESS_LOG_PATH = os.path.abspath(os.path.join("logs", "rsl_rl_progress.log"))

def progress(msg: str) -> None:
    """Write short progress messages to a dedicated file (and flush immediately)."""
    os.makedirs(os.path.dirname(PROGRESS_LOG_PATH), exist_ok=True)
    with open(PROGRESS_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(msg.rstrip() + "\n")
        f.flush()


def _get_checkpoint_curriculum_step(resume_path: str, num_steps_per_env: int) -> int | None:
    try:
        checkpoint = torch.load(resume_path, weights_only=False, map_location="cpu")
    except Exception as exc:
        progress(f"[WARN] Could not inspect checkpoint curriculum step: {exc}")
        return None

    checkpoint_iter = checkpoint.get("iter")
    if checkpoint_iter is None:
        return None

    curriculum_step = int(checkpoint_iter) * int(num_steps_per_env)
    progress(
        "[INFO] Train curriculum step read from checkpoint: "
        f"iter={checkpoint_iter}, step={curriculum_step}"
    )
    return curriculum_step


def _restore_env_curriculum_step(env, curriculum_step: int | None) -> None:
    if curriculum_step is None:
        return
    env.unwrapped.common_step_counter = int(curriculum_step)
    progress(
        "[INFO] Train environment common_step_counter restored: "
        f"step={env.unwrapped.common_step_counter}"
    )


def _set_initial_curriculum_stage(curriculum_stage: int | None) -> None:
    if curriculum_stage is None:
        os.environ.pop("FUTURE_ENGINEER_CURRICULUM_INITIAL_STAGE", None)
        return
    os.environ["FUTURE_ENGINEER_CURRICULUM_INITIAL_STAGE"] = str(curriculum_stage)
    progress(f"[INFO] Train curriculum initial stage set: stage={curriculum_stage}")


class _TeeStdout:
    """
    Tee sys.stdout:
    - still prints to terminal (so IsaacLab behavior doesn't change)
    - also copies ONLY training progress blocks into a file
    """

    def __init__(self, real_stdout):
        self.real_stdout = real_stdout
        self._buffer = ""

    def write(self, s: str) -> int:
        # Always pass-through to console
        self.real_stdout.write(s)

        # Collect lines and filter only what we want
        self._buffer += s
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)

            if (
                "Learning iteration" in line
                or "Mean reward" in line
                or "Mean episode length" in line
                or "Mean value_function loss" in line
                or "Mean surrogate loss" in line
                or "Mean entropy loss" in line
                or "Total timesteps" in line
                or "Iteration time" in line
                or "Time elapsed" in line
                or "ETA" in line
                or line.startswith("Episode_Reward/")
                or line.startswith("Metrics/")
                or line.startswith("Curriculum/")
                or line.startswith("Episode_Termination/")
                or line.startswith("#")  # separator blocks
            ):
                progress(line)

        return len(s)

    def flush(self) -> None:
        self.real_stdout.flush()

@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    # progress log 시작 표시
    progress("=" * 80)
    progress(
        "[START] "
        f"task={args_cli.task} num_envs={args_cli.num_envs} "
        f"max_iter={args_cli.max_iterations} seed={args_cli.seed} "
        f"curriculum_stage={args_cli.curriculum_stage}"
    )
    progress(f"[INFO] progress log => {PROGRESS_LOG_PATH}")
    progress("=" * 80)

    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )
    _set_initial_curriculum_stage(args_cli.curriculum_stage)

    # handle deprecated configurations
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.enable_cameras:
        env_cfg.sim.use_fabric = False
        progress("[INFO] --enable_cameras enabled: disabling Fabric for Isaac Sim 4.5 camera pose updates")
    if args_cli.ui:
        # Isaac Sim 4.5 camera sensors fail with Fabric in this setup, while GPU+no-Fabric does not update the UI.
        # CPU simulation keeps USD transforms synchronized, so the debug UI shows vehicle and obstacle movement.
        env_cfg.sim.device = "cpu"
        agent_cfg.device = "cpu"
        progress("[INFO] --ui enabled: using CPU simulation so the viewport shows reset/motion updates")
    # check for invalid combination of CPU device with distributed training
    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError(
            "Distributed training is not supported when using CPU device. "
            "Please use GPU device (e.g., --device cuda) for distributed training."
        )

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed_rank = app_launcher.global_rank if ssh_physical_gpu_id is not None else app_launcher.local_rank
        seed = agent_cfg.seed + seed_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed
        if ssh_physical_gpu_id is not None:
            progress(
                "[INFO] SSH worker gpu mapping: "
                f"rank={app_launcher.global_rank}, physical_gpu={ssh_physical_gpu_id}, device={agent_cfg.device}"
            )

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    progress(f"[INFO] log_root_path={log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")

    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    progress(f"[INFO] log_dir={os.path.abspath(log_dir)}")

    # Save resume path before creating the environment so curriculum reset events see the restored step.
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        progress(f"[INFO] resume_path={resume_path}")
        resume_curriculum_step = _get_checkpoint_curriculum_step(resume_path, agent_cfg.num_steps_per_env)
    else:
        resume_curriculum_step = None

    # set the IO descriptors export flag if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        logger.warning(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    _restore_env_curriculum_step(env, resume_curriculum_step)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # create runner from rsl-rl
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # run training
    old_stdout = sys.stdout
    sys.stdout = _TeeStdout(old_stdout)
    try:
        progress("[TRAIN] runner.learn() started")
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
        progress("[TRAIN] runner.learn() finished")
    finally:
        sys.stdout = old_stdout

    # close the simulator
    env.close()
    progress("[END] env closed")

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
    
