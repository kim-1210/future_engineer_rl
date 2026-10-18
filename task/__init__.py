import gymnasium as gym

from task.future_engineer.cfg import FutureEngineerCfg, FutureEngineerEnv
from task.future_engineer.agent.rsl_rl_ppo_cfg import FutureEngineerPPORunnerCfg

gym.register(
    id="future_engine_rsl_rl",
    entry_point="task.future_engineer.cfg:FutureEngineerEnv",
    kwargs={
        "env_cfg_entry_point": "task.future_engineer.cfg:FutureEngineerCfg",
        "rsl_rl_cfg_entry_point": "task.future_engineer.agent.rsl_rl_ppo_cfg:FutureEngineerPPORunnerCfg",
    },
)

gym.register(
    id="prev_future_engine_rsl_rl",
    entry_point="task.prev_future_engineer.cfg:FutureEngineerEnv",
    kwargs={
        "env_cfg_entry_point": "task.prev_future_engineer.cfg:FutureEngineerCfg",
        "rsl_rl_cfg_entry_point": "task.prev_future_engineer.agent.rsl_rl_ppo_cfg:FutureEngineerPPORunnerCfg",
    },
)
