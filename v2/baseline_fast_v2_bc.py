import os
import sys
from os.path import exists
from pathlib import Path

from red_gym_env_v2_bc import RedGymEnvBC as RedGymEnv
from stream_agent_wrapper import StreamWrapper
from stable_baselines3 import PPO
from stable_baselines3.common import env_checker
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from tensorboard_callback import TensorboardCallback
from persistent_explorer_callback import PersistentExplorerCallback


def make_env(rank, env_conf, seed=0):
    """
    Create one PPO TRAINING environment.

    rank 0..N-1 maps to logical Agents 2..(N+1).
    Agent 1 is intentionally NOT part of this VecEnv. In V3.2 it runs in a
    separate auto-balanced read-only process and cannot write to PPO.
    """

    def _init():
        logical_agent_number = rank + 2
        rank_conf = dict(env_conf)
        rank_conf["agent_number"] = logical_agent_number
        rank_conf["instance_id"] = f"agent_{logical_agent_number:02d}"
        rank_conf["persistent_agent"] = False
        rank_conf["bc_writer"] = True
        rank_conf["bc_reward_enabled"] = True

        env = StreamWrapper(
            RedGymEnv(rank_conf),
            stream_metadata={
                "user": "v3.1-local",
                "env_id": logical_agent_number,
                "color": "#447799",
                "extra": "",
            },
        )
        env.reset(seed=(seed + logical_agent_number))
        return env

    set_random_seed(seed)
    return _init


if __name__ == "__main__":
    use_wandb_logging = False
    ep_length = 2048 * 80
    sess_id = "runs"
    sess_path = Path(sess_id)

    env_config = {
        "headless": True,
        "save_final_state": False,
        "early_stop": False,
        "action_freq": 24,
        "init_state": "../init.state",
        "max_steps": ep_length,
        "print_rewards": False,
        "save_video": False,
        "fast_video": True,
        "session_path": sess_path,
        "gb_path": "../PokemonRed.gb",
        "debug": False,
        "reward_scale": 0.5,
        "explore_weight": 0.25,
        # Lightweight BC is enabled only in this experimental baseline.
        "bc_enabled": os.environ.get("POKEMON_BC_ENABLED", "1") == "1",
        "bc_socket_path": os.environ.get(
            "POKEMON_BC_SOCKET_PATH",
            str((sess_path / "bc_manager.sock").resolve()),
        ),
        "bc_snapshot_path": os.environ.get(
            "POKEMON_BC_SNAPSHOT_PATH",
            f"/dev/shm/pokemonred_bc_{os.getuid()}.json",
        ),
        "bc_reward_budget": float(
            os.environ.get("POKEMON_BC_REWARD_BUDGET", "0.25")
        ),
        "bc_trace_len": int(os.environ.get("POKEMON_BC_TRACE_LEN", "32")),
    }

    print(env_config)

    # Total visible agents includes the read-only Agent 1.
    # Default = 12 (1 explorer + 11 PPO trainers).
    # Example for a future 32-GB setup:
    #   POKEMON_TOTAL_AGENTS=20 python baseline_fast_v2.py
    total_agents = int(os.environ.get("POKEMON_TOTAL_AGENTS", "12"))
    if total_agents < 2:
        raise ValueError("POKEMON_TOTAL_AGENTS must be >= 2")

    num_train_envs = total_agents - 1
    env = SubprocVecEnv(
        [make_env(i, env_config) for i in range(num_train_envs)]
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=ep_length // 2,
        save_path=sess_path,
        name_prefix="poke",
    )

    agent1_target_ratio = float(
        os.environ.get("AGENT1_SPEED_RATIO", "1.0")
    )
    agent1_lead_steps = int(
        os.environ.get("AGENT1_LEAD_STEPS", "128")
    )

    persistent_explorer_callback = PersistentExplorerCallback(
        env_config,
        sess_path,
        sync_check_steps=250,
        status_interval_seconds=2.0,
        nice_value=8,
        num_train_envs=num_train_envs,
        target_ratio=agent1_target_ratio,
        lead_steps=agent1_lead_steps,
        verbose=1,
    )

    callbacks = [
        checkpoint_callback,
        TensorboardCallback(sess_path),
        persistent_explorer_callback,
    ]

    if use_wandb_logging:
        import wandb
        from wandb.integration.sb3 import WandbCallback

        wandb.tensorboard.patch(root_logdir=str(sess_path))
        run = wandb.init(
            project="pokemon-train",
            id=sess_id,
            name="v3.2-auto-balanced-agent1",
            config=env_config,
            sync_tensorboard=True,
            monitor_gym=True,
            save_code=True,
        )
        callbacks.append(WandbCallback())

    # env_checker.check_env(env)

    if sys.stdin.isatty():
        file_name = ""
    else:
        file_name = sys.stdin.read().strip()

    train_steps_batch = ep_length // 64
    resumed_from_checkpoint = exists(file_name + ".zip")

    if resumed_from_checkpoint:
        print("\nloading checkpoint")
        model = PPO.load(file_name, env=env)
        model.n_steps = train_steps_batch
        model.n_envs = num_train_envs
        model.rollout_buffer.buffer_size = train_steps_batch
        model.rollout_buffer.n_envs = num_train_envs
        model.rollout_buffer.reset()
    else:
        model = PPO(
            "MultiInputPolicy",
            env,
            verbose=1,
            n_steps=train_steps_batch,
            batch_size=512,
            n_epochs=1,
            gamma=0.997,
            ent_coef=0.01,
            tensorboard_log=sess_path,
        )

    print(model.policy)
    print(
        f"Total agents: {total_agents}; PPO training envs: {num_train_envs} "
        f"(logical Agents 2-{total_agents}); "
        "Agent 1 = AUTO-BALANCED ASYNC read-only persistent explorer; "
        "PPO BC = CENTRAL RAM MANAGER; "
        f"target={agent1_target_ratio:.2f}x"
    )

    try:
        model.learn(
            total_timesteps=ep_length * num_train_envs * 10000,
            callback=CallbackList(callbacks),
            tb_log_name="poke_ppo",
            # Preserve PPO's historical timestep counter when resuming a checkpoint.
            reset_num_timesteps=not resumed_from_checkpoint,
        )
    except KeyboardInterrupt:
        # V3.2 keeps the V3.1 improvement: a clean Ctrl-C now preserves the exact in-memory
        # PPO policy instead of falling back to the previous periodic checkpoint.
        interrupt_base = sess_path / f"poke_{int(model.num_timesteps)}_steps"
        model.save(str(interrupt_base))
        print(
            "\n[V3-BC] Ctrl-C checkpoint saved: "
            f"{interrupt_base}.zip",
            flush=True,
        )
    finally:
        # Idempotent: also covers KeyboardInterrupt paths where SB3 itself
        # did not reach callback.on_training_end().
        persistent_explorer_callback.close()
        try:
            env.close()
        except Exception:
            pass

        if use_wandb_logging:
            run.finish()
