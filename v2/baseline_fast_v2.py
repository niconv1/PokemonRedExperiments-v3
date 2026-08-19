import sys
from os.path import exists
from pathlib import Path

from red_gym_env_v2 import RedGymEnv
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

    rank 0..10 map to logical Agents 2..12.
    Agent 1 is intentionally NOT part of this VecEnv; it runs read-only in
    PersistentExplorerCallback and therefore cannot write experience to PPO.
    """

    def _init():
        logical_agent_number = rank + 2
        rank_conf = dict(env_conf)
        rank_conf["agent_number"] = logical_agent_number
        rank_conf["instance_id"] = f"agent_{logical_agent_number:02d}"
        rank_conf["persistent_agent"] = False

        env = StreamWrapper(
            RedGymEnv(rank_conf),
            stream_metadata={
                "user": "v2-default",
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
    }

    print(env_config)

    # PPO receives data ONLY from Agents 2-12.
    num_train_envs = 11
    env = SubprocVecEnv(
        [make_env(i, env_config) for i in range(num_train_envs)]
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=ep_length // 2,
        save_path=sess_path,
        name_prefix="poke",
    )

    callbacks = [
        checkpoint_callback,
        TensorboardCallback(sess_path),
        PersistentExplorerCallback(env_config, sess_path, verbose=1),
    ]

    if use_wandb_logging:
        import wandb
        from wandb.integration.sb3 import WandbCallback

        wandb.tensorboard.patch(root_logdir=str(sess_path))
        run = wandb.init(
            project="pokemon-train",
            id=sess_id,
            name="v2-a",
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
        f"PPO training envs: {num_train_envs} (logical Agents 2-12); "
        "Agent 1 = read-only persistent explorer (continuous world + counters)"
    )

    model.learn(
        total_timesteps=ep_length * num_train_envs * 10000,
        callback=CallbackList(callbacks),
        tb_log_name="poke_ppo",
        # Preserve PPO's historical timestep counter when resuming a checkpoint.
        reset_num_timesteps=not resumed_from_checkpoint,
    )

    if use_wandb_logging:
        run.finish()
