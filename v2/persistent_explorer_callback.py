from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from red_gym_env_v2 import RedGymEnv


class PersistentExplorerCallback(BaseCallback):
    """
    Read-only persistent explorer.

    Agent 1 uses the current PPO policy via model.predict(), but its
    observations/actions/rewards are NEVER inserted into PPO's rollout buffer.
    Only Agents 2-12 live inside the training VecEnv.
    """

    def __init__(self, env_config, session_path, seed=10001, verbose=0):
        super().__init__(verbose)
        self.env_config = dict(env_config)
        self.session_path = Path(session_path)
        self.seed = int(seed)
        self.env = None
        self.obs = None

    def _build_env(self):
        config = dict(self.env_config)
        config.update(
            {
                "instance_id": "agent_01",
                "agent_number": 1,
                "persistent_agent": True,
                "persistent_state_path": str(
                    self.session_path / "persistent_agent_1.state"
                ),
                "persistent_autosave_steps": 5000,
                "persistent_stats_limit": 5000,
                "print_rewards": False,
            }
        )

        env = RedGymEnv(config)
        obs, _ = env.reset(seed=self.seed)
        return env, obs

    def _on_training_start(self) -> None:
        self.env, self.obs = self._build_env()
        if self.verbose:
            print(
                "[persistent explorer] Agent 1 started READ-ONLY: "
                "policy input only, no PPO rollout writes"
            )

    def _on_step(self) -> bool:
        if self.env is None or self.obs is None:
            self.env, self.obs = self._build_env()

        # IMPORTANT: model.predict() only reads the current policy. This transition
        # is not passed to the training VecEnv and cannot enter PPO's rollout buffer.
        action, _ = self.model.predict(self.obs, deterministic=False)
        action_int = int(np.asarray(action).reshape(-1)[0])

        obs, _reward, terminated, truncated, _info = self.env.step(action_int)
        self.obs = obs

        # In de huidige RedGymEnv kan de persistent explorer niet door de
        # normale max_steps-limiet worden getruncated. Als een toekomstige
        # echte terminal condition ooit wordt toegevoegd, bewaren we eerst de
        # wereldstate en bouwen we de explorer opnieuw op.
        if terminated or truncated:
            try:
                self.env._save_persistent_state()
            except Exception:
                pass
            self.env, self.obs = self._build_env()

        return True

    def _on_training_end(self) -> None:
        if self.env is None:
            return

        try:
            # Preserve Agent 1 even if training is stopped between autosaves.
            self.env._save_persistent_state()
        except Exception as exc:
            print(f"[persistent explorer] final autosave failed: {exc}")

        try:
            self.env.pyboy.stop()
        except Exception:
            pass
