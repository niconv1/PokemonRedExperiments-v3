from pathlib import Path
import json
import multiprocessing as mp
import os
import time
import traceback

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from red_gym_env_v2 import RedGymEnv


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _agent1_balance_snapshot(
    ppo_timesteps_value,
    ppo_start_timesteps,
    num_train_envs,
    worker_steps,
    target_ratio,
    lead_steps,
):
    """Return automatic Agent-1 pacing information.

    PPO total_timesteps grows by ``num_train_envs`` for each vector step.
    Dividing the delta by that number gives the progress of one PPO trainer.
    Agent 1 is allowed to follow that per-trainer progress at ``target_ratio``.
    """
    train_envs = max(1, int(num_train_envs))
    ppo_now = int(ppo_timesteps_value.value)
    ppo_delta = max(0, ppo_now - int(ppo_start_timesteps))
    ppo_virtual_steps = ppo_delta / float(train_envs)
    allowed_worker_steps = (
        ppo_virtual_steps * float(target_ratio) + float(lead_steps)
    )
    ahead_steps = float(worker_steps) - allowed_worker_steps
    return {
        "ppo_timesteps": ppo_now,
        "ppo_virtual_steps": ppo_virtual_steps,
        "allowed_worker_steps": allowed_worker_steps,
        "ahead_steps": ahead_steps,
    }


def _build_agent1_env(env_config, session_path, seed):
    config = dict(env_config)
    session_path = Path(session_path)
    config.update(
        {
            "instance_id": "agent_01",
            "agent_number": 1,
            "persistent_agent": True,
            "persistent_state_path": str(
                session_path / "persistent_agent_1.state"
            ),
            "persistent_autosave_steps": 5000,
            "persistent_stats_limit": 5000,
            "print_rewards": False,
        }
    )

    env = RedGymEnv(config)
    obs, _ = env.reset(seed=int(seed))
    return env, obs


def _load_policy_weights(
    model,
    weights_path,
    current_version,
    current_mtime_ns,
):
    """
    Load an atomically published policy snapshot only when the file changed.

    Returns (newest_policy_timestep, newest_mtime_ns).
    """
    weights_path = Path(weights_path)
    if not weights_path.exists():
        return current_version, current_mtime_ns

    stat = weights_path.stat()
    mtime_ns = int(stat.st_mtime_ns)
    if mtime_ns == current_mtime_ns:
        return current_version, current_mtime_ns

    payload = torch.load(
        weights_path,
        map_location="cpu",
        weights_only=False,
    )

    version = int(payload.get("num_timesteps", -1))
    if version > current_version:
        model.policy.load_state_dict(
            payload["policy_state_dict"],
            strict=True,
        )
        model.policy.set_training_mode(False)
        current_version = version

    return current_version, mtime_ns


def _persistent_explorer_worker(
    env_config,
    session_path,
    seed,
    stop_event,
    bootstrap_model_path,
    weights_path,
    status_path,
    sync_check_steps,
    status_interval_seconds,
    nice_value,
    ppo_timesteps_value,
    ppo_start_timesteps,
    num_train_envs,
    target_ratio,
    lead_steps,
    min_sleep_seconds,
    max_sleep_seconds,
):
    """
    Dedicated Agent-1 process.

    It owns its own PyBoy environment and a CPU inference copy of the PPO
    policy.  No Agent-1 transition is ever visible to the PPO training VecEnv
    or rollout buffer.
    """
    env = None
    model = None
    session_path = Path(session_path)
    bootstrap_model_path = Path(bootstrap_model_path)
    weights_path = Path(weights_path)
    status_path = Path(status_path)

    try:
        try:
            os.nice(int(nice_value))
        except Exception:
            pass

        # Avoid a CPU thread-pool explosion next to the 11 PyBoy training
        # workers.  Agent 1 only needs small inference calls.
        try:
            torch.set_num_threads(1)
        except Exception:
            pass
        try:
            torch.set_num_interop_threads(1)
        except Exception:
            pass

        # Parent publishes the bootstrap model before spawning us, but keep a
        # short wait loop for filesystem visibility.
        wait_started = time.monotonic()
        while not bootstrap_model_path.exists():
            if stop_event.is_set():
                return
            if time.monotonic() - wait_started > 30:
                raise RuntimeError(
                    f"bootstrap policy missing: {bootstrap_model_path}"
                )
            time.sleep(0.05)

        model = PPO.load(
            str(bootstrap_model_path),
            env=None,
            device="cpu",
        )
        model.policy.set_training_mode(False)

        env, obs = _build_agent1_env(env_config, session_path, seed)

        policy_version = -1
        policy_mtime_ns = -1
        try:
            policy_version, policy_mtime_ns = _load_policy_weights(
                model,
                weights_path,
                policy_version,
                policy_mtime_ns,
            )
        except Exception:
            # The bootstrap model already contains a valid policy.
            policy_version = int(getattr(model, "num_timesteps", 0))

        started = time.monotonic()
        window_started = started
        window_steps = 0
        total_worker_steps = 0
        last_status_write = 0.0
        last_sync_check = -1
        throttle_sleep_total = 0.0
        throttled_seconds_window = 0.0

        initial_balance = _agent1_balance_snapshot(
            ppo_timesteps_value,
            ppo_start_timesteps,
            num_train_envs,
            total_worker_steps,
            target_ratio,
            lead_steps,
        )

        _atomic_json(
            status_path,
            {
                "alive": True,
                "pid": os.getpid(),
                "agent_number": 1,
                "mode": "persistent-read-only-auto-balanced",
                "worker_steps": total_worker_steps,
                "fps": 0.0,
                "policy_timesteps": int(policy_version),
                "num_train_envs": int(num_train_envs),
                "total_agents": int(num_train_envs) + 1,
                "target_ratio": float(target_ratio),
                "lead_steps": int(lead_steps),
                "balance_ahead_steps": round(
                    float(initial_balance["ahead_steps"]), 2
                ),
                "ppo_virtual_steps": round(
                    float(initial_balance["ppo_virtual_steps"]), 2
                ),
                "throttled": False,
                "throttle_sleep_seconds": 0.0,
                "error": "",
            },
        )

        while not stop_event.is_set():
            balance = _agent1_balance_snapshot(
                ppo_timesteps_value,
                ppo_start_timesteps,
                num_train_envs,
                total_worker_steps,
                target_ratio,
                lead_steps,
            )

            # Automatic pace matching:
            # - if Agent 1 is ahead of one PPO trainer, sleep;
            # - if PPO is ahead, Agent 1 runs full speed until caught up.
            #
            # This automatically scales when the number of PPO environments
            # changes (e.g. 11 trainers today, 19 trainers with 20 total agents).
            if balance["ahead_steps"] > 0:
                sleep_seconds = min(
                    float(max_sleep_seconds),
                    max(
                        float(min_sleep_seconds),
                        float(min_sleep_seconds)
                        + float(balance["ahead_steps"]) * 0.00025,
                    ),
                )
                time.sleep(sleep_seconds)
                throttle_sleep_total += sleep_seconds
                throttled_seconds_window += sleep_seconds

                now = time.monotonic()
                if now - last_status_write >= float(status_interval_seconds):
                    elapsed = max(now - window_started, 1e-6)
                    worker_fps = window_steps / elapsed
                    _atomic_json(
                        status_path,
                        {
                            "alive": True,
                            "pid": os.getpid(),
                            "agent_number": 1,
                            "mode": "persistent-read-only-auto-balanced",
                            "worker_steps": int(total_worker_steps),
                            "env_step": int(getattr(env, "step_count", 0)),
                            "fps": round(float(worker_fps), 2),
                            "policy_timesteps": int(policy_version),
                            "num_train_envs": int(num_train_envs),
                            "total_agents": int(num_train_envs) + 1,
                            "target_ratio": float(target_ratio),
                            "lead_steps": int(lead_steps),
                            "balance_ahead_steps": round(
                                float(balance["ahead_steps"]), 2
                            ),
                            "ppo_virtual_steps": round(
                                float(balance["ppo_virtual_steps"]), 2
                            ),
                            "throttled": True,
                            "throttled_window_seconds": round(
                                float(throttled_seconds_window), 3
                            ),
                            "throttle_sleep_seconds": round(
                                float(throttle_sleep_total), 3
                            ),
                            "uptime_seconds": round(now - started, 1),
                            "error": "",
                        },
                    )
                    window_started = now
                    window_steps = 0
                    throttled_seconds_window = 0.0
                    last_status_write = now
                continue
            if (
                last_sync_check < 0
                or total_worker_steps - last_sync_check
                >= int(sync_check_steps)
            ):
                last_sync_check = total_worker_steps
                try:
                    policy_version, policy_mtime_ns = _load_policy_weights(
                        model,
                        weights_path,
                        policy_version,
                        policy_mtime_ns,
                    )
                except Exception as exc:
                    # Keep playing on the last good policy.  A sync error
                    # should never feed anything back to PPO or kill training.
                    print(
                        "[agent1 async] policy sync skipped: "
                        f"{type(exc).__name__}: {exc}",
                        flush=True,
                    )

            action, _ = model.predict(obs, deterministic=False)
            action_int = int(np.asarray(action).reshape(-1)[0])

            obs, _reward, terminated, truncated, _info = env.step(action_int)

            total_worker_steps += 1
            window_steps += 1

            if terminated or truncated:
                # Persistent RedGymEnv currently never reaches max_steps, but
                # preserve a safe future path for real terminal conditions.
                try:
                    env._save_persistent_state()
                except Exception:
                    pass
                try:
                    env.pyboy.stop()
                except Exception:
                    pass
                env, obs = _build_agent1_env(
                    env_config,
                    session_path,
                    seed,
                )

            now = time.monotonic()
            if now - last_status_write >= float(status_interval_seconds):
                elapsed = max(now - window_started, 1e-6)
                worker_fps = window_steps / elapsed
                balance = _agent1_balance_snapshot(
                    ppo_timesteps_value,
                    ppo_start_timesteps,
                    num_train_envs,
                    total_worker_steps,
                    target_ratio,
                    lead_steps,
                )
                _atomic_json(
                    status_path,
                    {
                        "alive": True,
                        "pid": os.getpid(),
                        "agent_number": 1,
                        "mode": "persistent-read-only-auto-balanced",
                        "worker_steps": int(total_worker_steps),
                        "env_step": int(getattr(env, "step_count", 0)),
                        "fps": round(float(worker_fps), 2),
                        "policy_timesteps": int(policy_version),
                        "num_train_envs": int(num_train_envs),
                        "total_agents": int(num_train_envs) + 1,
                        "target_ratio": float(target_ratio),
                        "lead_steps": int(lead_steps),
                        "balance_ahead_steps": round(
                            float(balance["ahead_steps"]), 2
                        ),
                        "ppo_virtual_steps": round(
                            float(balance["ppo_virtual_steps"]), 2
                        ),
                        "throttled": False,
                        "throttled_window_seconds": round(
                            float(throttled_seconds_window), 3
                        ),
                        "throttle_sleep_seconds": round(
                            float(throttle_sleep_total), 3
                        ),
                        "uptime_seconds": round(now - started, 1),
                        "error": "",
                    },
                )
                window_started = now
                window_steps = 0
                throttled_seconds_window = 0.0
                last_status_write = now

    except BaseException as exc:
        try:
            _atomic_json(
                status_path,
                {
                    "alive": False,
                    "pid": os.getpid(),
                    "agent_number": 1,
                    "mode": "persistent-read-only-auto-balanced",
                    "fps": 0.0,
                    "policy_timesteps": int(
                        getattr(model, "num_timesteps", -1)
                        if model is not None
                        else -1
                    ),
                    "error": (
                        f"{type(exc).__name__}: {exc}\n"
                        f"{traceback.format_exc()}"
                    ),
                },
            )
        except Exception:
            pass
        print(
            f"[agent1 async] worker crashed: {type(exc).__name__}: {exc}",
            flush=True,
        )

    finally:
        if env is not None:
            try:
                # Exact world-state save when PPO is stopped.
                env._save_persistent_state()
            except Exception as exc:
                print(
                    f"[agent1 async] final autosave failed: {exc}",
                    flush=True,
                )
            try:
                env.pyboy.stop()
            except Exception:
                pass

        try:
            previous = {}
            if status_path.exists():
                previous = json.loads(status_path.read_text())
            previous.update(
                {
                    "alive": False,
                    "pid": os.getpid(),
                }
            )
            _atomic_json(status_path, previous)
        except Exception:
            pass


class PersistentExplorerCallback(BaseCallback):
    """
    V3.2 auto-balanced asynchronous read-only persistent explorer.

    PPO trains ONLY from Agents 2-12 inside SubprocVecEnv.

    Agent 1 runs in a dedicated spawned process. The parent publishes a
    read-only policy snapshot after PPO updates and exposes PPO progress via
    lock-free shared memory. The child automatically throttles itself to the
    speed of one PPO trainer. Agent-1 transitions never enter PPO.
    """

    def __init__(
        self,
        env_config,
        session_path,
        seed=10001,
        sync_check_steps=250,
        status_interval_seconds=2.0,
        nice_value=8,
        num_train_envs=11,
        target_ratio=1.0,
        lead_steps=128,
        min_sleep_seconds=0.001,
        max_sleep_seconds=0.05,
        verbose=0,
    ):
        super().__init__(verbose)
        self.env_config = dict(env_config)
        self.session_path = Path(session_path)
        self.seed = int(seed)
        self.sync_check_steps = int(sync_check_steps)
        self.status_interval_seconds = float(status_interval_seconds)
        self.nice_value = int(nice_value)
        self.num_train_envs = max(1, int(num_train_envs))
        self.target_ratio = max(0.05, float(target_ratio))
        self.lead_steps = max(0, int(lead_steps))
        self.min_sleep_seconds = max(0.0001, float(min_sleep_seconds))
        self.max_sleep_seconds = max(
            self.min_sleep_seconds,
            float(max_sleep_seconds),
        )

        self.bootstrap_model_path = (
            self.session_path / "persistent_agent_1_policy_bootstrap.zip"
        )
        self.weights_path = (
            self.session_path / "persistent_agent_1_policy.pt"
        )
        self.status_path = (
            self.session_path / "persistent_agent_1_worker_status.json"
        )

        self._ctx = None
        self._stop_event = None
        self._process = None
        self._last_published_timesteps = None
        self._closed = False
        self._first_rollout_start = True
        self._ppo_timesteps_value = None
        self._ppo_start_timesteps = None

    @property
    def process(self):
        return self._process

    def _publish_bootstrap_model(self):
        self.session_path.mkdir(parents=True, exist_ok=True)
        tmp = self.session_path / "persistent_agent_1_policy_bootstrap.tmp.zip"
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass

        self.model.save(str(tmp))
        tmp.replace(self.bootstrap_model_path)

    def _publish_policy_weights(self, force=False):
        version = int(self.model.num_timesteps)
        if (
            not force
            and self._last_published_timesteps is not None
            and version <= self._last_published_timesteps
        ):
            return

        # Copy only policy tensors to CPU.  This happens at rollout boundaries,
        # not on every environment step.
        state_dict = {
            key: value.detach().cpu()
            for key, value in self.model.policy.state_dict().items()
        }

        payload = {
            "num_timesteps": version,
            "policy_state_dict": state_dict,
        }

        tmp = Path(str(self.weights_path) + ".tmp")
        torch.save(payload, tmp)
        tmp.replace(self.weights_path)
        self._last_published_timesteps = version

        if self.verbose:
            print(
                "[agent1 async] published policy at PPO step "
                f"{version}",
                flush=True,
            )

    def _on_training_start(self) -> None:
        self._closed = False
        self.session_path.mkdir(parents=True, exist_ok=True)

        for path in (
            self.bootstrap_model_path,
            self.weights_path,
            self.status_path,
        ):
            try:
                path.unlink()
            except FileNotFoundError:
                pass

        self._publish_bootstrap_model()
        self._publish_policy_weights(force=True)

        self._ctx = mp.get_context("spawn")
        self._stop_event = self._ctx.Event()
        self._ppo_start_timesteps = int(self.model.num_timesteps)
        self._ppo_timesteps_value = self._ctx.Value(
            "q",
            self._ppo_start_timesteps,
            lock=False,
        )
        self._process = self._ctx.Process(
            target=_persistent_explorer_worker,
            args=(
                self.env_config,
                str(self.session_path),
                self.seed,
                self._stop_event,
                str(self.bootstrap_model_path),
                str(self.weights_path),
                str(self.status_path),
                self.sync_check_steps,
                self.status_interval_seconds,
                self.nice_value,
                self._ppo_timesteps_value,
                self._ppo_start_timesteps,
                self.num_train_envs,
                self.target_ratio,
                self.lead_steps,
                self.min_sleep_seconds,
                self.max_sleep_seconds,
            ),
            name="pokemon-agent1-readonly",
            daemon=True,
        )
        self._process.start()

        print(
            "[agent1 auto] Agent 1 started in separate READ-ONLY process "
            f"pid={self._process.pid}; trainers={self.num_train_envs}; "
            f"target={self.target_ratio:.2f}x per-trainer speed; "
            "PPO rollout writes = 0",
            flush=True,
        )

    def _on_rollout_start(self) -> None:
        if self._ppo_timesteps_value is not None:
            self._ppo_timesteps_value.value = int(self.model.num_timesteps)

        # _on_rollout_start is called again after the previous PPO update.
        # Publish only when num_timesteps changed, so the child receives the
        # freshly trained policy without blocking each vector step.
        if self._first_rollout_start:
            self._first_rollout_start = False
            return
        self._publish_policy_weights(force=False)

    def _on_step(self) -> bool:
        # One lock-free shared-memory write lets Agent 1 know exactly how far
        # one PPO trainer has progressed.  No file I/O, env.step(), predict(),
        # reward or rollout-buffer write happens here.
        if self._ppo_timesteps_value is not None:
            self._ppo_timesteps_value.value = int(self.model.num_timesteps)

        if (
            self._process is not None
            and self.n_calls % 2000 == 0
            and not self._process.is_alive()
        ):
            print(
                "[agent1 async] WARNING: Agent 1 worker is not alive; "
                f"see {self.status_path}",
                flush=True,
            )
        return True

    def close(self):
        if self._closed:
            return
        self._closed = True

        if self._stop_event is not None:
            self._stop_event.set()

        if self._process is not None:
            self._process.join(timeout=15)
            if self._process.is_alive():
                print(
                    "[agent1 async] graceful stop timed out; terminating worker",
                    flush=True,
                )
                self._process.terminate()
                self._process.join(timeout=5)

    def _on_training_end(self) -> None:
        self.close()
