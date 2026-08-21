"""
Pokemon Red AI V3 - lightweight Breadcrumb (BC) PPO environment.

This module deliberately subclasses the known-good stable RedGymEnv instead of
editing it.  The stable ~550-FPS V3 files therefore remain untouched and can be
started at any time with the original start helper.

Performance rules:
* No disk I/O on the PPO hot path.
* No locks in PPO workers.
* No extra event-flag scan: discovery reuses RedGymEnv's existing tracker.
* BC route evaluation happens only when the Game Boy map changes.
* The central manager is the only breadcrumb writer/persistence owner.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Set

from bc_manager import BCClient
from red_gym_env_v2 import RedGymEnv


class RedGymEnvBC(RedGymEnv):
    """Stable RedGymEnv plus a very small breadcrumb client for PPO agents."""

    def __init__(self, config=None):
        config = dict(config or {})

        self.bc_enabled = bool(config.get("bc_enabled", False))
        self.bc_writer = bool(config.get("bc_writer", False))
        self.bc_reward_enabled = bool(config.get("bc_reward_enabled", False))
        self.bc_reward_budget = float(config.get("bc_reward_budget", 0.25))
        self.bc_trace_len = int(config.get("bc_trace_len", 32))
        self.bc_socket_path = Path(
            config.get("bc_socket_path", "runs/bc_manager.sock")
        )
        self.bc_snapshot_path = Path(
            config.get("bc_snapshot_path", "/dev/shm/pokemonred_bc.json")
        )

        # Cumulative raw BC reward for this episode. get_game_state_reward()
        # scales this exactly like the other reward components.
        self._bc_reward_total = 0.0
        self._bc_last_step_reward = 0.0
        self._bc_last_map_id = None
        self._bc_discoveries_sent = 0
        self._bc_client = None

        super().__init__(config)

        if self.bc_enabled:
            self._bc_client = BCClient(
                agent_id=int(self.agent_number),
                socket_path=self.bc_socket_path,
                snapshot_path=self.bc_snapshot_path,
                writer=self.bc_writer,
                reward_enabled=self.bc_reward_enabled,
                reward_budget=self.bc_reward_budget,
                trace_len=self.bc_trace_len,
            )

    def _bc_active_named_targets(self) -> Set[str]:
        """One reset-time scan to avoid rewarding milestones already active."""
        active: Set[str] = set()
        try:
            for key, name in self.event_names.items():
                try:
                    addr_text, bit_text = key.split("-", 1)
                    address = int(addr_text, 16)
                    bit_idx = int(bit_text)
                except Exception:
                    continue

                if int(self.read_m(address)) & (1 << bit_idx):
                    active.add(str(name))
        except Exception:
            pass
        return active

    def reset(self, seed=None, options={}):
        # Reset BC accounting before stable reset() calls get_game_state_reward().
        self._bc_reward_total = 0.0
        self._bc_last_step_reward = 0.0
        self._bc_last_map_id = None

        obs, info = super().reset(seed=seed, options=options)

        if self._bc_client is not None:
            already_seen = self._bc_active_named_targets()
            self._bc_client.reset_episode(already_seen_targets=already_seen)

            # Seed the map state machine once.  This may load the tiny RAM
            # snapshot, but only at reset (and later on actual map changes).
            current_map = int(self.read_m(0xD35E))
            self._bc_last_map_id = current_map
            self._bc_client.observe_map(current_map)

        return obs, info

    def _bc_observe_map(self, map_id: int) -> None:
        """O(1) on the common same-map path; route work only on transitions."""
        self._bc_last_step_reward = 0.0

        if self._bc_client is None:
            return

        map_id = int(map_id)
        if self._bc_last_map_id == map_id:
            return

        self._bc_last_map_id = map_id
        raw_reward = float(self._bc_client.observe_map(map_id))
        if raw_reward > 0.0:
            self._bc_reward_total += raw_reward
            self._bc_last_step_reward = self.reward_scale * raw_reward

    def update_seen_coords(self):
        """
        Stable implementation with one BC hook that reuses the map_id already
        read for exploration.  This method is called before update_reward(), so
        breadcrumb reward is attributed to the same PPO step as the transition.
        """
        if self.read_m(0xD057) == 0:  # not in battle
            x_pos, y_pos, map_n = self.get_game_coords()

            self._bc_observe_map(map_n)

            coord_string = f"x:{x_pos} y:{y_pos} m:{map_n}"
            if coord_string in self.seen_coords.keys():
                self.seen_coords[coord_string] += 1
            else:
                self.seen_coords[coord_string] = 1
        else:
            self._bc_last_step_reward = 0.0

    def get_game_state_reward(self, print_stats=False):
        scores = super().get_game_state_reward(print_stats=print_stats)
        if self.bc_enabled and self.bc_reward_enabled:
            # bc_reward_budget is defined before normal environment scaling.
            scores["bc"] = self.reward_scale * float(self._bc_reward_total)
        return scores

    def _track_last_named_flag_transition(self):
        """
        Reuse the stable V3 event tracker.  It already scans named event flags
        every step for Last Flag, so BC performs no second event-memory scan.
        """
        had_tracker = hasattr(self, "_live_prev_named_flags")
        previous_flags = set(getattr(self, "_live_prev_named_flags", set()))
        previous_step = getattr(self, "_live_flag_prev_step", None)

        super()._track_last_named_flag_transition()

        if self._bc_client is None or not self.bc_writer:
            return

        # First tracker call or episode rollover is baseline initialization, not
        # a new milestone discovery.
        if not had_tracker or previous_step is None:
            return
        if int(self.step_count) < int(previous_step):
            return

        current_flags = set(getattr(self, "_live_prev_named_flags", set()))
        newly_set = current_flags - previous_flags
        if not newly_set:
            return

        # Preserve events.json ordering, but deduplicate equal display names.
        target_names = []
        seen_names = set()
        for key in self.event_names.keys():
            if key not in newly_set:
                continue
            name = str(self.event_names.get(key, key))
            if name in seen_names:
                continue
            seen_names.add(name)
            target_names.append(name)

        for target in target_names:
            if self._bc_client.discover(target):
                self._bc_discoveries_sent += 1

    def close(self):
        if self._bc_client is not None:
            try:
                self._bc_client.close()
            except Exception:
                pass
        try:
            self.pyboy.stop()
        except Exception:
            pass
