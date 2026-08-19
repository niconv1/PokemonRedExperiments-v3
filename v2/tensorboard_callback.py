import os
import json

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Image
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from einops import rearrange, reduce


def merge_dicts(dicts):
    sum_dict = {}
    count_dict = {}
    distrib_dict = {}

    for d in dicts:
        for k, v in d.items():
            if isinstance(v, (int, float)):
                sum_dict[k] = sum_dict.get(k, 0) + v
                count_dict[k] = count_dict.get(k, 0) + 1
                distrib_dict.setdefault(k, []).append(v)

    mean_dict = {}
    for k in sum_dict:
        mean_dict[k] = sum_dict[k] / count_dict[k]
        distrib_dict[k] = np.array(distrib_dict[k])

    return mean_dict, distrib_dict


class TensorboardCallback(BaseCallback):
    """
    TensorBoard logging for PPO TRAINING agents only.

    Agent 1 is intentionally absent from self.training_env, so its persistent
    world/reward/flags cannot contaminate PPO training metrics or all_flags.
    """

    def __init__(self, log_dir, verbose=0):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.writer = None
        self.discovered_flags = {}

    def _on_training_start(self):
        if self.writer is None:
            self.writer = SummaryWriter(
                log_dir=os.path.join(self.log_dir, "histogram")
            )

    def _on_step(self) -> bool:
        # Logical Agent 2 is training_env index 0 and follows the normal V2 reset.
        if self.training_env.env_method("check_if_done", indices=[0])[0]:
            all_infos = self.training_env.get_attr("agent_stats")
            all_final_infos = [stats[-1] for stats in all_infos if stats]

            if all_final_infos:
                mean_infos, distributions = merge_dicts(all_final_infos)
                for key, val in mean_infos.items():
                    self.logger.record(f"env_stats/{key}", val)

                for key, distrib in distributions.items():
                    self.writer.add_histogram(
                        f"env_stats_distribs/{key}", distrib, self.n_calls
                    )
                    self.logger.record(f"env_stats_max/{key}", max(distrib))

            explore_map = np.array(self.training_env.get_attr("explore_map"))
            map_sum = reduce(explore_map, "f h w -> h w", "max")
            self.logger.record(
                "trajectory/explore_sum",
                Image(map_sum, "HW"),
                exclude=("stdout", "log", "json", "csv"),
            )

            # 11 training agents is odd. Pad one blank tile only for visualization
            # so the 2-row TensorBoard image remains valid.
            map_for_grid = explore_map
            if map_for_grid.shape[0] % 2:
                pad = np.zeros_like(map_for_grid[:1])
                map_for_grid = np.concatenate([map_for_grid, pad], axis=0)

            map_row = rearrange(
                map_for_grid, "(r f) h w -> (r h) (f w)", r=2
            )
            self.logger.record(
                "trajectory/explore_map",
                Image(map_row, "HW"),
                exclude=("stdout", "log", "json", "csv"),
            )

            list_of_flag_dicts = self.training_env.get_attr(
                "current_event_flags_set"
            )

            # Per-episode progress of PPO TRAINING agents 2-12 only.
            episode_flags = {}
            for flag_dict in list_of_flag_dicts:
                for key, name in flag_dict.items():
                    episode_flags[key] = name
                    if key not in self.discovered_flags:
                        self.discovered_flags[key] = name

            self.logger.record(
                "trajectory/all_flags",
                json.dumps(episode_flags, ensure_ascii=False),
            )
            self.logger.record(
                "trajectory/current_flags",
                json.dumps(episode_flags, ensure_ascii=False),
            )
            self.logger.record(
                "trajectory/ever_discovered_flags",
                json.dumps(self.discovered_flags, ensure_ascii=False),
            )
            self.logger.record(
                "trajectory/current_flags_count", len(episode_flags)
            )
            self.logger.record(
                "trajectory/ever_discovered_flags_count",
                len(self.discovered_flags),
            )

        return True

    def _on_training_end(self):
        if self.writer:
            self.writer.close()
