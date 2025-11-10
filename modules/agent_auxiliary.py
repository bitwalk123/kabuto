import os
from typing import Callable

from stable_baselines3.common.callbacks import BaseCallback


class EpisodeLimitCallback(BaseCallback):
    """
    SB3 で Episode 回数を Epoch として扱うためのコールバック
    """

    def __init__(self, max_episodes, verbose=0):
        super().__init__(verbose)
        self.max_episodes = max_episodes
        self.episode_count = 0

    def _on_step(self) -> bool:
        # `done` フラグが True のとき、エピソード終了
        if self.locals.get("dones") is not None:
            self.episode_count += sum(self.locals["dones"])
        return self.episode_count < self.max_episodes


class SaveBestModelCallback(BaseCallback):
    """
    SB3 で最高報酬のモデルのみ保存するためのコールバック
    """

    def __init__(
            self,
            save_path: str,
            reward_path: str,
            should_stop: Callable[[], bool],
            verbose=1
    ):
        super().__init__(verbose)
        self.save_path = save_path
        self.reward_path = reward_path
        self.best_mean_reward = self._load_best_reward()
        self.should_stop = should_stop

    def _load_best_reward(self):
        if os.path.exists(self.reward_path):
            with open(self.reward_path, "r") as f:
                return float(f.read())
        return -float("inf")

    def _save_best_reward(self, reward: float):
        with open(self.reward_path, "w") as f:
            f.write(str(reward))

    def _on_step(self) -> bool:
        if self.should_stop():
            print("🛑 Training interrupted by user.")
            return False  # 学習を中断

        if "episode" in self.locals["infos"][0]:
            ep_reward = self.locals["infos"][0]["episode"]["r"]
            if ep_reward > self.best_mean_reward:
                self.best_mean_reward = ep_reward
                # ■■■ self.model はどこで定義されている？ ■■■
                self.model.save(self.save_path)
                self._save_best_reward(ep_reward)
                if self.verbose > 0:
                    print(f"✅ New best reward: {ep_reward:.2f} → Model saved.")
        return True


