"""エピソード単位の簡易リプレイバッファ。"""

from __future__ import annotations

from typing import Any

import numpy as np


class ReplayBuffer:
    """完了したエピソードを保存し、固定長シーケンスをサンプルする。"""

    def __init__(self, capacity: int = 1000):
        self.capacity = int(capacity)
        self._episodes: list[dict[str, Any]] = []

    def __len__(self) -> int:
        return len(self._episodes)

    @property
    def n_episodes(self) -> int:
        return len(self._episodes)

    def can_sample(self, n_steps: int) -> bool:
        """長さ n_steps 以上のチャンクが1本でもあれば真。"""
        return any(len(ep["rew"]) >= n_steps for ep in self._episodes)

    def add_episode(self, episode: dict[str, Any]) -> None:
        """
        episode は次のキーを持つ。
          obs: dict[str, list[np.ndarray]]  長さ T+1（reset の観測を含む）
          act: dict[str, list[np.ndarray]]  長さ T
          rew: list[float]                  長さ T
          done: list[bool]                  長さ T
        """
        self._episodes.append(episode)
        if len(self._episodes) > self.capacity:
            self._episodes = self._episodes[-self.capacity :]

    def sample_paths(
        self,
        n_paths: int,
        n_steps: int,
        rng: np.random.Generator | None = None,
    ) -> dict[str, Any]:
        """
        返り値の時間長は n_steps。
          obs: dict[str, np.ndarray]  (n_paths, n_steps+1, ...)
          act: dict[str, np.ndarray]  (n_paths, n_steps, ...)
          rew: np.ndarray             (n_paths, n_steps)
          reset: np.ndarray           (n_paths, n_steps)  先頭または done 直後が True
        """
        if not self._episodes:
            raise RuntimeError("リプレイが空です。")
        rng = rng or np.random.default_rng()
        # 十分な長さのエピソードだけを対象にする。
        valid = [ep for ep in self._episodes if len(ep["rew"]) >= n_steps]
        if not valid:
            raise RuntimeError(
                f"長さ {n_steps} 以上のエピソードがありません。"
                f"保有エピソード数={len(self._episodes)}"
            )

        obs_keys = list(valid[0]["obs"].keys())
        act_keys = list(valid[0]["act"].keys())
        batch_obs = {k: [] for k in obs_keys}
        batch_act = {k: [] for k in act_keys}
        batch_rew = []
        batch_reset = []

        for _ in range(n_paths):
            ep = valid[int(rng.integers(0, len(valid)))]
            t_max = len(ep["rew"]) - n_steps + 1
            start = int(rng.integers(0, t_max))
            end = start + n_steps
            for k in obs_keys:
                # 観測は start..end（n_steps+1 個）
                seq = ep["obs"][k][start : end + 1]
                batch_obs[k].append(np.stack(seq, axis=0))
            for k in act_keys:
                seq = ep["act"][k][start:end]
                batch_act[k].append(np.stack(seq, axis=0))
            batch_rew.append(np.asarray(ep["rew"][start:end], dtype=np.float32))
            resets = np.zeros((n_steps,), dtype=bool)
            resets[0] = start == 0
            # 途中の done の次を reset にする。
            dones = np.asarray(ep["done"][start:end], dtype=bool)
            # done[t] が True なら次のステップが新しいエピソードだが、
            # 切り出した部分列内では start==0 のみを確実に扱う。
            if start > 0 and ep["done"][start - 1]:
                resets[0] = True
            batch_reset.append(resets)

        return {
            "obs": {k: np.stack(v, axis=0) for k, v in batch_obs.items()},
            "act": {k: np.stack(v, axis=0) for k, v in batch_act.items()},
            "rew": np.stack(batch_rew, axis=0),
            "reset": np.stack(batch_reset, axis=0),
        }
