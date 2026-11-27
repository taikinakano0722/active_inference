"""リプレイバッファ。

ReplayBuffer は、環境をやり直してから終わるまでの系列を1件として保存する。
StreamReplayBuffer は、ステップを1本の列につなぎ、古いステップから捨てる。
"""

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
        dones = list(episode["done"])
        # 1件の中にエピソードの境目は置かない。終わりは最後のステップだけ。
        if any(dones[:-1]):
            raise ValueError(
                "1件の記録の途中に done があります。"
                "終わりは最後のステップだけにしてください。"
            )
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
          reset: np.ndarray           (n_paths, n_steps)  切り出しの先頭は常に True
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
            # 切り出しの先頭は、元の系列のどこからでも開始扱いにする。
            # 記憶と、その時刻の行動をゼロにしてから学習する。
            resets = np.zeros((n_steps,), dtype=bool)
            resets[0] = True
            batch_reset.append(resets)

        return {
            "obs": {k: np.stack(v, axis=0) for k, v in batch_obs.items()},
            "act": {k: np.stack(v, axis=0) for k, v in batch_act.items()},
            "rew": np.stack(batch_rew, axis=0),
            "reset": np.stack(batch_reset, axis=0),
        }


def _copy_leaves(values: dict[str, Any]) -> dict[str, np.ndarray]:
    """後から元の配列が書き換わっても、保存した中身が変わらないようにする。"""
    return {k: np.array(v, copy=True) for k, v in values.items()}


class StreamReplayBuffer:
    """ステップを1本の列として保存し、固定長の区間をサンプルする。

    エピソードが終わったあとも列は区切らず、次のエピソードのステップを続ける。
    容量を超えた分は、一番古いステップから捨てる。
    各ステップは、行動の前の観測と、行動のあとの観測を両方持つ。
    境目ではこの2つが隣のステップと一致しないので、別々に保存する。
    """

    def __init__(self, capacity: int = 100_000):
        self.capacity = int(capacity)
        self._obs: dict[str, list[np.ndarray]] = {}
        self._next_obs: dict[str, list[np.ndarray]] = {}
        self._act: dict[str, list[np.ndarray]] = {}
        self._rew: list[float] = []
        self._done: list[bool] = []
        self._is_first: list[bool] = []

    def __len__(self) -> int:
        return len(self._rew)

    def can_sample(self, n_steps: int) -> bool:
        """列の長さが n_steps 以上なら真。"""
        return len(self._rew) >= n_steps

    def add_step(
        self,
        obs: dict[str, Any],
        act: dict[str, Any],
        rew: float,
        done: bool,
        is_first: bool,
        next_obs: dict[str, Any],
    ) -> None:
        """1ステップを列の末尾に足す。

        obs は行動の前の観測、next_obs は行動のあとの観測。
        is_first が真のとき、このステップは新しいエピソードの開始で、
        直前のステップとはつながっていない。
        """
        obs_c = _copy_leaves(obs)
        next_c = _copy_leaves(next_obs)
        act_c = _copy_leaves(act)
        if set(obs_c) != set(next_c):
            raise ValueError("行動前と行動後の観測で、キーが違います。")
        if not self._rew:
            self._obs = {k: [] for k in obs_c}
            self._next_obs = {k: [] for k in next_c}
            self._act = {k: [] for k in act_c}
        if set(obs_c) != set(self._obs) or set(next_c) != set(self._next_obs):
            raise ValueError("観測のキーが、列に入っているステップと違います。")
        if set(act_c) != set(self._act):
            raise ValueError("行動のキーが、列に入っているステップと違います。")
        for k in self._obs:
            self._obs[k].append(obs_c[k])
            self._next_obs[k].append(next_c[k])
        for k in self._act:
            self._act[k].append(act_c[k])
        self._rew.append(float(rew))
        self._done.append(bool(done))
        self._is_first.append(bool(is_first))
        overflow = len(self._rew) - self.capacity
        if overflow > 0:
            self._drop_oldest(overflow)

    def _drop_oldest(self, n: int) -> None:
        """先頭から n ステップを捨てる。1つずつ捨てたときと同じ並びに残る。"""
        for k in self._obs:
            del self._obs[k][:n]
            del self._next_obs[k][:n]
        for k in self._act:
            del self._act[k][:n]
        del self._rew[:n]
        del self._done[:n]
        del self._is_first[:n]

    def sample_paths(
        self,
        n_paths: int,
        n_steps: int,
        rng: np.random.Generator | None = None,
    ) -> dict[str, Any]:
        """
        返り値の時間長は n_steps。形は ReplayBuffer.sample_paths と同じ。
          obs: dict[str, np.ndarray]  (n_paths, n_steps+1, ...)
          act: dict[str, np.ndarray]  (n_paths, n_steps, ...)
          rew: np.ndarray             (n_paths, n_steps)
          reset: np.ndarray           (n_paths, n_steps)
            切り出しの先頭は常に True。
            区間の中で is_first が真のステップも True。
        """
        if len(self._rew) < n_steps:
            raise RuntimeError(
                f"長さ {n_steps} 以上の列がありません。保有ステップ数={len(self._rew)}"
            )
        rng = rng or np.random.default_rng()
        obs_keys = list(self._obs.keys())
        act_keys = list(self._act.keys())
        batch_obs = {k: [] for k in obs_keys}
        batch_act = {k: [] for k in act_keys}
        batch_rew = []
        batch_reset = []
        n_starts = len(self._rew) - n_steps + 1

        for _ in range(n_paths):
            start = int(rng.integers(0, n_starts))
            end = start + n_steps
            for k in obs_keys:
                # 先頭は行動前の観測。以降は各行動のあとの観測。
                seq = [self._obs[k][start]]
                seq.extend(self._next_obs[k][t] for t in range(start, end))
                batch_obs[k].append(np.stack(seq, axis=0))
            for k in act_keys:
                seq = self._act[k][start:end]
                batch_act[k].append(np.stack(seq, axis=0))
            batch_rew.append(np.asarray(self._rew[start:end], dtype=np.float32))
            resets = np.asarray(self._is_first[start:end], dtype=bool).copy()
            resets[0] = True
            batch_reset.append(resets)

        return {
            "obs": {k: np.stack(v, axis=0) for k, v in batch_obs.items()},
            "act": {k: np.stack(v, axis=0) for k, v in batch_act.items()},
            "rew": np.stack(batch_rew, axis=0),
            "reset": np.stack(batch_reset, axis=0),
        }
