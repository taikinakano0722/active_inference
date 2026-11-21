"""観測・行動空間の定義。環境が宣言する空間が正本である。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class Space:
    """単一キーの空間。DreamerV3 の elements.Space に相当する。"""

    dtype: Any
    shape: tuple[int, ...]
    discrete: bool = False
    low: float | None = None
    high: float | None = None
    n: int | None = None  # 離散のクラス数。discrete=True のとき必須。

    def __post_init__(self) -> None:
        object.__setattr__(self, "shape", tuple(int(x) for x in self.shape))
        if self.discrete and self.n is None:
            raise ValueError("discrete Space には n（クラス数）が必要です。")

    @property
    def is_image(self) -> bool:
        return len(self.shape) == 3

    @property
    def is_vector(self) -> bool:
        return len(self.shape) <= 2

    def sample(self, rng: np.random.Generator | None = None) -> np.ndarray:
        """環境側（numpy）でのランダムサンプル。"""
        rng = rng or np.random.default_rng()
        if self.discrete:
            assert self.n is not None
            # one-hot
            idx = int(rng.integers(0, self.n))
            out = np.zeros(self.n, dtype=np.float32)
            out[idx] = 1.0
            return out
        low = -1.0 if self.low is None else float(self.low)
        high = 1.0 if self.high is None else float(self.high)
        return rng.uniform(low, high, size=self.shape).astype(np.float32)

    def zeros(self) -> np.ndarray:
        if self.discrete:
            assert self.n is not None
            return np.zeros(self.n, dtype=np.float32)
        return np.zeros(self.shape, dtype=np.asarray(1, dtype=self.dtype).dtype)


def is_image_space(space: Space) -> bool:
    return space.is_image


def is_vector_space(space: Space) -> bool:
    return space.is_vector


def flatten_action(action: dict[str, np.ndarray | jnp.ndarray], act_space: dict[str, Space]) -> jnp.ndarray:
    """行動辞書を1本のベクトルに連結する。離散は one-hot をそのまま使う。"""
    parts = []
    for key in sorted(act_space.keys()):
        a = action[key]
        parts.append(jnp.asarray(a, dtype=jnp.float32).reshape(-1))
    return jnp.concatenate(parts, axis=-1)


def action_dim(act_space: dict[str, Space]) -> int:
    total = 0
    for space in act_space.values():
        if space.discrete:
            assert space.n is not None
            total += space.n
        else:
            total += int(np.prod(space.shape))
    return total
