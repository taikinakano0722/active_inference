"""Point1D に画像観測を足した環境。第二段の確認用。"""

from __future__ import annotations

from typing import Any

import numpy as np

from active_inference.envs.point1d import Point1DEnv
from active_inference.spaces import Space


class Point1DImageEnv(Point1DEnv):
    """position/goal に加え、1D 位置を描いた小さな画像を返す。"""

    def __init__(self, image_size: int = 32, **kwargs):
        super().__init__(**kwargs)
        self.image_size = int(image_size)
        self.obs_space = {
            **self.obs_space,
            "image": Space(np.uint8, (self.image_size, self.image_size, 1)),
        }

    def _render(self) -> np.ndarray:
        img = np.zeros((self.image_size, self.image_size, 1), dtype=np.uint8)
        # x 座標を列に写す。行中央に点を打つ。
        def to_col(x: float) -> int:
            t = (x - self.world_low) / (self.world_high - self.world_low)
            return int(np.clip(t * (self.image_size - 1), 0, self.image_size - 1))

        row = self.image_size // 2
        img[row, to_col(self._goal), 0] = 128
        img[row, to_col(self._position), 0] = 255
        return img

    def _obs(self) -> dict[str, np.ndarray]:
        obs = super()._obs()
        obs["image"] = self._render()
        return obs
