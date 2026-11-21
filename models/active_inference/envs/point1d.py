"""1 次元の位置積分環境。

観測キーは position と goal。行動は速度。速さに上限があり、
1 ステップでは目標に届かない。離散は負方向・停止・正方向の 3 つ。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from active_inference.spaces import Space


class Point1DEnv:
    """自分の位置を速度で動かし、目標に近づくほど報酬が高い環境。"""

    def __init__(
        self,
        discrete: bool = False,
        max_speed: float = 0.1,
        world_low: float = -1.0,
        world_high: float = 1.0,
        max_steps: int = 100,
        seed: int | None = None,
    ):
        if max_speed <= 0:
            raise ValueError("max_speed は正である必要があります。")
        self.discrete = discrete
        self.max_speed = float(max_speed)
        self.world_low = float(world_low)
        self.world_high = float(world_high)
        self.max_steps = int(max_steps)
        self._rng = np.random.default_rng(seed)

        self.obs_space: dict[str, Space] = {
            "position": Space(np.float32, (1,)),
            "goal": Space(np.float32, (1,)),
        }
        if discrete:
            # one-hot: [負方向, 停止, 正方向]
            self.act_space: dict[str, Space] = {
                "action": Space(np.float32, (3,), discrete=True, n=3),
            }
            self._discrete_speeds = np.array(
                [-self.max_speed, 0.0, self.max_speed], dtype=np.float32
            )
        else:
            self.act_space = {
                "action": Space(
                    np.float32,
                    (1,),
                    discrete=False,
                    low=-self.max_speed,
                    high=self.max_speed,
                ),
            }

        self._position = 0.0
        self._goal = 0.0
        self._step = 0
        self._done = True

    def seed(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)

    def _obs(self) -> dict[str, np.ndarray]:
        return {
            "position": np.array([self._position], dtype=np.float32),
            "goal": np.array([self._goal], dtype=np.float32),
        }

    def _reward(self) -> float:
        # 近いほど高い。距離 0 で 1、離れるほど減衰。
        dist = abs(self._position - self._goal)
        return float(np.exp(-(dist ** 2) / 0.1))

    def reset(self, seed: int | None = None) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        if seed is not None:
            self.seed(seed)
        self._position = float(self._rng.uniform(self.world_low, self.world_high))
        self._goal = float(self._rng.uniform(self.world_low, self.world_high))
        # 目標が近すぎると 1 ステップ判定になりやすいので、最低距離を確保する。
        min_gap = 2.5 * self.max_speed
        while abs(self._position - self._goal) < min_gap:
            self._goal = float(self._rng.uniform(self.world_low, self.world_high))
        self._step = 0
        self._done = False
        return self._obs(), {"is_first": True}

    def _action_to_speed(self, action: np.ndarray | dict[str, np.ndarray]) -> float:
        if isinstance(action, dict):
            action = action["action"]
        action = np.asarray(action, dtype=np.float32)
        if self.discrete:
            if action.ndim == 0:
                idx = int(action)
            elif action.shape[-1] == 3:
                idx = int(np.argmax(action))
            else:
                idx = int(action.reshape(-1)[0])
            idx = int(np.clip(idx, 0, 2))
            return float(self._discrete_speeds[idx])
        speed = float(action.reshape(-1)[0])
        return float(np.clip(speed, -self.max_speed, self.max_speed))

    def step(
        self, action: np.ndarray | dict[str, np.ndarray]
    ) -> tuple[dict[str, np.ndarray], float, bool, dict[str, Any]]:
        if self._done:
            raise RuntimeError("エピソード終了後に step が呼ばれました。reset してください。")
        speed = self._action_to_speed(action)
        self._position = float(
            np.clip(self._position + speed, self.world_low, self.world_high)
        )
        self._step += 1
        reward = self._reward()
        self._done = self._step >= self.max_steps
        info = {
            "is_first": False,
            "is_last": self._done,
            "speed": speed,
            "distance": abs(self._position - self._goal),
        }
        return self._obs(), reward, self._done, info

    def sample_action(self) -> dict[str, np.ndarray]:
        return {k: v.sample(self._rng) for k, v in self.act_space.items()}
