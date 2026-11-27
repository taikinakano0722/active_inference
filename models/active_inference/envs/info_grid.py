"""情報利得評価用の格子環境。

報酬版は 3x5、報酬なし版は 3x3。行動は上下左右の 4 つ。
外壁の外へ出る行動は、その場に留まる。
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from active_inference.spaces import Space

# 行動番号: 0=左, 1=右, 2=上, 3=下
_DELTA = {
    0: (-1, 0),
    1: (1, 0),
    2: (0, 1),
    3: (0, -1),
}

# 暫定の観測ベクトル。報酬マスの観測は空と同じで、正解側がバレないようにする。
EMPTY = np.array([0.0, 0.0, 0.0], dtype=np.float32)
SYM_K = np.array([1.0, 0.0, 0.0], dtype=np.float32)
SYM_A = np.array([0.0, 1.0, 0.0], dtype=np.float32)  # 手がかり / C の記号 A（左=L）
SYM_B = np.array([0.0, 0.0, 1.0], dtype=np.float32)  # 手がかり / C の記号 B（右=R）

START = (0, 0)
CUE = (2, 0)
REWARD_L = (1, 2)
REWARD_R = (1, -2)
CELL_K = (0, 2)
CELL_C = (2, 0)


class InfoGridEnv:
    """情報利得の評価用格子。mode で報酬版と報酬なし版を切り替える。"""

    def __init__(
        self,
        mode: Literal["reward", "no_reward"] = "reward",
        seed: int | None = None,
        max_steps: int | None = None,
        lock_steps: int = 4,
    ):
        if mode not in ("reward", "no_reward"):
            raise ValueError("mode は 'reward' か 'no_reward' にしてください。")
        self.mode = mode
        self.lock_steps = int(lock_steps)
        if mode == "reward":
            self.x_min, self.x_max = 0, 2
            self.y_min, self.y_max = -2, 2
            self.max_steps = 20 if max_steps is None else int(max_steps)
        else:
            self.x_min, self.x_max = 0, 2
            self.y_min, self.y_max = 0, 2
            self.max_steps = 8 if max_steps is None else int(max_steps)

        self._rng = np.random.default_rng(seed)
        self.obs_space: dict[str, Space] = {
            "cell": Space(np.float32, (3,)),
        }
        self.act_space: dict[str, Space] = {
            "action": Space(np.float32, (4,), discrete=True, n=4),
        }

        self._x = 0
        self._y = 0
        self._step = 0
        self._done = True
        self._correct_side: Literal["L", "R"] = "L"
        self._cue_symbol = SYM_A.copy()
        self._locked = False
        self._lock_remaining = 0
        self._collected = False
        self._first_visit: str | None = None

    def seed(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)

    @property
    def position(self) -> tuple[int, int]:
        return (self._x, self._y)

    @property
    def cells(self) -> list[tuple[int, int]]:
        """歩けるマスの一覧。"""
        return [
            (x, y)
            for x in range(self.x_min, self.x_max + 1)
            for y in range(self.y_min, self.y_max + 1)
        ]

    def _in_bounds(self, x: int, y: int) -> bool:
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max

    def _obs_at(self, x: int, y: int) -> np.ndarray:
        pos = (x, y)
        if self.mode == "reward":
            if pos == CUE:
                return self._cue_symbol.copy()
            return EMPTY.copy()
        # no_reward
        if pos == CELL_K:
            return SYM_K.copy()
        if pos == CELL_C:
            return self._cue_symbol.copy()
        return EMPTY.copy()

    def _obs(self) -> dict[str, np.ndarray]:
        return {"cell": self._obs_at(self._x, self._y)}

    def _action_to_idx(self, action: np.ndarray | dict[str, np.ndarray]) -> int:
        if isinstance(action, dict):
            action = action["action"]
        action = np.asarray(action, dtype=np.float32)
        if action.ndim == 0:
            idx = int(action)
        elif action.shape[-1] == 4:
            idx = int(np.argmax(action))
        else:
            idx = int(action.reshape(-1)[0])
        return int(np.clip(idx, 0, 3))

    def reset(self, seed: int | None = None) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        if seed is not None:
            self.seed(seed)
        self._x, self._y = START
        self._step = 0
        self._done = False
        self._locked = False
        self._lock_remaining = 0
        self._collected = False
        self._first_visit = None
        if self._rng.random() < 0.5:
            self._correct_side = "L"
            self._cue_symbol = SYM_A.copy()
        else:
            self._correct_side = "R"
            self._cue_symbol = SYM_B.copy()
        info = {
            "is_first": True,
            "position": (self._x, self._y),
            "correct_side": self._correct_side,
            "first_visit": None,
        }
        return self._obs(), info

    def step(
        self, action: np.ndarray | dict[str, np.ndarray]
    ) -> tuple[dict[str, np.ndarray], float, bool, dict[str, Any]]:
        if self._done:
            raise RuntimeError("エピソード終了後に step が呼ばれました。reset してください。")

        idx = self._action_to_idx(action)
        dx, dy = _DELTA[idx]
        bumped = False
        reward = 0.0

        if self._locked:
            # 報酬を得たあとは動けない。報酬は 0。
            self._lock_remaining -= 1
            self._step += 1
            if self._lock_remaining <= 0:
                self._done = True
        else:
            nx, ny = self._x + dx, self._y + dy
            if not self._in_bounds(nx, ny):
                bumped = True
                nx, ny = self._x, self._y
            self._x, self._y = nx, ny
            self._step += 1

            if self.mode == "reward":
                pos = (self._x, self._y)
                if pos in (REWARD_L, REWARD_R) and not self._collected:
                    correct = REWARD_L if self._correct_side == "L" else REWARD_R
                    reward = 1.0 if pos == correct else -0.5
                    self._collected = True
                    self._locked = True
                    self._lock_remaining = self.lock_steps
                if self._step >= self.max_steps and not self._locked:
                    self._done = True
            else:
                # no_reward: 報酬は常に 0。K か C に着いたら終わる。
                pos = (self._x, self._y)
                if pos == CELL_K:
                    self._first_visit = "K"
                    self._done = True
                elif pos == CELL_C:
                    self._first_visit = "C"
                    self._done = True
                elif self._step >= self.max_steps:
                    self._done = True

        info = {
            "is_first": False,
            "is_last": self._done,
            "position": (self._x, self._y),
            "bumped": bumped,
            "locked": self._locked,
            "correct_side": self._correct_side,
            "first_visit": self._first_visit,
            "collected": self._collected,
        }
        return self._obs(), float(reward), self._done, info

    def sample_action(self) -> dict[str, np.ndarray]:
        return {k: v.sample(self._rng) for k, v in self.act_space.items()}
