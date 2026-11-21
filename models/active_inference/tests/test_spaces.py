"""Space の基本確認。"""

from __future__ import annotations

import numpy as np

from active_inference.spaces import Space, action_dim, flatten_action


def test_discrete_and_continuous():
    disc = Space(np.float32, (3,), discrete=True, n=3)
    cont = Space(np.float32, (1,), low=-0.1, high=0.1)
    a = disc.sample()
    assert a.shape == (3,)
    assert abs(a.sum() - 1.0) < 1e-5
    b = cont.sample()
    assert b.shape == (1,)
    assert -0.1 <= b[0] <= 0.1


def test_action_dim_and_flatten():
    act_space = {"action": Space(np.float32, (3,), discrete=True, n=3)}
    assert action_dim(act_space) == 3
    flat = flatten_action({"action": np.array([0, 1, 0], dtype=np.float32)}, act_space)
    assert flat.shape == (3,)


if __name__ == "__main__":
    test_discrete_and_continuous()
    test_action_dim_and_flatten()
    print("test_spaces: ok")
