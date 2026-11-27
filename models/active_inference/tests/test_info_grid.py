"""InfoGridEnv の動作確認。"""

from __future__ import annotations

import numpy as np

from active_inference.envs.info_grid import (
    CELL_C,
    CELL_K,
    CUE,
    EMPTY,
    InfoGridEnv,
    REWARD_L,
    REWARD_R,
    START,
    SYM_A,
    SYM_B,
    SYM_K,
)


def _one_hot(idx: int) -> np.ndarray:
    a = np.zeros(4, dtype=np.float32)
    a[idx] = 1.0
    return a


def test_reward_wall_bump():
    env = InfoGridEnv(mode="reward", seed=0)
    obs, info = env.reset()
    assert info["position"] == START
    assert np.allclose(obs["cell"], EMPTY)
    # 左へ出ると外壁。位置は変わらない。
    obs, reward, done, info = env.step(_one_hot(0))
    assert info["position"] == START
    assert info["bumped"] is True
    assert reward == 0.0
    assert not done


def test_reward_hit_locks_and_one_shot():
    env = InfoGridEnv(mode="reward", seed=0, lock_steps=4)
    # 正しい側を L に固定するため、reset 後に内部を上書きする。
    env.reset()
    env._correct_side = "L"
    env._cue_symbol = SYM_A.copy()
    # 近道で L へ: 上, 上, 右
    path = [_one_hot(2), _one_hot(2), _one_hot(1)]
    total = 0.0
    done = False
    for a in path:
        obs, reward, done, info = env.step(a)
        total += reward
    assert info["position"] == REWARD_L
    assert total == 1.0
    assert info["locked"] is True
    assert not done  # まだ 4 歩残る

    # locked 中は動けない。報酬は 0。
    for i in range(4):
        obs, reward, done, info = env.step(_one_hot(3))  # 下へ出ようとしても留まる
        assert info["position"] == REWARD_L
        assert reward == 0.0
        assert info["locked"] is True
        if i < 3:
            assert not done
    assert done is True


def test_reward_miss_then_cannot_take_other():
    env = InfoGridEnv(mode="reward", seed=1, lock_steps=4)
    env.reset()
    env._correct_side = "L"
    env._cue_symbol = SYM_A.copy()
    # 近道で R（外れ）へ: 下, 下, 右
    for a in [_one_hot(3), _one_hot(3), _one_hot(1)]:
        obs, reward, done, info = env.step(a)
    assert info["position"] == REWARD_R
    assert reward == -0.5
    assert info["locked"] is True
    assert info["collected"] is True
    # locked 中に L へ行けない。
    for _ in range(4):
        obs, reward, done, info = env.step(_one_hot(2))
        assert info["position"] == REWARD_R
        assert reward == 0.0
    assert done is True


def test_reward_timeout_without_collect():
    env = InfoGridEnv(mode="reward", seed=2, max_steps=5, lock_steps=4)
    env.reset()
    # スタートで左を繰り返す。
    done = False
    for _ in range(5):
        _, reward, done, info = env.step(_one_hot(0))
        assert reward == 0.0
        assert info["collected"] is False
    assert done is True


def test_cue_symbol_matches_side():
    env = InfoGridEnv(mode="reward", seed=3)
    env.reset()
    env._correct_side = "R"
    env._cue_symbol = SYM_B.copy()
    # 手がかりへ: 右, 右
    env.step(_one_hot(1))
    obs, _, _, info = env.step(_one_hot(1))
    assert info["position"] == CUE
    assert np.allclose(obs["cell"], SYM_B)


def test_no_reward_k_and_c():
    env = InfoGridEnv(mode="no_reward", seed=0)
    env.reset()
    # K へ: 上, 上
    env.step(_one_hot(2))
    obs, reward, done, info = env.step(_one_hot(2))
    assert info["position"] == CELL_K
    assert np.allclose(obs["cell"], SYM_K)
    assert reward == 0.0
    assert info["first_visit"] == "K"
    assert done is True

    env.reset()
    env._cue_symbol = SYM_A.copy()
    # C へ: 右, 右
    env.step(_one_hot(1))
    obs, reward, done, info = env.step(_one_hot(1))
    assert info["position"] == CELL_C
    assert np.allclose(obs["cell"], SYM_A)
    assert info["first_visit"] == "C"
    assert done is True


def test_no_reward_neither():
    env = InfoGridEnv(mode="no_reward", seed=0, max_steps=8)
    env.reset()
    done = False
    for _ in range(8):
        _, _, done, info = env.step(_one_hot(0))  # 左へ壁打ち
        assert info["first_visit"] is None
    assert done is True


if __name__ == "__main__":
    test_reward_wall_bump()
    test_reward_hit_locks_and_one_shot()
    test_reward_miss_then_cannot_take_other()
    test_reward_timeout_without_collect()
    test_cue_symbol_matches_side()
    test_no_reward_k_and_c()
    test_no_reward_neither()
    print("test_info_grid: ok")
