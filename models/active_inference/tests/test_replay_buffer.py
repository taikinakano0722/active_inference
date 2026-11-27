"""リプレイの切り出し。episode は1件ずつ、stream は1本の列。"""

from __future__ import annotations

import numpy as np

from active_inference.replay_buffer import ReplayBuffer, StreamReplayBuffer


def _episode(length: int, *, done_at: int | None = None) -> dict:
    """報酬が 0, 1, 2, ... の記録。done_at を指定した位置だけ終わりにする。"""
    obs = {
        "position": [np.float32(i) for i in range(length + 1)],
    }
    act = {
        "velocity": [np.float32(i) for i in range(length)],
    }
    done = [False] * length
    if done_at is not None:
        done[done_at] = True
    return {
        "obs": obs,
        "act": act,
        "rew": [float(i) for i in range(length)],
        "done": done,
    }


def _add_stream_step(
    buffer: StreamReplayBuffer,
    index: int,
    *,
    is_first: bool,
    obs: float | None = None,
    next_obs: float | None = None,
) -> None:
    before = float(index if obs is None else obs)
    after = float(index + 10 if next_obs is None else next_obs)
    buffer.add_step(
        {"position": np.float32(before)},
        {"velocity": np.float32(index)},
        float(index),
        False,
        is_first,
        {"position": np.float32(after)},
    )


def test_episode_window_start_is_always_reset():
    buffer = ReplayBuffer()
    buffer.add_episode(_episode(8, done_at=7))
    batch = buffer.sample_paths(24, 3, np.random.default_rng(0))
    assert batch["reset"].shape == (24, 3)
    assert batch["reset"][:, 0].all()
    assert not batch["reset"][:, 1:].any()
    # 先頭以外から切った区間でも、先頭の reset は真。
    started_later = batch["rew"][:, 0] != 0
    assert started_later.any()
    assert batch["reset"][started_later, 0].all()


def test_episode_rejects_done_before_the_last_step():
    buffer = ReplayBuffer()
    try:
        buffer.add_episode(_episode(4, done_at=1))
    except ValueError as exc:
        assert "done" in str(exc)
    else:
        raise AssertionError("途中の done はエラーになるべき")
    # 最後だけ真、または一度も真でなくても入れられる。
    buffer.add_episode(_episode(4, done_at=3))
    buffer.add_episode(_episode(4, done_at=None))
    assert len(buffer) == 2


def test_stream_drops_oldest_steps():
    buffer = StreamReplayBuffer(capacity=3)
    for i in range(5):
        _add_stream_step(buffer, i, is_first=(i == 0))
    assert len(buffer) == 3
    assert buffer.can_sample(3)
    assert not buffer.can_sample(4)
    batch = buffer.sample_paths(1, 3, np.random.default_rng(0))
    assert list(batch["rew"][0]) == [2.0, 3.0, 4.0]


def test_stream_resets_at_episode_boundary():
    buffer = StreamReplayBuffer(capacity=10)
    # 1本目の行動後観測は 10, 11。2本目は 100 から始まるので、列の観測はそこで飛ぶ。
    _add_stream_step(buffer, 0, is_first=True, obs=0, next_obs=10)
    _add_stream_step(buffer, 1, is_first=False, obs=10, next_obs=11)
    _add_stream_step(buffer, 2, is_first=True, obs=100, next_obs=110)
    _add_stream_step(buffer, 3, is_first=False, obs=110, next_obs=111)
    batch = buffer.sample_paths(1, 4, np.random.default_rng(0))
    assert list(batch["reset"][0]) == [True, False, True, False]
    assert list(batch["obs"]["position"][0]) == [0.0, 10.0, 11.0, 110.0, 111.0]


def test_stream_window_start_is_always_reset():
    buffer = StreamReplayBuffer(capacity=10)
    for i in range(6):
        _add_stream_step(buffer, i, is_first=(i == 0))
    batch = buffer.sample_paths(24, 3, np.random.default_rng(1))
    assert batch["reset"][:, 0].all()
    started_later = batch["rew"][:, 0] != 0
    assert started_later.any()
    assert batch["reset"][started_later, 0].all()
    # 本当の開始は列の先頭だけなので、2歩目以降の reset は立たない。
    assert not batch["reset"][:, 1:].any()


if __name__ == "__main__":
    test_episode_window_start_is_always_reset()
    test_episode_rejects_done_before_the_last_step()
    test_stream_drops_oldest_steps()
    test_stream_resets_at_episode_boundary()
    test_stream_window_start_is_always_reset()
    print("test_replay_buffer: ok")
