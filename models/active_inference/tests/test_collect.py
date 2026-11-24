"""ランダム収集の単位。既定はエピソード、random_episodes==0 のときはステップ。"""

from __future__ import annotations

import jax
import numpy as np

from active_inference.config import Config
from active_inference.envs.point1d import Point1DEnv
from active_inference.replay_buffer import ReplayBuffer
from active_inference.train import collect_episode, collect_random


def test_collect_episode_stops_at_max_steps():
    env = Point1DEnv(discrete=True, max_steps=10, seed=0)
    cfg = Config()
    key = jax.random.PRNGKey(0)
    ep, _, _ = collect_episode(env, None, cfg, key, random=True, max_steps=4)
    assert len(ep["rew"]) == 4
    assert len(ep["done"]) == 4
    assert len(ep["obs"]["position"]) == 5
    assert ep["done"][-1] is False


def test_collect_episode_runs_until_done():
    env = Point1DEnv(discrete=True, max_steps=10, seed=0)
    cfg = Config()
    key = jax.random.PRNGKey(0)
    ep, _, _ = collect_episode(env, None, cfg, key, random=True)
    assert len(ep["rew"]) == 10
    assert ep["done"][-1] is True


def test_random_steps_when_episodes_is_zero():
    env = Point1DEnv(discrete=True, max_steps=10, seed=1)
    cfg = Config(random_episodes=0, random_steps=25)
    key = jax.random.PRNGKey(1)
    chunks, returns, _ = collect_random(env, cfg, key)
    lengths = [len(ep["rew"]) for ep in chunks]
    assert lengths == [10, 10, 5]
    assert sum(lengths) == 25
    assert chunks[0]["done"][-1] is True
    assert chunks[1]["done"][-1] is True
    assert chunks[2]["done"][-1] is False
    assert len(returns) == 3

    buffer = ReplayBuffer()
    for ep in chunks:
        buffer.add_episode(ep)
    batch = buffer.sample_paths(2, 8, np.random.default_rng(0))
    assert batch["rew"].shape == (2, 8)
    assert batch["reset"].shape == (2, 8)


def test_random_episodes_ignores_random_steps():
    env = Point1DEnv(discrete=True, max_steps=6, seed=2)
    cfg = Config(random_episodes=2, random_steps=100)
    key = jax.random.PRNGKey(2)
    chunks, _, _ = collect_random(env, cfg, key)
    assert len(chunks) == 2
    assert [len(ep["rew"]) for ep in chunks] == [6, 6]
    assert all(ep["done"][-1] is True for ep in chunks)


def test_random_steps_required_when_episodes_is_zero():
    env = Point1DEnv(discrete=True, max_steps=5, seed=3)
    cfg = Config(random_episodes=0, random_steps=0)
    key = jax.random.PRNGKey(3)
    try:
        collect_random(env, cfg, key)
    except ValueError as exc:
        assert "random_steps" in str(exc)
    else:
        raise AssertionError("random_steps が 0 のときはエラーになるべき")


if __name__ == "__main__":
    test_collect_episode_stops_at_max_steps()
    test_collect_episode_runs_until_done()
    test_random_steps_when_episodes_is_zero()
    test_random_episodes_ignores_random_steps()
    test_random_steps_required_when_episodes_is_zero()
    print("test_collect: ok")
