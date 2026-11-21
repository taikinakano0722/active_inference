"""世界モデル損失が下がることの確認。"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from active_inference.agent import Agent, make_wm_step, world_model_loss
from active_inference.config import Config
from active_inference.envs.point1d import Point1DEnv
from active_inference.replay_buffer import ReplayBuffer
from active_inference.train import batch_to_jax, collect_episode


def test_world_model_loss_decreases():
    env = Point1DEnv(discrete=True, max_steps=30, seed=0)
    cfg = Config(
        deter_dim=32,
        stoch_dim=8,
        hidden_dim=32,
        vec_units=16,
        batch_size=8,
        seq_len=16,
        free_nats=0.5,
        lr=1e-3,
    )
    key = jax.random.PRNGKey(0)
    key, k_agent = jax.random.split(key)
    agent = Agent.create(env.obs_space, env.act_space, cfg, k_agent)
    buffer = ReplayBuffer()
    for _ in range(12):
        key, k_ep = jax.random.split(key)
        ep, _, key = collect_episode(env, None, cfg, k_ep, random=True)
        buffer.add_episode(ep)

    opt, step = make_wm_step(cfg)
    opt_state = opt.init(
        __import__("equinox").filter(agent, __import__("equinox").is_inexact_array)
    )
    rng = np.random.default_rng(0)
    losses = []
    for i in range(30):
        batch = batch_to_jax(buffer.sample_paths(cfg.batch_size, cfg.seq_len, rng))
        key, k_step = jax.random.split(key)
        agent, opt_state, loss, metrics = step(agent, opt_state, batch, k_step)
        losses.append(float(loss))
    # 後半平均が前半より小さいこと。
    assert np.mean(losses[-10:]) < np.mean(losses[:10])


if __name__ == "__main__":
    test_world_model_loss_decreases()
    print("test_agent: ok")
