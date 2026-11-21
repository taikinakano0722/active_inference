"""第二段: 画像キー付き環境でエージェントが組めること。"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from active_inference.agent import Agent, make_wm_step
from active_inference.config import Config
from active_inference.envs.point1d_image import Point1DImageEnv
from active_inference.replay_buffer import ReplayBuffer
from active_inference.train import batch_to_jax, collect_episode


def test_agent_with_image_obs():
    env = Point1DImageEnv(discrete=True, image_size=32, max_steps=20, seed=0)
    assert "image" in env.obs_space
    cfg = Config(
        deter_dim=32,
        stoch_dim=8,
        hidden_dim=32,
        vec_units=16,
        img_depths=(16, 32),
        img_kernel=4,
        batch_size=4,
        seq_len=10,
        lr=1e-3,
    )
    key = jax.random.PRNGKey(0)
    agent = Agent.create(env.obs_space, env.act_space, cfg, key)
    assert agent.encoder.img_keys == ("image",)
    assert "image" in agent.obs_decoder.img_keys

    buffer = ReplayBuffer()
    for _ in range(6):
        key, k_ep = jax.random.split(key)
        ep, _, key = collect_episode(env, None, cfg, k_ep, random=True)
        buffer.add_episode(ep)

    import equinox as eqx

    opt, step = make_wm_step(cfg)
    opt_state = opt.init(eqx.filter(agent, eqx.is_inexact_array))
    rng = np.random.default_rng(0)
    batch = batch_to_jax(buffer.sample_paths(cfg.batch_size, cfg.seq_len, rng))
    key, k_step = jax.random.split(key)
    agent, opt_state, loss, metrics = step(agent, opt_state, batch, k_step)
    assert np.isfinite(float(loss))


if __name__ == "__main__":
    test_agent_with_image_obs()
    print("test_image_env: ok")
