"""EFE 計算の確認。"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from active_inference.agent import Agent
from active_inference.config import Config
from active_inference.efe import compute_efe_from_parts, evaluate_action_sequences
from active_inference.envs.point1d import Point1DEnv
from active_inference.policy import sample_action_sequences


def test_pragmatic_prefers_high_reward():
    cfg = Config(pragmatic_weight=1.0, epistemic_weight=0.0, efe_mode="entropy")
    low = compute_efe_from_parts(jnp.array(0.1), jnp.array(1.0), None, cfg)
    high = compute_efe_from_parts(jnp.array(0.9), jnp.array(1.0), None, cfg)
    assert float(high) < float(low)


def test_candidate_sequences_shape():
    env = Point1DEnv(discrete=True, seed=0)
    cfg = Config(
        deter_dim=16,
        stoch_dim=4,
        hidden_dim=16,
        vec_units=16,
        horizon=3,
        candidate_count=27,
        efe_mode="entropy",
        epistemic_weight=0.0,
    )
    key = jax.random.PRNGKey(0)
    agent = Agent.create(env.obs_space, env.act_space, cfg, key)
    key, k1, k2 = jax.random.split(key, 3)
    seqs = sample_action_sequences(env.act_space, cfg.horizon, cfg.candidate_count, k1)
    assert seqs.shape[0] == 27
    assert seqs.shape[1] == 3
    efes = evaluate_action_sequences(
        agent.rssm,
        agent.reward_decoder,
        agent.encoder,
        agent.obs_decoder,
        agent.rssm.initial(),
        seqs,
        cfg,
        k2,
    )
    assert efes.shape == (27,)
    assert np.isfinite(np.asarray(efes)).all()


if __name__ == "__main__":
    test_pragmatic_prefers_high_reward()
    test_candidate_sequences_shape()
    print("test_efe: ok")
