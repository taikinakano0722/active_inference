"""RSSM の形状と scan の確認。"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from active_inference.rssm import RSSM


def test_prior_posterior_shapes():
    key = jax.random.PRNGKey(0)
    rssm = RSSM(action_dim=3, embed_dim=8, deter_dim=16, stoch_dim=4, hidden_dim=16, key=key)
    state = rssm.initial()
    action = jnp.zeros((3,), dtype=jnp.float32)
    embed = jnp.zeros((8,), dtype=jnp.float32)
    key, k1, k2 = jax.random.split(key, 3)
    prior_state, prior_dist = rssm.prior(state, action, k1)
    assert prior_state["deter"].shape == (16,)
    assert prior_state["stoch"].shape == (4,)
    post_state, _, post_dist = rssm.posterior(state, action, embed, k2)
    assert post_state["deter"].shape == (16,)
    kl = post_dist.kl_divergence(prior_dist)
    assert kl.shape == ()


def test_observe_imagine_scan():
    key = jax.random.PRNGKey(1)
    rssm = RSSM(action_dim=3, embed_dim=8, deter_dim=16, stoch_dim=4, hidden_dim=16, key=key)
    T = 5
    actions = jnp.zeros((T, 3), dtype=jnp.float32)
    embeds = jnp.zeros((T, 8), dtype=jnp.float32)
    resets = jnp.zeros((T,), dtype=bool).at[0].set(True)
    key, k1, k2 = jax.random.split(key, 3)
    final, outs = rssm.observe(rssm.initial(), actions, embeds, resets, k1)
    assert outs["deter"].shape == (T, 16)
    assert outs["stoch"].shape == (T, 4)
    assert outs["kl"].shape == (T,)
    final2, outs2 = rssm.imagine(rssm.initial(), actions, k2)
    assert outs2["deter"].shape == (T, 16)
    assert outs2["prior_entropy"].shape == (T,)


def test_categorical_stoch():
    key = jax.random.PRNGKey(2)
    rssm = RSSM(
        action_dim=2,
        embed_dim=4,
        deter_dim=8,
        stoch_dim=3,
        stoch_type="categorical",
        stoch_classes=5,
        hidden_dim=8,
        key=key,
    )
    state = rssm.initial()
    assert state["stoch"].shape == (3, 5)
    key, k1 = jax.random.split(key)
    prior_state, dist = rssm.prior(state, jnp.zeros((2,)), k1)
    assert prior_state["stoch"].shape == (3, 5)
    assert dist.entropy().shape == ()


if __name__ == "__main__":
    test_prior_posterior_shapes()
    test_observe_imagine_scan()
    test_categorical_stoch()
    print("test_rssm: ok")
