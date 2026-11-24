"""期待自由エネルギー（実用項 + 情報利得）。"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import distrax

from .config import Config
from .decoder import ObsDecoder, RewardDecoder
from .encoder import Encoder
from .rssm import RSSM


def compute_efe_from_parts(
    reward_mean: jnp.ndarray,
    prior_entropy: jnp.ndarray,
    epistemic_kl: jnp.ndarray | None,
    config: Config,
) -> jnp.ndarray:
    """スカラーの EFE。小さいほど良い。"""
    pragmatic = -reward_mean
    if config.efe_mode == "entropy":
        epistemic = prior_entropy
    elif config.efe_mode == "kl":
        if epistemic_kl is None:
            raise ValueError("efe_mode='kl' には epistemic_kl が必要です。")
        epistemic = epistemic_kl
    else:
        raise ValueError(f"未知の efe_mode: {config.efe_mode}")
    return (
        config.pragmatic_weight * pragmatic
        - config.epistemic_weight * epistemic
    )


def _kl_from_imagined_obs(
    rssm: RSSM,
    encoder: Encoder,
    obs_decoder: ObsDecoder,
    state: dict[str, jnp.ndarray],
    prior_stats: jnp.ndarray,
    key: jax.Array,
) -> jnp.ndarray:
    """想像した観測の平均をエンコードし直し、事後と事前の KL を取る。"""
    feat = rssm.feature(state)
    dists = obs_decoder(feat)
    # デコーダ平均を観測辞書に戻す。
    obs: dict[str, jnp.ndarray] = {}
    for name in encoder.vec_keys:
        mean = dists[name].mode().reshape(obs_decoder.vec_shapes[name])
        obs[name] = mean
    for name in encoder.img_keys:
        mean = dists[name].mode().reshape(obs_decoder.img_shapes[name])
        # エンコーダは 0-255 想定なので戻す。
        obs[name] = jnp.clip(mean * 255.0, 0.0, 255.0)
    embed = encoder(obs)
    # 行動ゼロで「同じ deter 上の事後」を近似するため、
    # deter を固定し、posterior_net だけを再計算する。
    # 事後の入力は deter と埋め込みだけで、事前サンプルは含めない。
    prior_dist = rssm.dist_from_stats(prior_stats)
    x = jnp.concatenate(
        [
            state["deter"],
            embed.reshape(-1),
        ],
        axis=-1,
    )
    stats = rssm.posterior_net(x)
    post_dist = rssm.dist_from_stats(stats)
    return post_dist.kl_divergence(prior_dist)


def evaluate_action_sequence(
    rssm: RSSM,
    reward_decoder: RewardDecoder,
    encoder: Encoder,
    obs_decoder: ObsDecoder,
    init_state: dict[str, jnp.ndarray],
    actions: jnp.ndarray,
    config: Config,
    key: jax.Array,
) -> jnp.ndarray:
    """1 本の行動列 (T, action_dim) の EFE 和。"""

    def step(carry, action):
        state, key = carry
        key, k_use = jax.random.split(key)
        state, prior_dist = rssm.prior(state, action, k_use)
        feat = rssm.feature(state)
        rew_dist = reward_decoder(feat)
        if config.efe_mode == "kl":
            kl = _kl_from_imagined_obs(
                rssm,
                encoder,
                obs_decoder,
                state,
                rssm._dist_stats(prior_dist),
                k_use,
            )
        else:
            kl = None
        efe = compute_efe_from_parts(
            rew_dist.mean(),
            prior_dist.entropy(),
            kl,
            config,
        )
        return (state, key), efe

    (_, _), efes = jax.lax.scan(step, (init_state, key), actions)
    return jnp.sum(efes)


def evaluate_action_sequences(
    rssm: RSSM,
    reward_decoder: RewardDecoder,
    encoder: Encoder,
    obs_decoder: ObsDecoder,
    init_state: dict[str, jnp.ndarray],
    action_seqs: jnp.ndarray,
    config: Config,
    key: jax.Array,
) -> jnp.ndarray:
    """複数候補 (N, T, action_dim) の EFE 和を返す。形は (N,)。"""
    n = action_seqs.shape[0]
    keys = jax.random.split(key, n)

    def one(actions, k):
        return evaluate_action_sequence(
            rssm,
            reward_decoder,
            encoder,
            obs_decoder,
            init_state,
            actions,
            config,
            k,
        )

    return jax.vmap(one)(action_seqs, keys)
