"""方策ネットと候補モード。"""

from __future__ import annotations

import itertools
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import distrax

from .config import Config
from .decoder import ObsDecoder, RewardDecoder
from .efe import evaluate_action_sequences
from .encoder import Encoder
from .networks import MLP
from .rssm import RSSM
from .spaces import Space, action_dim


class PolicyNet(eqx.Module):
    """潜在特徴 → 行動分布。単一サンプル。"""

    mlp: MLP
    discrete: bool = eqx.field(static=True)
    out_dim: int = eqx.field(static=True)
    low: float = eqx.field(static=True)
    high: float = eqx.field(static=True)
    min_std: float = eqx.field(static=True)

    def __init__(
        self,
        feat_dim: int,
        act_space: dict[str, Space],
        hidden_dim: int = 200,
        layers: int = 3,
        min_std: float = 1e-4,
        key: jax.Array | None = None,
    ):
        if key is None:
            key = jax.random.PRNGKey(0)
        if len(act_space) != 1:
            raise NotImplementedError("最初のモデルは行動キーが1つのみ対応です。")
        space = next(iter(act_space.values()))
        self.discrete = bool(space.discrete)
        if self.discrete:
            assert space.n is not None
            self.out_dim = int(space.n)
            out = self.out_dim
            self.low = 0.0
            self.high = 1.0
        else:
            self.out_dim = int(np.prod(space.shape))
            out = 2 * self.out_dim
            self.low = -1.0 if space.low is None else float(space.low)
            self.high = 1.0 if space.high is None else float(space.high)
        self.min_std = float(min_std)
        self.mlp = MLP(feat_dim, out, hidden_dim, layers, key=key)

    def __call__(self, feat: jnp.ndarray) -> distrax.Distribution:
        raw = self.mlp(feat)
        if self.discrete:
            return distrax.OneHotCategorical(logits=raw)
        mean, raw_std = jnp.split(raw, 2, axis=-1)
        # 平均を行動範囲へ押し込む。
        mid = 0.5 * (self.high + self.low)
        half = 0.5 * (self.high - self.low)
        mean = mid + half * jnp.tanh(mean)
        std = jax.nn.softplus(raw_std) + self.min_std
        return distrax.MultivariateNormalDiag(loc=mean, scale_diag=std)


def sample_action_sequences(
    act_space: dict[str, Space],
    horizon: int,
    n_candidates: int,
    key: jax.Array,
) -> jnp.ndarray:
    """
    候補行動列を作る。返り値は (N, T, action_dim)。
    離散で全列の数が n_candidates 以下なら全列挙。
    """
    if len(act_space) != 1:
        raise NotImplementedError("行動キーは1つのみ対応です。")
    space = next(iter(act_space.values()))
    dim = action_dim(act_space)

    if space.discrete:
        assert space.n is not None
        n = int(space.n)
        total = n ** horizon
        if total <= n_candidates:
            # 全列挙
            seqs = []
            for prod in itertools.product(range(n), repeat=horizon):
                onehots = []
                for a in prod:
                    v = np.zeros((n,), dtype=np.float32)
                    v[a] = 1.0
                    onehots.append(v)
                seqs.append(np.stack(onehots, axis=0))
            return jnp.asarray(np.stack(seqs, axis=0))
        # ランダム one-hot
        idx = jax.random.randint(key, (n_candidates, horizon), 0, n)
        return jax.nn.one_hot(idx, n, dtype=jnp.float32)

    low = -1.0 if space.low is None else float(space.low)
    high = 1.0 if space.high is None else float(space.high)
    return jax.random.uniform(
        key,
        (n_candidates, horizon, dim),
        minval=low,
        maxval=high,
        dtype=jnp.float32,
    )


def _plan_with_seqs(
    rssm: RSSM,
    reward_decoder: RewardDecoder,
    encoder: Encoder,
    obs_decoder: ObsDecoder,
    state: dict[str, jnp.ndarray],
    action_seqs: jnp.ndarray,
    config: Config,
    key: jax.Array,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    efes = evaluate_action_sequences(
        rssm,
        reward_decoder,
        encoder,
        obs_decoder,
        state,
        action_seqs,
        config,
        key,
    )
    best = jnp.argmin(efes)
    best_seq = action_seqs[best]
    return best_seq[0], efes[best], jnp.mean(efes)


class CandidatePlanner:
    """候補モード。行動空間から列を作り、EFE 最小を選ぶ。"""

    def plan(
        self,
        rssm: RSSM,
        reward_decoder: RewardDecoder,
        encoder: Encoder,
        obs_decoder: ObsDecoder,
        state: dict[str, jnp.ndarray],
        act_space: dict[str, Space],
        config: Config,
        key: jax.Array,
    ) -> tuple[jnp.ndarray, dict[str, Any]]:
        key, k_samp, k_eval = jax.random.split(key, 3)
        action_seqs = sample_action_sequences(
            act_space,
            config.horizon,
            config.candidate_count,
            k_samp,
        )
        # numpy 全列挙の結果を JAX に載せた場合もある。
        action_seqs = jnp.asarray(action_seqs, dtype=jnp.float32)
        first_action, best_efe, mean_efe = _plan_with_seqs(
            rssm,
            reward_decoder,
            encoder,
            obs_decoder,
            state,
            action_seqs,
            config,
            k_eval,
        )
        return first_action, {
            "best_efe": best_efe,
            "mean_efe": mean_efe,
            "n_candidates": action_seqs.shape[0],
        }
