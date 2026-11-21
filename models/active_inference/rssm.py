"""RSSM: 決定的 deter と確率的 stoch を持つ再帰状態空間モデル。"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import distrax

from .networks import GRUCell, MLP


def softplus_std(raw: jnp.ndarray, min_std: float = 0.1) -> jnp.ndarray:
    return jax.nn.softplus(raw) + min_std


class RSSM(eqx.Module):
    """単一サンプルの prior / posterior。バッチ化は呼び出し側の vmap。"""

    gru: GRUCell
    rnn_in: eqx.nn.Linear
    prior_net: MLP
    posterior_net: MLP
    deter_dim: int = eqx.field(static=True)
    stoch_dim: int = eqx.field(static=True)
    stoch_type: str = eqx.field(static=True)
    stoch_classes: int = eqx.field(static=True)
    action_dim: int = eqx.field(static=True)
    embed_dim: int = eqx.field(static=True)
    min_std: float = eqx.field(static=True)

    def __init__(
        self,
        action_dim: int,
        embed_dim: int,
        deter_dim: int = 200,
        stoch_dim: int = 30,
        hidden_dim: int = 200,
        stoch_type: str = "gaussian",
        stoch_classes: int = 32,
        min_std: float = 0.1,
        key: jax.Array | None = None,
    ):
        if key is None:
            key = jax.random.PRNGKey(0)
        if stoch_type not in ("gaussian", "categorical"):
            raise ValueError(f"未知の stoch_type: {stoch_type}")
        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.action_dim = int(action_dim)
        self.embed_dim = int(embed_dim)
        self.deter_dim = int(deter_dim)
        self.stoch_dim = int(stoch_dim)
        self.stoch_type = stoch_type
        self.stoch_classes = int(stoch_classes)
        self.min_std = float(min_std)

        stoch_flat = self._stoch_flat_dim()
        self.rnn_in = eqx.nn.Linear(action_dim + stoch_flat, hidden_dim, key=k1)
        self.gru = GRUCell(hidden_dim, deter_dim, key=k2)

        if stoch_type == "gaussian":
            prior_out = 2 * stoch_dim
            post_out = 2 * stoch_dim
        else:
            prior_out = stoch_dim * stoch_classes
            post_out = stoch_dim * stoch_classes

        self.prior_net = MLP(deter_dim, prior_out, hidden_dim, layers=2, key=k3)
        # posterior は deter + stoch_flat + embed
        self.posterior_net = MLP(
            deter_dim + stoch_flat + embed_dim,
            post_out,
            hidden_dim,
            layers=2,
            key=k4,
        )

    def _stoch_flat_dim(self) -> int:
        if self.stoch_type == "gaussian":
            return self.stoch_dim
        return self.stoch_dim * self.stoch_classes

    def initial(self) -> dict[str, jnp.ndarray]:
        """単一サンプルのゼロ状態。"""
        if self.stoch_type == "gaussian":
            stoch = jnp.zeros((self.stoch_dim,), dtype=jnp.float32)
        else:
            # 一様 one-hot
            stoch = jnp.ones((self.stoch_dim, self.stoch_classes), dtype=jnp.float32)
            stoch = stoch / self.stoch_classes
        return {
            "deter": jnp.zeros((self.deter_dim,), dtype=jnp.float32),
            "stoch": stoch,
        }

    def _flatten_stoch(self, stoch: jnp.ndarray) -> jnp.ndarray:
        return stoch.reshape(-1)

    def _stats_to_dist(self, stats: jnp.ndarray) -> distrax.Distribution:
        if self.stoch_type == "gaussian":
            mean, raw_std = jnp.split(stats, 2, axis=-1)
            std = softplus_std(raw_std, self.min_std)
            return distrax.MultivariateNormalDiag(loc=mean, scale_diag=std)
        logits = stats.reshape(self.stoch_dim, self.stoch_classes)
        # 各 stoch 変数は独立な Categorical。Independent でまとめる。
        return distrax.Independent(distrax.OneHotCategorical(logits=logits), 1)

    def _sample(self, dist: distrax.Distribution, key: jax.Array) -> jnp.ndarray:
        sample = dist.sample(seed=key)
        if self.stoch_type == "categorical":
            # straight-through
            probs = dist.distribution.probs
            sample = sample + probs - jax.lax.stop_gradient(probs)
        return sample

    def _core(
        self, prev_state: dict[str, jnp.ndarray], action: jnp.ndarray
    ) -> jnp.ndarray:
        """deter を1歩更新する。"""
        action = action.astype(jnp.float32).reshape(-1)
        stoch_flat = self._flatten_stoch(prev_state["stoch"])
        x = jnp.concatenate([action, stoch_flat], axis=-1)
        x = jax.nn.elu(self.rnn_in(x))
        return self.gru(x, prev_state["deter"])

    def prior(
        self,
        prev_state: dict[str, jnp.ndarray],
        action: jnp.ndarray,
        key: jax.Array,
    ) -> tuple[dict[str, jnp.ndarray], distrax.Distribution]:
        deter = self._core(prev_state, action)
        stats = self.prior_net(deter)
        dist = self._stats_to_dist(stats)
        stoch = self._sample(dist, key)
        state = {"deter": deter, "stoch": stoch}
        return state, dist

    def posterior(
        self,
        prev_state: dict[str, jnp.ndarray],
        action: jnp.ndarray,
        obs_embed: jnp.ndarray,
        key: jax.Array,
    ) -> tuple[dict[str, jnp.ndarray], distrax.Distribution, distrax.Distribution]:
        """prior を踏み、観測埋め込みで事後を出す。"""
        prior_state, prior_dist = self.prior(prev_state, action, key)
        # prior のサンプルは事後ネットの入力にも使う（contrastive-aif と同じ）。
        x = jnp.concatenate(
            [
                prior_state["deter"],
                self._flatten_stoch(prior_state["stoch"]),
                obs_embed.astype(jnp.float32).reshape(-1),
            ],
            axis=-1,
        )
        stats = self.posterior_net(x)
        post_dist = self._stats_to_dist(stats)
        # 事後サンプル用に別キーが欲しいが、単一 key しか無いときは split。
        key2 = jax.random.fold_in(key, 1)
        stoch = self._sample(post_dist, key2)
        post_state = {"deter": prior_state["deter"], "stoch": stoch}
        return post_state, prior_dist, post_dist

    def _dist_stats(self, dist: distrax.Distribution) -> jnp.ndarray:
        if self.stoch_type == "gaussian":
            return jnp.concatenate([dist.loc, dist.scale_diag], axis=-1)
        return dist.distribution.logits.reshape(-1)

    def observe(
        self,
        init_state: dict[str, jnp.ndarray],
        actions: jnp.ndarray,
        obs_embeds: jnp.ndarray,
        resets: jnp.ndarray,
        key: jax.Array,
    ) -> tuple[dict[str, jnp.ndarray], dict[str, jnp.ndarray]]:
        """時系列 observe。actions/obs_embeds/resets は時間軸が先頭 (T, ...)。"""
        # bound method を scan に渡すと Equinox+JAX で unhashable になるため、
        # ネスト関数から self を閉じ込める。
        def step(carry, inputs):
            state, key = carry
            action, obs_embed, reset = inputs
            key, k_use = jax.random.split(key)
            zero_state = self.initial()
            state = jax.tree.map(
                lambda s, z: jnp.where(reset, z, s), state, zero_state
            )
            action = jnp.where(reset, jnp.zeros_like(action), action)
            post_state, prior_dist, post_dist = self.posterior(
                state, action, obs_embed, k_use
            )
            if self.stoch_type == "gaussian":
                prior_mean = prior_dist.mode()
                post_mean = post_dist.mode()
            else:
                prior_mean = prior_dist.distribution.logits
                post_mean = post_dist.distribution.logits
            out = {
                "deter": post_state["deter"],
                "stoch": post_state["stoch"],
                "prior_mean": prior_mean,
                "post_mean": post_mean,
                "kl": post_dist.kl_divergence(prior_dist),
                "prior_entropy": prior_dist.entropy(),
                "post_entropy": post_dist.entropy(),
                "prior_stats": self._dist_stats(prior_dist),
                "post_stats": self._dist_stats(post_dist),
            }
            return (post_state, key), out

        (final_state, _), outs = jax.lax.scan(
            step,
            (init_state, key),
            (actions, obs_embeds, resets),
        )
        return final_state, outs

    def imagine(
        self,
        init_state: dict[str, jnp.ndarray],
        actions: jnp.ndarray,
        key: jax.Array,
    ) -> tuple[dict[str, jnp.ndarray], dict[str, jnp.ndarray]]:
        """時系列 imagine。actions は (T, action_dim)。"""

        def step(carry, action):
            state, key = carry
            key, k_use = jax.random.split(key)
            state, prior_dist = self.prior(state, action, k_use)
            out = {
                "deter": state["deter"],
                "stoch": state["stoch"],
                "prior_entropy": prior_dist.entropy(),
                "prior_stats": self._dist_stats(prior_dist),
            }
            return (state, key), out

        (final_state, _), outs = jax.lax.scan(
            step,
            (init_state, key),
            actions,
        )
        return final_state, outs

    def feature(self, state: dict[str, jnp.ndarray]) -> jnp.ndarray:
        return jnp.concatenate(
            [state["deter"], self._flatten_stoch(state["stoch"])], axis=-1
        )

    def feature_dim(self) -> int:
        return self.deter_dim + self._stoch_flat_dim()

    def dist_from_stats(self, stats: jnp.ndarray) -> distrax.Distribution:
        return self._stats_to_dist(stats)
