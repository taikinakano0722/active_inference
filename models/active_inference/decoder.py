"""観測デコーダと報酬デコーダ。"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import distrax

from .networks import Conv2DDecoder, MLP
from .spaces import Space


class ObsDecoder(eqx.Module):
    """潜在特徴 → 観測キーごとの分布。単一サンプル。"""

    vec_keys: tuple[str, ...] = eqx.field(static=True)
    img_keys: tuple[str, ...] = eqx.field(static=True)
    vec_shapes: dict[str, tuple[int, ...]] = eqx.field(static=True)
    img_shapes: dict[str, tuple[int, ...]] = eqx.field(static=True)
    vec_heads: dict[str, MLP]
    img_heads: dict[str, Conv2DDecoder]

    def __init__(
        self,
        obs_space: dict[str, Space],
        feat_dim: int,
        vec_units: int = 128,
        vec_layers: int = 2,
        img_depths: tuple[int, ...] = (32, 64, 128, 256),
        img_kernel: int = 4,
        key: jax.Array | None = None,
    ):
        if key is None:
            key = jax.random.PRNGKey(0)
        exclude = {"reward", "is_first", "is_last", "is_terminal"}
        keys = sorted(k for k in obs_space if k not in exclude)
        vec_keys = [k for k in keys if obs_space[k].is_vector]
        img_keys = [k for k in keys if obs_space[k].is_image]

        vec_heads: dict[str, MLP] = {}
        img_heads: dict[str, Conv2DDecoder] = {}
        vec_shapes: dict[str, tuple[int, ...]] = {}
        img_shapes: dict[str, tuple[int, ...]] = {}
        keys_list = jax.random.split(key, max(len(vec_keys) + len(img_keys), 1))
        ki = 0
        for name in vec_keys:
            shape = tuple(int(x) for x in obs_space[name].shape)
            out_dim = int(jnp.prod(jnp.array(shape)))
            vec_heads[name] = MLP(
                feat_dim, out_dim, vec_units, vec_layers, key=keys_list[ki]
            )
            vec_shapes[name] = shape
            ki += 1
        for name in img_keys:
            h, w, c = obs_space[name].shape
            img_heads[name] = Conv2DDecoder(
                in_dim=feat_dim,
                out_channels=c,
                img_hw=(h, w),
                depths=img_depths,
                kernel=img_kernel,
                key=keys_list[ki],
            )
            img_shapes[name] = (h, w, c)
            ki += 1

        self.vec_keys = tuple(vec_keys)
        self.img_keys = tuple(img_keys)
        self.vec_shapes = vec_shapes
        self.img_shapes = img_shapes
        self.vec_heads = vec_heads
        self.img_heads = img_heads

    def __call__(self, feat: jnp.ndarray) -> dict[str, distrax.Distribution]:
        dists: dict[str, distrax.Distribution] = {}
        for name in self.vec_keys:
            mean = self.vec_heads[name](feat).reshape(self.vec_shapes[name])
            flat = mean.reshape(-1)
            dists[name] = distrax.MultivariateNormalDiag(
                loc=flat, scale_diag=jnp.ones_like(flat)
            )
        for name in self.img_keys:
            mean = self.img_heads[name](feat)  # (H, W, C) in [0, 1]
            flat = mean.reshape(-1)
            dists[name] = distrax.MultivariateNormalDiag(
                loc=flat, scale_diag=jnp.ones_like(flat)
            )
        return dists

    def log_prob(self, feat: jnp.ndarray, obs: dict[str, jnp.ndarray]) -> jnp.ndarray:
        dists = self(feat)
        logp = jnp.float32(0.0)
        for name, dist in dists.items():
            if name in self.img_keys:
                target = obs[name].astype(jnp.float32) / 255.0
            else:
                target = obs[name].astype(jnp.float32)
            logp = logp + dist.log_prob(target.reshape(-1))
        return logp


class RewardDecoder(eqx.Module):
    """潜在特徴 → 報酬分布。"""

    mlp: MLP

    def __init__(
        self,
        feat_dim: int,
        hidden_dim: int = 200,
        layers: int = 2,
        key: jax.Array | None = None,
    ):
        if key is None:
            key = jax.random.PRNGKey(0)
        self.mlp = MLP(feat_dim, 1, hidden_dim, layers, key=key)

    def __call__(self, feat: jnp.ndarray) -> distrax.Normal:
        mean = self.mlp(feat).reshape(())
        return distrax.Normal(loc=mean, scale=1.0)
