"""観測エンコーダ。キーごとに変換し、結合してトークンにする。"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from .networks import Conv2DEncoder, MLP
from .spaces import Space


class Encoder(eqx.Module):
    """単一サンプルの観測辞書 → トークンベクトル。"""

    vec_keys: tuple[str, ...] = eqx.field(static=True)
    img_keys: tuple[str, ...] = eqx.field(static=True)
    vec_mlps: dict[str, MLP]
    img_cnns: dict[str, Conv2DEncoder]
    embed_dim: int = eqx.field(static=True)

    def __init__(
        self,
        obs_space: dict[str, Space],
        vec_units: int = 128,
        vec_layers: int = 2,
        img_depths: tuple[int, ...] = (32, 64, 128, 256),
        img_kernel: int = 4,
        key: jax.Array | None = None,
    ):
        if key is None:
            key = jax.random.PRNGKey(0)
        # 報酬やフラグはエンコードしない。
        exclude = {"reward", "is_first", "is_last", "is_terminal"}
        keys = sorted(k for k in obs_space if k not in exclude)
        vec_keys = [k for k in keys if obs_space[k].is_vector]
        img_keys = [k for k in keys if obs_space[k].is_image]
        if not vec_keys and not img_keys:
            raise ValueError("エンコード対象の観測キーがありません。")

        vec_mlps: dict[str, MLP] = {}
        img_cnns: dict[str, Conv2DEncoder] = {}
        embed = 0
        keys_list = jax.random.split(key, max(len(vec_keys) + len(img_keys), 1))
        ki = 0
        for name in vec_keys:
            space = obs_space[name]
            in_dim = int(jnp.prod(jnp.array(space.shape)))
            vec_mlps[name] = MLP(
                in_dim, vec_units, vec_units, vec_layers, key=keys_list[ki]
            )
            embed += vec_units
            ki += 1
        for name in img_keys:
            space = obs_space[name]
            h, w, c = space.shape
            cnn = Conv2DEncoder(
                in_channels=c,
                depths=img_depths,
                kernel=img_kernel,
                key=keys_list[ki],
            )
            # 実際の flatten 長をダミー入力で確定する。
            dummy = jnp.zeros((h, w, c), dtype=jnp.float32)
            out = cnn(dummy)
            img_cnns[name] = cnn
            embed += int(out.shape[-1])
            ki += 1

        self.vec_keys = tuple(vec_keys)
        self.img_keys = tuple(img_keys)
        self.vec_mlps = vec_mlps
        self.img_cnns = img_cnns
        self.embed_dim = int(embed)

    def __call__(self, obs: dict[str, jnp.ndarray]) -> jnp.ndarray:
        parts: list[jnp.ndarray] = []
        for name in self.vec_keys:
            x = obs[name].astype(jnp.float32).reshape(-1)
            parts.append(self.vec_mlps[name](x))
        for name in self.img_keys:
            parts.append(self.img_cnns[name](obs[name]))
        return jnp.concatenate(parts, axis=-1)
