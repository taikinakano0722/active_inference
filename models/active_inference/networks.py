"""Equinox の基本ネットワーク。単一サンプル用。"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp


class MLP(eqx.Module):
    """全結合ネットワーク。最終層の活性化は付けない。"""

    layers: tuple[eqx.nn.Linear, ...]
    activation: Callable[[jnp.ndarray], jnp.ndarray] = eqx.field(static=True)

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int,
        layers: int,
        key: jax.Array,
        activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.elu,
    ):
        if layers < 1:
            raise ValueError("layers は 1 以上である必要があります。")
        keys = jax.random.split(key, layers)
        mods: list[eqx.nn.Linear] = []
        if layers == 1:
            mods.append(eqx.nn.Linear(in_dim, out_dim, key=keys[0]))
        else:
            mods.append(eqx.nn.Linear(in_dim, hidden_dim, key=keys[0]))
            for i in range(1, layers - 1):
                mods.append(eqx.nn.Linear(hidden_dim, hidden_dim, key=keys[i]))
            mods.append(eqx.nn.Linear(hidden_dim, out_dim, key=keys[-1]))
        self.layers = tuple(mods)
        self.activation = activation

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.layers[-1](x)


class GRUCell(eqx.Module):
    """単一ステップの GRU。入力は (input_dim,)、隠れは (hidden_dim,)。"""

    cell: eqx.nn.GRUCell

    def __init__(self, input_dim: int, hidden_dim: int, key: jax.Array):
        self.cell = eqx.nn.GRUCell(input_dim, hidden_dim, key=key)

    @property
    def hidden_size(self) -> int:
        return self.cell.hidden_size

    def __call__(self, x: jnp.ndarray, h: jnp.ndarray) -> jnp.ndarray:
        return self.cell(x, h)


class Conv2DEncoder(eqx.Module):
    """画像 (H, W, C) → 特徴ベクトル。第二段で実装する。"""

    depths: tuple[int, ...] = eqx.field(static=True)
    kernel: int = eqx.field(static=True)
    layers: tuple[eqx.nn.Conv2d, ...]
    out_dim: int = eqx.field(static=True)

    def __init__(
        self,
        in_channels: int,
        depths: Sequence[int] = (32, 64, 128, 256),
        kernel: int = 4,
        key: jax.Array | None = None,
    ):
        if key is None:
            key = jax.random.PRNGKey(0)
        depths = tuple(int(d) for d in depths)
        keys = jax.random.split(key, len(depths))
        layers: list[eqx.nn.Conv2d] = []
        c_in = in_channels
        for i, depth in enumerate(depths):
            layers.append(
                eqx.nn.Conv2d(
                    c_in,
                    depth,
                    kernel_size=kernel,
                    stride=2,
                    key=keys[i],
                )
            )
            c_in = depth
        self.depths = depths
        self.kernel = int(kernel)
        self.layers = tuple(layers)
        # 64x64 入力を仮定したときの概算。実際の out_dim は forward で flatten 後の長さ。
        self.out_dim = depths[-1]

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: (H, W, C) → (C, H, W)
        if x.ndim != 3:
            raise ValueError(f"画像は (H, W, C) である必要があります。got {x.shape}")
        y = jnp.transpose(x.astype(jnp.float32) / 255.0 - 0.5, (2, 0, 1))
        for layer in self.layers:
            y = jax.nn.elu(layer(y))
        return y.reshape(-1)


class Conv2DDecoder(eqx.Module):
    """特徴ベクトル → 画像 (H, W, C)。第二段で使う。"""

    depths: tuple[int, ...] = eqx.field(static=True)
    kernel: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)
    img_hw: tuple[int, int] = eqx.field(static=True)
    linear: eqx.nn.Linear
    layers: tuple[eqx.nn.ConvTranspose2d, ...]
    start_hw: tuple[int, int] = eqx.field(static=True)
    start_c: int = eqx.field(static=True)

    def __init__(
        self,
        in_dim: int,
        out_channels: int,
        img_hw: tuple[int, int] = (64, 64),
        depths: Sequence[int] = (32, 64, 128, 256),
        kernel: int = 4,
        key: jax.Array | None = None,
    ):
        if key is None:
            key = jax.random.PRNGKey(0)
        depths = tuple(int(d) for d in depths)
        # 各 stride=2 で解像度が半分になる前提で、開始マップサイズを決める。
        factor = 2 ** len(depths)
        start_h = img_hw[0] // factor
        start_w = img_hw[1] // factor
        if start_h < 1 or start_w < 1:
            raise ValueError(f"画像サイズ {img_hw} に対して depths が深すぎます。")
        start_c = depths[-1]
        k_lin, *k_rest = jax.random.split(key, len(depths) + 1)
        self.linear = eqx.nn.Linear(in_dim, start_c * start_h * start_w, key=k_lin)
        # depths を逆順に戻す。最後は out_channels。
        channels = list(reversed(depths[:-1])) + [out_channels]
        c_in = start_c
        layers: list[eqx.nn.ConvTranspose2d] = []
        for i, c_out in enumerate(channels):
            layers.append(
                eqx.nn.ConvTranspose2d(
                    c_in,
                    c_out,
                    kernel_size=kernel,
                    stride=2,
                    key=k_rest[i],
                )
            )
            c_in = c_out
        self.depths = depths
        self.kernel = int(kernel)
        self.out_channels = int(out_channels)
        self.img_hw = (int(img_hw[0]), int(img_hw[1]))
        self.layers = tuple(layers)
        self.start_hw = (start_h, start_w)
        self.start_c = start_c

    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        y = self.linear(z)
        y = y.reshape(self.start_c, self.start_hw[0], self.start_hw[1])
        for i, layer in enumerate(self.layers):
            y = layer(y)
            if i < len(self.layers) - 1:
                y = jax.nn.elu(y)
            else:
                y = jax.nn.sigmoid(y)
        # (C, H, W) → (H, W, C)
        y = jnp.transpose(y, (1, 2, 0))
        # 端数でサイズがずれる場合があるので、目標サイズへ切り詰める/パディングする。
        h, w = self.img_hw
        y = y[:h, :w, :]
        if y.shape[0] < h or y.shape[1] < w:
            y = jnp.pad(
                y,
                ((0, h - y.shape[0]), (0, w - y.shape[1]), (0, 0)),
            )
        return y
