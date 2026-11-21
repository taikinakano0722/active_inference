"""デフォルト設定。形状の数値は環境の空間が正本であり、ここには書かない。"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Config:
    # RSSM
    deter_dim: int = 200
    stoch_dim: int = 30
    stoch_type: str = "gaussian"  # "gaussian" | "categorical"
    stoch_classes: int = 32  # categorical のとき
    hidden_dim: int = 200

    # Encoder / Decoder
    vec_layers: int = 2
    vec_units: int = 128
    img_depths: tuple[int, ...] = (32, 64, 128, 256)
    img_kernel: int = 4

    # EFE
    efe_mode: str = "entropy"  # "entropy" | "kl"
    epistemic_weight: float = 1.0
    pragmatic_weight: float = 1.0

    # 行動
    action_mode: str = "policy"  # "policy" | "candidate"
    candidate_count: int = 256
    horizon: int = 5
    actor_entropy: float = 1e-4

    # 学習
    lr: float = 3e-4
    policy_lr: float = 8e-5
    grad_clip: float = 100.0
    batch_size: int = 32
    seq_len: int = 50
    free_nats: float = 1.0
    kl_balance: float = 0.8  # dyn 側の重み。rep は 1 - kl_balance
    kl_scale: float = 1.0
    gamma: float = 0.99

    # 環境ループ
    random_episodes: int = 20
    train_steps: int = 200
    wm_epochs: int = 50
    policy_epochs: int = 20
    seed: int = 0

    # 追加の任意キー
    extras: dict = field(default_factory=dict)
