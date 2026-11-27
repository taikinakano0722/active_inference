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
    # 方策ヘッド。DreamerV3 の policy に合わせ、最終層を小さくして最初はほぼ一様にする。
    # unimix は学習中も残る一様分布の混合。連続の標準偏差は正規化空間 [-1, 1] での幅。
    policy_outscale: float = 0.01
    policy_unimix: float = 0.01
    policy_min_std: float = 0.1
    policy_max_std: float = 1.0

    # 学習
    lr: float = 3e-4
    policy_lr: float = 8e-5
    grad_clip: float = 100.0
    batch_size: int = 32
    seq_len: int = 50
    free_nats: float = 1.0
    # dyn は事前側、rep は事後側への KL の係数。DreamerV3 の loss_scales に合わせる。
    dyn_scale: float = 1.0
    rep_scale: float = 0.1
    kl_scale: float = 1.0
    # ※ 現在は EFE の和に掛かっていない。将来割引を入れるときの予約。
    gamma: float = 0.99

    # 環境ループ
    # 収集量は周ごとに別に指定する。1周目は空のバッファを埋めるため、2周目以降より多くすることが多い。
    # episodes が 1 以上のときは、その本数だけエピソードを終端まで集める。
    # 0 のときだけ steps を使い、合計ステップ数で収集量を指定する。
    # その場合、done が来ればそこでチャンクを区切り、残りのステップは次のエピソードで集める。
    # random_initial が真のときだけ、1周目の行動は環境の一様サンプル。
    # 偽なら、1周目から select_action（方策モードなら方策、候補モードなら候補列）。
    random_initial: bool = False
    random_episodes: int = 20
    random_steps: int = 0
    # 2周目以降の収集量。行動は select_action。
    collect_episodes: int = 5
    collect_steps: int = 0
    # 収集と更新を繰り返す周回数。更新そのものは各チャンクのあと、train_ratio に従って行う。
    n_rounds: int = 1
    # replay_mode が "episode" のとき、残す記録の本数。1本は環境をやり直してから終わるまでの系列。
    buffer_capacity: int = 1000
    # "episode": 1本ずつ保存し、満杯なら古い1本を捨てる。1本の途中に終わりはない。
    # "stream": ステップを1本の列につなぎ、stream_capacity を超えたら古いステップから捨てる。
    replay_mode: str = "episode"
    # replay_mode が "stream" のとき、列に残すステップ数の上限。
    stream_capacity: int = 100_000
    # 新しい環境ステップ 1 に対し、リプレイから何ステップ分を学習に使うか。
    # 更新 1 回が使う量は batch_size * seq_len。端数は次のチャンクへ持ち越す。
    # 系列を切れるようになる前のステップは、この比に含めない。
    train_ratio: float = 32.0
    seed: int = 0

    # 追加の任意キー
    extras: dict = field(default_factory=dict)
