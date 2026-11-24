"""Point1D の実験。次元、ホライズン、離散か連続かはここで決め、train() に渡す。"""

from __future__ import annotations

import argparse

from active_inference.config import Config
from active_inference.envs.point1d import Point1DEnv
from active_inference.train import train


def make_point1d(seed: int, discrete: bool) -> tuple[Point1DEnv, Config]:
    """1次元環境と、その実験で使う設定を返す。"""
    config = Config(seed=seed)
    # 第一段は小さめのネットで十分。
    # これより下は、1次元環境向けに Config の既定値を上書きしている。
    # deter は GRU が持つ記憶、stoch は各時刻サンプルする確率的な潜在の次元。
    config.deter_dim = 64
    config.stoch_dim = 16
    config.hidden_dim = 64
    config.vec_units = 32
    config.vec_layers = 2
    config.batch_size = 16
    # 世界モデルが1回の更新で見る系列の長さ。エピソード長（max_steps=40）より短い。
    config.seq_len = 20
    # 行動選択で想像する未来の長さ。候補モードも方策モードもこの値を共有する。
    config.horizon = 4
    config.candidate_count = 81  # 3^4 = 81 で離散全列挙
    # 1周目は空のバッファを埋めるため、エピソードを多めに集める。
    # 0 にすると random_steps のステップ数で集める。
    config.random_episodes = 30
    # 2周目以降は学習したモデルで、1周目より少なく集める。
    config.collect_episodes = 10
    config.n_rounds = 2
    config.buffer_capacity = 500
    # 新しい環境ステップ 1 に対し、リプレイから 32 ステップ分を学習に使う。
    # 更新 1 回が使う量は batch_size * seq_len。
    config.train_ratio = 32.0
    # 1周目も select_action。一様サンプルで埋めるときは True。
    config.random_initial = False
    # 情報利得は、想像した stoch の事前エントロピー。観測をデコードし直す KL は使わない。
    config.efe_mode = "entropy"
    # 実用項（予測報酬の符号反転）を主にし、情報利得の寄与は小さくしておく。
    config.epistemic_weight = 0.01
    # 収集は最初から方策。候補列で集めるときは "candidate" にする。
    config.action_mode = "policy"

    # 数直線上の位置を速度で動かす環境。1ステップの移動は max_speed までなので、
    # 目標へは複数ステップかかる。エピソードは 40 ステップで打ち切る。
    env = Point1DEnv(discrete=discrete, max_speed=0.1, max_steps=40, seed=seed)
    return env, config


def main() -> None:
    """コマンドラインから Point1D の実験を起動する。

    既定は離散の3値行動。--continuous を付けると、行動が速度の連続値になる。
    収集と更新の繰り返しは train() が行い、この環境の数値はここに置く。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--continuous", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    env, config = make_point1d(seed=args.seed, discrete=not args.continuous)
    train(env, config)


if __name__ == "__main__":
    main()
