"""Point1D で能動的推論の部品を段階確認する実験入口。

--stage world     ランダムデータで世界モデルを学習し、1歩先予測を測る
--stage candidate 世界モデル学習のあと、候補列4歩の距離変化を測る
--stage policy    世界モデルのあと方策を追加学習し、方策と候補を比較する
--stage all       上を順に実行する（世界モデル学習は1回だけ）
"""

from __future__ import annotations

import argparse
from copy import deepcopy

from active_inference.agent import Agent
from active_inference.config import Config
from active_inference.envs.point1d import Point1DEnv
from active_inference.experiments.eval_point1d import (
    eval_candidate,
    eval_policy,
    eval_world_model,
    print_metrics,
)
from active_inference.experiments.point1d import make_point1d
from active_inference.train import train


def _world_config(base: Config) -> Config:
    """世界モデル段階用。候補モードで WM だけ更新し、1周目は一様サンプル。"""
    config = deepcopy(base)
    config.random_initial = True
    config.action_mode = "candidate"
    config.n_rounds = 1
    config.epistemic_weight = 0.0
    config.efe_mode = "entropy"
    config.horizon = 4
    config.candidate_count = 81
    # 1歩先予測がベースラインを下回るまで、収集と更新を厚めにする。
    config.random_episodes = 100
    config.train_ratio = 64.0
    config.free_nats = 0.25
    return config


def _policy_config(base: Config) -> Config:
    """方策追加学習用。渡した agent の重みから続ける。"""
    config = deepcopy(base)
    config.random_initial = False
    config.action_mode = "policy"
    config.efe_mode = "entropy"
    config.epistemic_weight = 0.0
    config.n_rounds = 6
    config.random_episodes = 20
    config.collect_episodes = 20
    config.train_ratio = 64.0
    config.free_nats = 0.25
    config.horizon = 4
    config.candidate_count = 81
    return config


def _eval_config(base: Config) -> Config:
    """評価時の EFE 設定。実用項だけを見る。"""
    config = deepcopy(base)
    config.epistemic_weight = 0.0
    config.efe_mode = "entropy"
    config.horizon = 4
    config.candidate_count = 81
    return config


def train_world_model(seed: int) -> tuple[Agent, Point1DEnv, Config]:
    env, base = make_point1d(seed=seed, discrete=True)
    config = _world_config(base)
    print("=== stage world: 世界モデル学習（ランダム収集 / 候補モード更新）===")
    agent = train(env, config)
    return agent, env, config


def run_world(agent: Agent, env: Point1DEnv, config: Config, seed: int) -> dict:
    metrics = eval_world_model(
        agent, env, _eval_config(config), n_episodes=20, seed=seed + 1000
    )
    print_metrics("=== eval world model ===", metrics)
    ok = metrics["position_mae"] < metrics["baseline_position_mae"]
    print(
        f"  check position_mae < baseline: "
        f"{'PASS' if ok else 'FAIL'} "
        f"({metrics['position_mae']:.6f} < {metrics['baseline_position_mae']:.6f})"
    )
    return metrics


def run_candidate(agent: Agent, env: Point1DEnv, config: Config, seed: int) -> dict:
    metrics = eval_candidate(
        agent, env, _eval_config(config), n_episodes=30, seed=seed + 2000
    )
    print_metrics("=== eval candidate ===", metrics)
    ok = metrics["candidate_dist_reduction"] > metrics["random_dist_reduction"]
    print(
        f"  check candidate > random dist reduction: "
        f"{'PASS' if ok else 'FAIL'} "
        f"({metrics['candidate_dist_reduction']:.6f} > "
        f"{metrics['random_dist_reduction']:.6f})"
    )
    return metrics


def run_policy(
    agent: Agent, env: Point1DEnv, base: Config, seed: int
) -> tuple[Agent, dict, dict]:
    pol_cfg = _policy_config(base)
    print("=== stage policy: 方策追加学習 ===")
    agent = train(env, pol_cfg, agent=agent)
    eval_cfg = _eval_config(pol_cfg)
    pol_metrics = eval_policy(agent, env, eval_cfg, n_episodes=50, seed=seed + 3000)
    print_metrics("=== eval policy ===", pol_metrics)
    cand_metrics = eval_candidate(
        agent, env, eval_cfg, n_episodes=50, seed=seed + 4000
    )
    print_metrics("=== eval candidate (after policy training) ===", cand_metrics)
    ok_rand = pol_metrics["policy_dist_reduction"] > pol_metrics["random_dist_reduction"]
    # 方策が候補に近いこと。厳しすぎないよう、候補の半分以上を目安にする。
    ok_cand = (
        pol_metrics["policy_dist_reduction"]
        >= 0.5 * cand_metrics["candidate_dist_reduction"]
    )
    print(
        f"  check policy > random: "
        f"{'PASS' if ok_rand else 'FAIL'} "
        f"({pol_metrics['policy_dist_reduction']:.6f} > "
        f"{pol_metrics['random_dist_reduction']:.6f})"
    )
    print(
        f"  check policy >= 0.5 * candidate: "
        f"{'PASS' if ok_cand else 'FAIL'} "
        f"({pol_metrics['policy_dist_reduction']:.6f} >= "
        f"{0.5 * cand_metrics['candidate_dist_reduction']:.6f})"
    )
    return agent, pol_metrics, cand_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("world", "candidate", "policy", "all"),
        default="all",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    agent, env, world_cfg = train_world_model(args.seed)
    _, base = make_point1d(seed=args.seed, discrete=True)

    if args.stage in ("world", "all"):
        run_world(agent, env, world_cfg, args.seed)
    if args.stage in ("candidate", "all"):
        run_candidate(agent, env, world_cfg, args.seed)
    if args.stage in ("policy", "all"):
        run_policy(agent, env, base, args.seed)


if __name__ == "__main__":
    main()
