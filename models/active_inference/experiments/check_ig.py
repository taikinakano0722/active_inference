"""情報利得格子で、想像観測 KL が効いているかを確認する実験入口。

--mode reward|no_reward|all
--data scripted|model|all  （報酬版のみ）
--seed N
"""

from __future__ import annotations

import argparse
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import numpy as np

from active_inference.agent import Agent, make_wm_step
from active_inference.config import Config
from active_inference.envs.info_grid import InfoGridEnv
from active_inference.experiments.eval_ig import (
    eval_no_reward_sweep,
    eval_reward_sweep,
)
from active_inference.experiments.ig_data import (
    collect_coverage,
    collect_scripted_no_reward,
    collect_scripted_reward,
)
from active_inference.replay_buffer import StreamReplayBuffer
from active_inference.train import (
    TrainRatio,
    _store_episode,
    _train_on_chunk,
    collect_episode,
)


# 収集中に固定する情報利得係数。評価の列とは別に置く。
COLLECT_EPISTEMIC_WEIGHT = 1.0
# 評価で振る係数の列。観測と KL の規模が分かったら見直す。
EVAL_WEIGHTS = [0.0, 0.1, 0.5, 1.0, 2.0]


def make_ig_config(seed: int, mode: str) -> Config:
    """格子向けの小さめ設定。ストリームバッファ、候補モード、KL。"""
    config = Config(seed=seed)
    config.deter_dim = 64
    config.stoch_dim = 16
    config.hidden_dim = 64
    config.vec_units = 32
    config.vec_layers = 2
    config.batch_size = 16
    # ストリームなので境目をまたげる長さ。報酬版の成功は最長 9 歩、打ち切り 20 歩。
    config.seq_len = 16
    config.horizon = 4
    config.candidate_count = 256
    config.action_mode = "candidate"
    config.efe_mode = "kl"
    config.epistemic_weight = COLLECT_EPISTEMIC_WEIGHT
    config.pragmatic_weight = 1.0
    config.replay_mode = "stream"
    config.stream_capacity = 100_000
    config.train_ratio = 64.0
    config.free_nats = 0.25
    config.random_initial = False
    config.n_rounds = 1
    # train() は使わず自前で集めるが、検証用に置いておく。
    config.random_episodes = 1
    config.collect_episodes = 1
    config.extras = {"ig_mode": mode}
    return config


def _new_buffer(config: Config) -> StreamReplayBuffer:
    return StreamReplayBuffer(capacity=config.stream_capacity)


def _init_wm_train(
    env: InfoGridEnv,
    config: Config,
    key: jax.Array,
    agent: Agent | None = None,
) -> tuple[Agent, Any, Any, TrainRatio, jax.Array, np.random.Generator, int]:
    """世界モデル更新の部品を用意する。agent が無ければ新規作成。"""
    key, k_agent = jax.random.split(key)
    if agent is None:
        agent = Agent.create(env.obs_space, env.act_space, config, k_agent)
    wm_opt, wm_step = make_wm_step(config)
    params = eqx.filter(agent, eqx.is_inexact_array)
    wm_opt_state = wm_opt.init(params)
    credit = TrainRatio(config.train_ratio, config.batch_size * config.seq_len)
    rng = np.random.default_rng(config.seed)
    return agent, wm_step, wm_opt_state, credit, key, rng, 0


def train_on_episodes(
    agent: Agent,
    buffer: StreamReplayBuffer,
    episodes: list[dict[str, Any]],
    config: Config,
    wm_step,
    wm_opt_state,
    credit: TrainRatio,
    key: jax.Array,
    rng: np.random.Generator,
    done_updates: int,
) -> tuple[Agent, Any, jax.Array, int, int]:
    """エピソードをバッファへ入れ、train_ratio に従って世界モデルを更新する。"""
    total_steps = 0
    for ep in episodes:
        taken = len(ep["rew"])
        if taken <= 0:
            raise RuntimeError("空のエピソードがあります。")
        _store_episode(buffer, ep, config.replay_mode)
        total_steps += taken
        agent, wm_opt_state, _, key, done_updates = _train_on_chunk(
            agent,
            buffer,
            config,
            taken,
            credit,
            wm_step,
            wm_opt_state,
            None,
            None,
            key,
            rng,
            done_updates,
        )
    return agent, wm_opt_state, key, done_updates, total_steps


def collect_model_episodes(
    env: InfoGridEnv,
    agent: Agent,
    config: Config,
    n_episodes: int,
    key: jax.Array,
) -> tuple[list[dict[str, Any]], jax.Array]:
    """候補モードでモデル自身に集めさせる。"""
    cfg = deepcopy(config)
    cfg.action_mode = "candidate"
    cfg.efe_mode = "kl"
    cfg.epistemic_weight = COLLECT_EPISTEMIC_WEIGHT
    episodes: list[dict[str, Any]] = []
    for _ in range(n_episodes):
        key, k_ep = jax.random.split(key)
        ep, _, key = collect_episode(env, agent, cfg, k_ep, random=False)
        episodes.append(ep)
    return episodes, key


def _save_result(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"saved: {path}")


def _results_dir() -> Path:
    # models/active_inference/results に置く。
    return Path(__file__).resolve().parents[1] / "results"


def run_no_reward(seed: int, n_scripted: int = 100, n_eval: int = 50) -> dict[str, Any]:
    print("=== mode no_reward ===")
    env = InfoGridEnv(mode="no_reward", seed=seed)
    config = make_ig_config(seed, "no_reward")
    key = jax.random.PRNGKey(seed)
    rng = np.random.default_rng(seed)

    print("--- coverage ---")
    coverage = collect_coverage(env, rng, n_min=max(20, n_scripted // 2))
    print(f"  coverage episodes: {len(coverage)}")

    buffer = _new_buffer(config)
    agent, wm_step, wm_opt_state, credit, key, train_rng, done_updates = _init_wm_train(
        env, config, key
    )
    agent, wm_opt_state, key, done_updates, _ = train_on_episodes(
        agent,
        buffer,
        coverage,
        config,
        wm_step,
        wm_opt_state,
        credit,
        key,
        train_rng,
        done_updates,
    )
    print(f"  updates after coverage: {done_updates}")

    print("--- scripted K/C ---")
    scripted = collect_scripted_no_reward(env, n_scripted, rng)
    agent, wm_opt_state, key, done_updates, _ = train_on_episodes(
        agent,
        buffer,
        scripted,
        config,
        wm_step,
        wm_opt_state,
        credit,
        key,
        train_rng,
        done_updates,
    )
    print(f"  updates after scripted: {done_updates}")

    print("--- eval sweep ---")
    sweep = eval_no_reward_sweep(
        agent, env, config, EVAL_WEIGHTS, n_episodes=n_eval, seed=seed + 5000
    )
    payload = {
        "mode": "no_reward",
        "data_source": "scripted",
        "seed": seed,
        "coverage_episodes": len(coverage),
        "scripted_episodes": len(scripted),
        "train_updates": done_updates,
        "collect_epistemic_weight": None,
        "sweep": sweep,
    }
    _save_result(_results_dir() / f"ig_no_reward_scripted_{seed}.json", payload)
    return payload


def run_reward(
    seed: int,
    data: str,
    n_scripted: int = 100,
    n_model: int = 100,
    n_eval: int = 50,
) -> list[dict[str, Any]]:
    print("=== mode reward ===")
    env = InfoGridEnv(mode="reward", seed=seed)
    config = make_ig_config(seed, "reward")
    key = jax.random.PRNGKey(seed)
    rng = np.random.default_rng(seed)

    print("--- coverage ---")
    coverage = collect_coverage(env, rng, n_min=max(20, n_scripted // 2))
    print(f"  coverage episodes: {len(coverage)}")

    buffer0 = _new_buffer(config)
    agent0, wm_step0, wm_opt0, credit0, key, train_rng, done0 = _init_wm_train(
        env, config, key
    )
    agent0, wm_opt0, key, done0, _ = train_on_episodes(
        agent0, buffer0, coverage, config, wm_step0, wm_opt0, credit0, key, train_rng, done0
    )
    print(f"  updates after coverage: {done0}")

    payloads: list[dict[str, Any]] = []
    do_scripted = data in ("scripted", "all")
    do_model = data in ("model", "all")

    if do_scripted:
        print("--- scripted reward paths ---")
        # 被覆済み重みから分岐。バッファも被覆分を引き継ぐ。
        agent = deepcopy(agent0)
        buffer = deepcopy(buffer0)
        wm_opt, wm_step = make_wm_step(config)
        params = eqx.filter(agent, eqx.is_inexact_array)
        wm_opt_state = wm_opt.init(params)
        credit = TrainRatio(config.train_ratio, config.batch_size * config.seq_len)
        # 被覆で使った分は比の持ち越しを捨て、追加データ分だけ更新する。
        done_updates = done0
        scripted = collect_scripted_reward(env, n_scripted, rng)
        agent, wm_opt_state, key, done_updates, _ = train_on_episodes(
            agent,
            buffer,
            scripted,
            config,
            wm_step,
            wm_opt_state,
            credit,
            key,
            train_rng,
            done_updates,
        )
        print(f"  updates after scripted: {done_updates}")
        print("--- eval sweep (scripted) ---")
        sweep = eval_reward_sweep(
            agent, env, config, EVAL_WEIGHTS, n_episodes=n_eval, seed=seed + 6000
        )
        payload = {
            "mode": "reward",
            "data_source": "scripted",
            "seed": seed,
            "coverage_episodes": len(coverage),
            "scripted_episodes": len(scripted),
            "train_updates": done_updates,
            "collect_epistemic_weight": None,
            "sweep": sweep,
        }
        _save_result(_results_dir() / f"ig_reward_scripted_{seed}.json", payload)
        payloads.append(payload)

    if do_model:
        print("--- model collect (candidate, fixed epistemic) ---")
        agent = deepcopy(agent0)
        buffer = deepcopy(buffer0)
        wm_opt, wm_step = make_wm_step(config)
        params = eqx.filter(agent, eqx.is_inexact_array)
        wm_opt_state = wm_opt.init(params)
        credit = TrainRatio(config.train_ratio, config.batch_size * config.seq_len)
        done_updates = done0
        model_eps, key = collect_model_episodes(env, agent, config, n_model, key)
        agent, wm_opt_state, key, done_updates, _ = train_on_episodes(
            agent,
            buffer,
            model_eps,
            config,
            wm_step,
            wm_opt_state,
            credit,
            key,
            train_rng,
            done_updates,
        )
        print(f"  updates after model collect: {done_updates}")
        print("--- eval sweep (model) ---")
        sweep = eval_reward_sweep(
            agent, env, config, EVAL_WEIGHTS, n_episodes=n_eval, seed=seed + 7000
        )
        payload = {
            "mode": "reward",
            "data_source": "model",
            "seed": seed,
            "coverage_episodes": len(coverage),
            "model_episodes": len(model_eps),
            "train_updates": done_updates,
            "collect_epistemic_weight": COLLECT_EPISTEMIC_WEIGHT,
            "sweep": sweep,
        }
        _save_result(_results_dir() / f"ig_reward_model_{seed}.json", payload)
        payloads.append(payload)

    return payloads


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("reward", "no_reward", "all"),
        default="all",
    )
    parser.add_argument(
        "--data",
        choices=("scripted", "model", "all"),
        default="all",
        help="報酬版のデータ源。no_reward では無視する。",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-scripted", type=int, default=100)
    parser.add_argument("--n-model", type=int, default=100)
    parser.add_argument("--n-eval", type=int, default=50)
    args = parser.parse_args()

    # 収集中の候補列は JIT が重いので、ログで分かるようにする。
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    if args.mode in ("no_reward", "all"):
        run_no_reward(args.seed, n_scripted=args.n_scripted, n_eval=args.n_eval)
    if args.mode in ("reward", "all"):
        run_reward(
            args.seed,
            args.data,
            n_scripted=args.n_scripted,
            n_model=args.n_model,
            n_eval=args.n_eval,
        )


if __name__ == "__main__":
    main()
