"""cueworld/subtasks_for_cueworld の環境（SanityEnv / CommunicationEnv）で学習・評価する入口。

使い方は README_cueworld.md（このモデルの直下）を参照。

流れ
  1. 環境を作る。--condition yoke のときは、ランダム行動で録った親の列を
     reset のたびに再生する（いまの行動と無関係な観測になる）。
  2. train() で収集と更新を n_rounds 周。
  3. 学習後の行動と、一様ランダムの行動で、評価エピソードを回し、
     環境ごとの指標を JSON に書く。

指標（評価エピソードの全ステップの平均）
  共通          action_hist（各行動の割合）、mean_return
  sanity        frac_deterministic（次の記号が決まる行動の割合）、frac_reward
  short         frac_meaningful（返事が決まる記号の割合）
  long          frac_determined（その時点の位相で返事が決まる行動の割合）
  branch        frac_determined（残りの枝がその行動に返す記号が1つの割合）、
                final_remaining（エピソード末の残り枝数の平均）
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from active_inference.agent import Agent, step_latent_and_action
from active_inference.config import Config
from active_inference.envs.cueworld_subtasks import (
    GymSubtaskEnv,
    make_communication_env,
    make_sanity_env,
)
from active_inference.spaces import action_dim, flatten_action
from active_inference.train import train

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "cueworld"


# --------------------------------------------------------------------------
# 環境
# --------------------------------------------------------------------------


def build_env(args: argparse.Namespace, seed: int) -> GymSubtaskEnv:
    if args.env == "sanity":
        return make_sanity_env(
            max_steps=args.max_steps,
            n_actions=args.n_actions,
            n_symbols=args.n_symbols,
            reward_mode=args.reward_mode,
            symbol_noise=args.symbol_noise,
            obs_noise=args.obs_noise,
            seed=seed,
        )
    return make_communication_env(
        max_steps=args.max_steps,
        n_meaningful=args.n_meaningful,
        dependency=args.dependency,
        obs_noise=args.obs_noise,
        seed=seed,
    )


def record_tapes(args: argparse.Namespace, n_tapes: int, seed: int) -> list[Any]:
    """依存ありの環境をランダム行動で回し、yoke 用の列を n_tapes 本録る。

    sanity は観測の列 (max_steps, n_symbols)、communication は親記号の列 (max_steps,)。
    """
    env = build_env(args, seed)
    rng = np.random.default_rng(seed)
    tapes = []
    for _ in range(n_tapes):
        env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        seq = []
        done = False
        while not done:
            obs, _, done, info = env.step(int(rng.integers(0, env.n_actions)))
            if args.env == "sanity":
                seq.append(obs["obs"])
            else:
                seq.append(int(info["parent_symbol"]))
        tapes.append(np.asarray(seq, dtype=np.float32 if args.env == "sanity" else np.int64))
    return tapes


class YokeEnv:
    """reset のたびに録った列を1本選び、subtasks 側の yoke オプションで再生する。"""

    def __init__(self, env: GymSubtaskEnv, tapes: list[Any], option_key: str, seed: int):
        self.env = env
        self.tapes = tapes
        self.option_key = option_key
        self.obs_space = env.obs_space
        self.act_space = env.act_space
        self.n_actions = env.n_actions
        self._rng = np.random.default_rng(seed)

    def reset(self, seed: int | None = None, options: dict | None = None):
        tape = self.tapes[int(self._rng.integers(0, len(self.tapes)))]
        opts = dict(options or {})
        opts[self.option_key] = tape
        return self.env.reset(seed=seed, options=opts)

    def step(self, action):
        return self.env.step(action)

    def sample_action(self):
        return self.env.sample_action()


def maybe_yoke(env: GymSubtaskEnv, args: argparse.Namespace, seed: int):
    if args.condition != "yoke":
        return env
    tapes = record_tapes(args, args.n_tapes, seed + 10_000)
    key = "yoke_observations" if args.env == "sanity" else "yoke_parent_symbols"
    return YokeEnv(env, tapes, key, seed)


# --------------------------------------------------------------------------
# 設定
# --------------------------------------------------------------------------


def build_config(args: argparse.Namespace) -> Config:
    """記号環境向けの小さめの設定。CLI で上書きしたものだけ変わる。"""
    c = Config(seed=args.seed)
    c.deter_dim = args.deter_dim
    c.stoch_dim = args.stoch_dim
    c.stoch_type = args.stoch_type
    c.hidden_dim = args.deter_dim
    c.vec_units = 64
    c.vec_layers = 2
    c.batch_size = args.batch_size
    c.seq_len = args.seq_len
    c.horizon = args.horizon
    c.candidate_count = args.candidate_count
    c.action_mode = args.action_mode
    c.efe_mode = args.efe_mode
    c.epistemic_weight = args.epistemic_weight
    c.pragmatic_weight = args.pragmatic_weight
    c.random_initial = args.random_initial
    c.random_episodes = args.random_episodes
    c.collect_episodes = args.collect_episodes
    c.n_rounds = args.rounds
    c.train_ratio = args.train_ratio
    c.buffer_capacity = args.buffer_capacity
    c.replay_mode = "episode"
    return c


# --------------------------------------------------------------------------
# 評価
# --------------------------------------------------------------------------


def run_eval_episode(
    env,
    agent: Agent | None,
    config: Config,
    key: jax.Array,
    rng: np.random.Generator,
) -> tuple[list[dict[str, Any]], float, jax.Array]:
    """1エピソード回し、各ステップの (行動, 行動前の info, 行動後の info) を返す。

    agent が None なら一様ランダム。学習はしない。
    """
    obs, info = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
    prev_info = info
    state = agent.rssm.initial() if agent is not None else None
    prev_action = jnp.zeros((action_dim(env.act_space),), dtype=jnp.float32)
    is_first = jnp.bool_(True)
    steps: list[dict[str, Any]] = []
    total = 0.0
    done = False
    while not done:
        key, k_post, k_act = jax.random.split(key, 3)
        if agent is None:
            action = env.sample_action()
            action_jax = {k: jnp.asarray(v) for k, v in action.items()}
        else:
            state, action_jax = step_latent_and_action(
                agent,
                state,
                prev_action,
                {k: jnp.asarray(v) for k, v in obs.items()},
                k_post,
                k_act,
                is_first,
                config.action_mode,
                config.efe_mode,
                config.horizon,
                config.candidate_count,
                config.epistemic_weight,
                config.pragmatic_weight,
            )
            action = {k: np.asarray(v) for k, v in action_jax.items()}
        obs, reward, done, info = env.step(action)
        steps.append({"action": info["action"], "before": prev_info, "after": info})
        total += reward
        prev_info = info
        prev_action = flatten_action(action_jax, env.act_space)
        is_first = jnp.bool_(False)
    return steps, total, key


def _determined(args: argparse.Namespace, step: dict[str, Any]) -> bool:
    """その行動の次の観測が、環境の規則で1つに決まっていたか。"""
    a = step["action"]
    before, after = step["before"], step["after"]
    if args.env == "sanity":
        return a == after["deterministic_action"]
    if args.dependency == "short":
        return a < args.n_meaningful
    if args.dependency == "long":
        phase = before.get("phase", "outside")
        if phase == "outside":
            return a in (0, 1)
        if phase == "mid_a":
            return a == 2
        return a == 3
    # branch
    replies = after.get("possible_replies") or ()
    return len(replies) == 1


def evaluate(env, agent, config, args, n_episodes: int, seed: int) -> dict[str, Any]:
    key = jax.random.PRNGKey(seed)
    rng = np.random.default_rng(seed)
    n_act = env.n_actions
    hist = np.zeros(n_act, dtype=np.int64)
    returns = []
    det = []
    rew_act = []
    final_remaining = []
    for _ in range(n_episodes):
        key, k_ep = jax.random.split(key)
        steps, ret, key = run_eval_episode(env, agent, config, k_ep, rng)
        returns.append(ret)
        for s in steps:
            hist[s["action"]] += 1
            det.append(_determined(args, s))
            if args.env == "sanity":
                rew_act.append(s["action"] == s["after"]["reward_action"])
        if args.env == "communication" and args.dependency == "branch":
            final_remaining.append(steps[-1]["after"]["n_remaining"])
    out: dict[str, Any] = {
        "n_episodes": n_episodes,
        "mean_return": float(np.mean(returns)),
        "action_hist": (hist / max(hist.sum(), 1)).round(4).tolist(),
        "frac_determined": float(np.mean(det)),
    }
    if args.env == "sanity":
        out["frac_reward"] = float(np.mean(rew_act))
    if final_remaining:
        out["final_remaining"] = float(np.mean(final_remaining))
    return out


# --------------------------------------------------------------------------
# 入口
# --------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group("環境")
    g.add_argument("--env", choices=["sanity", "communication"], required=True)
    g.add_argument("--dependency", choices=["short", "long", "branch"], default="short",
                   help="communication のみ")
    g.add_argument("--n-meaningful", type=int, default=1, help="communication short のみ")
    g.add_argument("--n-actions", type=int, default=3, help="sanity のみ")
    g.add_argument("--n-symbols", type=int, default=4, help="sanity のみ")
    g.add_argument("--reward-mode", action="store_true", help="sanity のみ。乱数行動の1つに報酬 1")
    g.add_argument("--symbol-noise", type=float, default=0.0, help="sanity のみ")
    g.add_argument("--obs-noise", type=float, default=0.0)
    g.add_argument("--max-steps", type=int, default=50)
    g.add_argument("--condition", choices=["dependent", "yoke"], default="dependent",
                   help="yoke: 録った親の列を行動と無関係に再生（学習も評価も）")
    g.add_argument("--n-tapes", type=int, default=50, help="yoke 用に録る列の本数")

    g = p.add_argument_group("モデル・行動")
    g.add_argument("--action-mode", choices=["policy", "candidate"], default="policy")
    g.add_argument("--efe-mode", choices=["entropy", "kl"], default="entropy",
                   help="policy モードは entropy のみ")
    g.add_argument("--epistemic-weight", type=float, default=1.0,
                   help="EFE = -pw*報酬 - ew*情報利得。負にすると予測しやすい未来を好む")
    g.add_argument("--pragmatic-weight", type=float, default=1.0)
    g.add_argument("--horizon", type=int, default=3)
    g.add_argument("--candidate-count", type=int, default=64,
                   help="離散で n_actions^horizon 以下なら全列挙")
    g.add_argument("--deter-dim", type=int, default=64)
    g.add_argument("--stoch-dim", type=int, default=16)
    g.add_argument("--stoch-type", choices=["gaussian", "categorical"], default="gaussian")

    g = p.add_argument_group("学習")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--rounds", type=int, default=3)
    g.add_argument("--random-episodes", type=int, default=30, help="1周目に集める本数")
    g.add_argument("--collect-episodes", type=int, default=10, help="2周目以降に集める本数")
    g.add_argument("--random-initial", action="store_true", help="1周目を一様ランダムで集める")
    g.add_argument("--train-ratio", type=float, default=32.0)
    g.add_argument("--batch-size", type=int, default=16)
    g.add_argument("--seq-len", type=int, default=32, help="max_steps 以下にする")
    g.add_argument("--buffer-capacity", type=int, default=500)

    g = p.add_argument_group("評価・保存")
    g.add_argument("--eval-episodes", type=int, default=20)
    g.add_argument("--tag", default="", help="出力ファイル名に付ける文字列")
    g.add_argument("--out", default=None, help="結果 JSON のパス。既定は results/cueworld/<名前>.json")
    g.add_argument("--save", default=None, help="学習後の重みを保存する .eqx のパス")
    g.add_argument("--load", default=None, help="学習せず、この重みを読み込んで評価だけする")
    args = p.parse_args(argv)

    if args.seq_len > args.max_steps:
        p.error("--seq-len は --max-steps 以下にしてください（系列を切れず学習できません）。")
    if args.action_mode == "policy" and args.efe_mode != "entropy":
        p.error("policy モードは --efe-mode entropy のみです。")
    return args


def run_name(args: argparse.Namespace) -> str:
    if args.env == "sanity":
        env = "sanity" + ("_reward" if args.reward_mode else "")
    else:
        env = f"comm_{args.dependency}"
        if args.dependency == "short":
            env += f"_m{args.n_meaningful}"
    name = f"{env}_{args.condition}_{args.action_mode}_ew{args.epistemic_weight:g}_s{args.seed}"
    return name + (f"_{args.tag}" if args.tag else "")


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    name = run_name(args)
    print(f"== {name}")
    config = build_config(args)
    env = maybe_yoke(build_env(args, args.seed), args, args.seed)

    t0 = time.time()
    if args.load:
        skeleton = Agent.create(env.obs_space, env.act_space, config, jax.random.PRNGKey(0))
        agent = eqx.tree_deserialise_leaves(args.load, skeleton)
        print(f"重みを読み込みました: {args.load}")
    else:
        agent = train(env, config)
    train_sec = time.time() - t0
    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        eqx.tree_serialise_leaves(args.save, agent)
        print(f"重みを保存しました: {args.save}")

    # 評価は学習と同じ条件の環境を、別の seed で作り直して行う。
    eval_seed = args.seed + 1000
    eval_env = maybe_yoke(build_env(args, eval_seed), args, eval_seed)
    print("評価中...")
    res_agent = evaluate(eval_env, agent, config, args, args.eval_episodes, eval_seed)
    res_random = evaluate(eval_env, None, config, args, args.eval_episodes, eval_seed)

    result = {
        "name": name,
        "args": vars(args),
        "config": {k: v for k, v in asdict(config).items() if k != "extras"},
        "train_sec": round(train_sec, 1),
        "agent": res_agent,
        "random": res_random,
    }
    print(json.dumps({"agent": res_agent, "random": res_random}, ensure_ascii=False, indent=2))

    out = Path(args.out) if args.out else RESULTS_DIR / f"{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    print(f"結果を書きました: {out}")
    return result


if __name__ == "__main__":
    main()
