"""情報利得格子の評価。候補モードで毎ステップ計画し直す。"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from active_inference.agent import Agent, step_latent_and_action
from active_inference.config import Config
from active_inference.envs.info_grid import InfoGridEnv
from active_inference.spaces import action_dim, flatten_action


def _run_candidate_episode(
    agent: Agent,
    env: InfoGridEnv,
    config: Config,
    key: jax.Array,
) -> tuple[float, str | None, jax.Array]:
    """候補モードで1エピソード。累積報酬と first_visit を返す。"""
    key, k_reset, k_post, k_act = jax.random.split(key, 4)
    obs, info = env.reset(seed=int(jax.random.randint(k_reset, (), 0, 2**31 - 1)))
    state = None
    prev_action = jnp.zeros((action_dim(env.act_space),), dtype=jnp.float32)
    is_first = jnp.bool_(True)
    total = 0.0
    first_visit = info.get("first_visit")
    done = False

    while not done:
        key, k_post, k_act = jax.random.split(key, 3)
        state, action_jax = step_latent_and_action(
            agent,
            state if state is not None else agent.rssm.initial(),
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
        obs, reward, done, step_info = env.step(action)
        total += float(reward)
        if step_info.get("first_visit") is not None:
            first_visit = step_info["first_visit"]
        prev_action = flatten_action(action_jax, env.act_space)
        is_first = jnp.bool_(False)

    return total, first_visit, key


def eval_no_reward(
    agent: Agent,
    env: InfoGridEnv,
    config: Config,
    n_episodes: int = 50,
    seed: int = 0,
) -> dict[str, Any]:
    """報酬なし版。K / C / neither の件数と割合を返す。"""
    if env.mode != "no_reward":
        raise ValueError("eval_no_reward は mode='no_reward' 用です。")
    cfg = deepcopy(config)
    cfg.action_mode = "candidate"
    cfg.efe_mode = "kl"
    key = jax.random.PRNGKey(seed)
    counts = {"K": 0, "C": 0, "neither": 0}

    for _ in range(n_episodes):
        key, k_ep = jax.random.split(key)
        _, first_visit, key = _run_candidate_episode(agent, env, cfg, k_ep)
        if first_visit == "K":
            counts["K"] += 1
        elif first_visit == "C":
            counts["C"] += 1
        else:
            counts["neither"] += 1

    n = float(n_episodes)
    return {
        "n_episodes": n_episodes,
        "count_K": counts["K"],
        "count_C": counts["C"],
        "count_neither": counts["neither"],
        "rate_K": counts["K"] / n,
        "rate_C": counts["C"] / n,
        "rate_neither": counts["neither"] / n,
        "epistemic_weight": float(cfg.epistemic_weight),
    }


def eval_reward(
    agent: Agent,
    env: InfoGridEnv,
    config: Config,
    n_episodes: int = 50,
    seed: int = 0,
) -> dict[str, Any]:
    """報酬版。平均累積報酬を返す。"""
    if env.mode != "reward":
        raise ValueError("eval_reward は mode='reward' 用です。")
    cfg = deepcopy(config)
    cfg.action_mode = "candidate"
    cfg.efe_mode = "kl"
    key = jax.random.PRNGKey(seed)
    returns: list[float] = []

    for _ in range(n_episodes):
        key, k_ep = jax.random.split(key)
        total, _, key = _run_candidate_episode(agent, env, cfg, k_ep)
        returns.append(total)

    return {
        "n_episodes": n_episodes,
        "mean_cumulative_reward": float(np.mean(returns)),
        "std_cumulative_reward": float(np.std(returns)),
        "epistemic_weight": float(cfg.epistemic_weight),
    }


def eval_reward_sweep(
    agent: Agent,
    env: InfoGridEnv,
    base_config: Config,
    weights: list[float],
    n_episodes: int = 50,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """epistemic_weight を列で振り、各値で eval_reward を呼ぶ。"""
    results: list[dict[str, Any]] = []
    for i, w in enumerate(weights):
        cfg = deepcopy(base_config)
        cfg.epistemic_weight = float(w)
        cfg.pragmatic_weight = 1.0
        cfg.action_mode = "candidate"
        cfg.efe_mode = "kl"
        metrics = eval_reward(
            agent, env, cfg, n_episodes=n_episodes, seed=seed + 1000 * (i + 1)
        )
        results.append(metrics)
        print(
            f"  epistemic_weight={w}: "
            f"mean_return={metrics['mean_cumulative_reward']:.4f}"
        )
    return results


def eval_no_reward_sweep(
    agent: Agent,
    env: InfoGridEnv,
    base_config: Config,
    weights: list[float],
    n_episodes: int = 50,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """epistemic_weight を列で振り、各値で eval_no_reward を呼ぶ。"""
    results: list[dict[str, Any]] = []
    for i, w in enumerate(weights):
        cfg = deepcopy(base_config)
        cfg.epistemic_weight = float(w)
        cfg.pragmatic_weight = 1.0
        cfg.action_mode = "candidate"
        cfg.efe_mode = "kl"
        metrics = eval_no_reward(
            agent, env, cfg, n_episodes=n_episodes, seed=seed + 1000 * (i + 1)
        )
        results.append(metrics)
        print(
            f"  epistemic_weight={w}: "
            f"K={metrics['rate_K']:.2f} C={metrics['rate_C']:.2f} "
            f"neither={metrics['rate_neither']:.2f}"
        )
    return results


def print_metrics(title: str, metrics: dict[str, Any]) -> None:
    print(title)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.6f}")
        else:
            print(f"  {k}: {v}")
