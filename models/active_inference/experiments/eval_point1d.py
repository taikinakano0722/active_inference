"""Point1D 上で能動的推論モデルの部品を段階評価する。

世界モデルの1歩先予測、候補列の4歩計画、方策の4歩実行を分ける。
各関数は数値だけの辞書を返し、呼び出し側が印刷する。
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from active_inference.agent import Agent, get_posterior_state
from active_inference.config import Config
from active_inference.efe import evaluate_action_sequences
from active_inference.envs.point1d import Point1DEnv
from active_inference.policy import sample_action_sequences
from active_inference.spaces import action_dim, flatten_action


def _snapshot(env: Point1DEnv) -> tuple[float, float, int, bool]:
    """評価中に同じ開始地点へ戻すための内部状態。"""
    return env._position, env._goal, env._step, env._done


def _restore(env: Point1DEnv, snap: tuple[float, float, int, bool]) -> None:
    env._position, env._goal, env._step, env._done = snap


def _distance(obs: dict[str, np.ndarray]) -> float:
    return float(abs(float(obs["position"][0]) - float(obs["goal"][0])))


def _one_hot(idx: int, n: int = 3) -> np.ndarray:
    out = np.zeros((n,), dtype=np.float32)
    out[int(idx)] = 1.0
    return out


def _oracle_action(env: Point1DEnv, obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """真の状態から、目標へ近づく1手（届く距離なら停止）。"""
    pos = float(obs["position"][0])
    goal = float(obs["goal"][0])
    gap = goal - pos
    if abs(gap) <= env.max_speed:
        idx = 1
    elif gap > 0:
        idx = 2
    else:
        idx = 0
    if env.discrete:
        return {"action": _one_hot(idx)}
    speed = float(np.clip(gap, -env.max_speed, env.max_speed))
    return {"action": np.array([speed], dtype=np.float32)}


def _action_toward_goal(action: np.ndarray, obs: dict[str, np.ndarray]) -> bool:
    """行動が目標方向を向いているか。停止は False。"""
    gap = float(obs["goal"][0]) - float(obs["position"][0])
    if abs(gap) < 1e-8:
        return True
    if action.shape[-1] == 3:
        idx = int(np.argmax(action))
        if idx == 1:
            return False
        return (idx == 2 and gap > 0) or (idx == 0 and gap < 0)
    speed = float(action.reshape(-1)[0])
    if abs(speed) < 1e-8:
        return False
    return (speed > 0 and gap > 0) or (speed < 0 and gap < 0)


def _run_actions(
    env: Point1DEnv,
    actions: list[dict[str, np.ndarray]],
) -> tuple[float, float]:
    """与えた行動列を実行し、開始距離と終了距離を返す。"""
    obs = env._obs()
    start = _distance(obs)
    for action in actions:
        if env._done:
            break
        obs, _, _, _ = env.step(action)
    return start, _distance(obs)


def eval_world_model(
    agent: Agent,
    env: Point1DEnv,
    config: Config,
    n_episodes: int = 20,
    seed: int = 0,
) -> dict[str, float]:
    """ランダム行動の軌道で、prior 1歩先の位置・目標・報酬の絶対誤差を測る。

    事後は予測したい観測をすでに見ているので使わない。
    ベースラインは「位置が動かない」と予測したときの位置誤差。
    """
    key = jax.random.PRNGKey(seed)
    pos_errs: list[float] = []
    goal_errs: list[float] = []
    rew_errs: list[float] = []
    baseline_errs: list[float] = []

    for ep_i in range(n_episodes):
        key, k_reset = jax.random.split(key)
        obs, _ = env.reset(seed=int(jax.random.randint(k_reset, (), 0, 2**31 - 1)))
        state = agent.rssm.initial()
        prev_action = jnp.zeros((action_dim(env.act_space),), dtype=jnp.float32)
        is_first = True
        done = False
        while not done:
            key, k_post, k_prior = jax.random.split(key, 3)
            # 現在の観測までの事後。次の観測はまだ見ていない。
            state = get_posterior_state(
                agent,
                state,
                prev_action,
                {k: jnp.asarray(v) for k, v in obs.items()},
                k_post,
                is_first=is_first,
            )
            action = env.sample_action()
            action_vec = flatten_action(action, env.act_space)
            # 行動だけを入れた prior で1歩先を想像する。
            # 評価の点推定にはサンプルではなく事前分布の平均を使う。
            next_state, prior_dist = agent.rssm.prior(state, action_vec, k_prior)
            stoch_mean = prior_dist.mean()
            next_state = {
                "deter": next_state["deter"],
                "stoch": stoch_mean.reshape(next_state["stoch"].shape),
            }
            feat = agent.rssm.feature(next_state)
            obs_dists = agent.obs_decoder(feat)
            pred_pos = float(np.asarray(obs_dists["position"].mean()).reshape(-1)[0])
            pred_goal = float(np.asarray(obs_dists["goal"].mean()).reshape(-1)[0])
            pred_rew = float(np.asarray(agent.reward_decoder(feat).mean()))

            next_obs, reward, done, _ = env.step(action)
            true_pos = float(next_obs["position"][0])
            true_goal = float(next_obs["goal"][0])
            cur_pos = float(obs["position"][0])

            pos_errs.append(abs(pred_pos - true_pos))
            goal_errs.append(abs(pred_goal - true_goal))
            rew_errs.append(abs(pred_rew - float(reward)))
            baseline_errs.append(abs(cur_pos - true_pos))

            prev_action = action_vec
            obs = next_obs
            is_first = False

    return {
        "position_mae": float(np.mean(pos_errs)),
        "goal_mae": float(np.mean(goal_errs)),
        "reward_mae": float(np.mean(rew_errs)),
        "baseline_position_mae": float(np.mean(baseline_errs)),
        "n_steps": float(len(pos_errs)),
    }


def eval_candidate(
    agent: Agent,
    env: Point1DEnv,
    config: Config,
    n_episodes: int = 30,
    seed: int = 0,
) -> dict[str, float]:
    """候補列を計画し直しせず4歩実行し、距離の変化をランダム・正解と比較する。"""
    horizon = int(config.horizon)
    key = jax.random.PRNGKey(seed)
    cand_reductions: list[float] = []
    rand_reductions: list[float] = []
    oracle_reductions: list[float] = []
    direction_hits = 0
    direction_total = 0

    for ep_i in range(n_episodes):
        key, k_reset, k_post, k_seq, k_eval, k_rand = jax.random.split(key, 6)
        obs, _ = env.reset(seed=int(jax.random.randint(k_reset, (), 0, 2**31 - 1)))
        snap = _snapshot(env)
        start_dist = _distance(obs)

        state = get_posterior_state(
            agent,
            agent.rssm.initial(),
            jnp.zeros((action_dim(env.act_space),), dtype=jnp.float32),
            {k: jnp.asarray(v) for k, v in obs.items()},
            k_post,
            is_first=True,
        )
        seqs = sample_action_sequences(
            env.act_space, horizon, config.candidate_count, k_seq
        )
        seqs = jnp.asarray(seqs, dtype=jnp.float32)
        efes = evaluate_action_sequences(
            agent.rssm,
            agent.reward_decoder,
            agent.encoder,
            agent.obs_decoder,
            state,
            seqs,
            config,
            k_eval,
        )
        best = int(np.asarray(jnp.argmin(efes)))
        best_seq = np.asarray(seqs[best])

        if start_dist > env.max_speed:
            direction_total += 1
            if _action_toward_goal(best_seq[0], obs):
                direction_hits += 1

        cand_actions = [{"action": np.asarray(best_seq[t], dtype=np.float32)} for t in range(horizon)]
        s0, s1 = _run_actions(env, cand_actions)
        cand_reductions.append(s0 - s1)

        _restore(env, snap)
        rng = np.random.default_rng(int(jax.random.randint(k_rand, (), 0, 2**31 - 1)))
        rand_actions = []
        for _ in range(horizon):
            act = {k: space.sample(rng) for k, space in env.act_space.items()}
            rand_actions.append(act)
        r0, r1 = _run_actions(env, rand_actions)
        rand_reductions.append(r0 - r1)

        _restore(env, snap)
        oracle_actions = []
        for _ in range(horizon):
            cur = env._obs()
            oracle_actions.append(_oracle_action(env, cur))
            if env._done:
                break
            env.step(oracle_actions[-1])
        # 上のループですでに実行済みなので、距離だけ読む。
        o0 = start_dist
        o1 = _distance(env._obs())
        oracle_reductions.append(o0 - o1)

    rate = float(direction_hits / direction_total) if direction_total > 0 else 0.0
    return {
        "candidate_dist_reduction": float(np.mean(cand_reductions)),
        "random_dist_reduction": float(np.mean(rand_reductions)),
        "oracle_dist_reduction": float(np.mean(oracle_reductions)),
        "correct_direction_rate": rate,
        "n_episodes": float(n_episodes),
    }


def eval_policy(
    agent: Agent,
    env: Point1DEnv,
    config: Config,
    n_episodes: int = 30,
    seed: int = 0,
) -> dict[str, float]:
    """方策がサンプルした4手をそのまま実行し、距離の変化をランダムと比較する。"""
    horizon = int(config.horizon)
    key = jax.random.PRNGKey(seed)
    pol_reductions: list[float] = []
    rand_reductions: list[float] = []
    act_key = next(iter(env.act_space.keys()))

    for ep_i in range(n_episodes):
        key, k_reset, k_post = jax.random.split(key, 3)
        obs, _ = env.reset(seed=int(jax.random.randint(k_reset, (), 0, 2**31 - 1)))
        snap = _snapshot(env)
        start_dist = _distance(obs)

        state = get_posterior_state(
            agent,
            agent.rssm.initial(),
            jnp.zeros((action_dim(env.act_space),), dtype=jnp.float32),
            {k: jnp.asarray(v) for k, v in obs.items()},
            k_post,
            is_first=True,
        )
        prev_action = jnp.zeros((action_dim(env.act_space),), dtype=jnp.float32)
        for t in range(horizon):
            key, k_post_t = jax.random.split(key)
            if t > 0:
                state = get_posterior_state(
                    agent,
                    state,
                    prev_action,
                    {k: jnp.asarray(v) for k, v in obs.items()},
                    k_post_t,
                    is_first=False,
                )
            feat = agent.rssm.feature(state)
            dist = agent.policy(feat)
            # 評価はサンプルではなく最頻値（連続は平均）で、学習の有無を見やすくする。
            if agent.policy.discrete:
                action_vec = dist.mode()
            else:
                action_vec = dist.mean()
            action = {act_key: np.asarray(action_vec, dtype=np.float32)}
            if env._done:
                break
            obs, _, done, _ = env.step(action)
            prev_action = flatten_action(action, env.act_space)
            if done:
                break
        pol_reductions.append(start_dist - _distance(obs))

        _restore(env, snap)
        key, k_rand = jax.random.split(key)
        rng = np.random.default_rng(int(jax.random.randint(k_rand, (), 0, 2**31 - 1)))
        rand_actions = [
            {k: space.sample(rng) for k, space in env.act_space.items()}
            for _ in range(horizon)
        ]
        r0, r1 = _run_actions(env, rand_actions)
        rand_reductions.append(r0 - r1)

    return {
        "policy_dist_reduction": float(np.mean(pol_reductions)),
        "random_dist_reduction": float(np.mean(rand_reductions)),
        "n_episodes": float(n_episodes),
    }


def print_metrics(title: str, metrics: dict[str, Any]) -> None:
    print(title)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.6f}")
        else:
            print(f"  {k}: {v}")
