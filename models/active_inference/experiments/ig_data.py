"""情報利得格子向けのスクリプト・被覆データ収集。

collect_episode と同じ形のエピソード辞書を返す。
  obs: dict[str, list]  長さ T+1
  act: dict[str, list]  長さ T
  rew: list[float]      長さ T
  done: list[bool]      長さ T
"""

from __future__ import annotations

from typing import Any

import numpy as np

from active_inference.envs.info_grid import (
    CELL_C,
    CELL_K,
    CUE,
    InfoGridEnv,
    REWARD_L,
    REWARD_R,
    START,
    SYM_A,
    SYM_B,
)


def _one_hot(idx: int) -> np.ndarray:
    a = np.zeros(4, dtype=np.float32)
    a[idx] = 1.0
    return a


def _empty_episode(obs: dict[str, np.ndarray]) -> dict[str, Any]:
    return {
        "obs": {k: [v.copy()] for k, v in obs.items()},
        "act": {"action": []},
        "rew": [],
        "done": [],
    }


def _append_step(
    episode: dict[str, Any],
    action: np.ndarray,
    next_obs: dict[str, np.ndarray],
    reward: float,
    done: bool,
) -> None:
    episode["act"]["action"].append(np.asarray(action, dtype=np.float32))
    for k, v in next_obs.items():
        episode["obs"][k].append(v.copy())
    episode["rew"].append(float(reward))
    episode["done"].append(bool(done))


def _run_actions(
    env: InfoGridEnv,
    actions: list[np.ndarray],
    *,
    force_side: str | None = None,
    force_cue: np.ndarray | None = None,
) -> tuple[dict[str, Any], float]:
    """決めた行動列を実行し、エピソード辞書と合計報酬を返す。"""
    obs, _ = env.reset()
    if force_side is not None:
        env._correct_side = force_side  # type: ignore[assignment]
        env._cue_symbol = (SYM_A if force_side == "L" else SYM_B).copy()
    if force_cue is not None:
        env._cue_symbol = np.asarray(force_cue, dtype=np.float32).copy()
    # force 後は観測をやり直す（スタート地点の観測は空なので実害は小さい）。
    obs = env._obs()
    episode = _empty_episode(obs)
    total = 0.0
    done = False
    for action in actions:
        if done:
            break
        next_obs, reward, done, _ = env.step(action)
        _append_step(episode, action, next_obs, reward, done)
        total += float(reward)
    # 報酬版で locked が残っているときは、終わるまで適当な行動で埋める。
    while not done:
        next_obs, reward, done, _ = env.step(_one_hot(0))
        _append_step(episode, _one_hot(0), next_obs, reward, done)
        total += float(reward)
    return episode, total


def _terminal_cells(env: InfoGridEnv) -> set[tuple[int, int]]:
    """入った瞬間に終わるマス。そこからの外壁衝突はデータに残せない。"""
    if env.mode == "reward":
        return {REWARD_L, REWARD_R}
    return {CELL_K, CELL_C}


def required_bumps(env: InfoGridEnv) -> set[tuple[int, int, int]]:
    """各マスで外壁に当たる行動の集合。(x, y, action_idx)。

    入って終わるマスは除外する。到着のあとに行動を出せないため。
    """
    out: set[tuple[int, int, int]] = set()
    deltas = {0: (-1, 0), 1: (1, 0), 2: (0, 1), 3: (0, -1)}
    skip = _terminal_cells(env)
    for x, y in env.cells:
        if (x, y) in skip:
            continue
        for idx, (dx, dy) in deltas.items():
            nx, ny = x + dx, y + dy
            if not env._in_bounds(nx, ny):
                out.add((x, y, idx))
    return out


def _coverage_complete(
    visited: set[tuple[int, int]],
    bumps: set[tuple[int, int, int]],
    env: InfoGridEnv,
) -> bool:
    return visited >= set(env.cells) and bumps >= required_bumps(env)


def collect_coverage(
    env: InfoGridEnv,
    rng: np.random.Generator,
    n_min: int = 20,
    max_episodes: int = 5000,
) -> list[dict[str, Any]]:
    """全マスと各外壁衝突が1回は出るまでランダムエピソードを集める。

    n_min 本は最低集める。足りなければ被覆が終わるまで続ける。
    """
    visited: set[tuple[int, int]] = set()
    bumps: set[tuple[int, int, int]] = set()
    episodes: list[dict[str, Any]] = []

    while len(episodes) < n_min or not _coverage_complete(visited, bumps, env):
        if len(episodes) >= max_episodes:
            missing_cells = set(env.cells) - visited
            missing_bumps = required_bumps(env) - bumps
            raise RuntimeError(
                "被覆が終わりませんでした。"
                f" missing_cells={missing_cells} missing_bumps={missing_bumps}"
            )
        obs, info = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        episode = _empty_episode(obs)
        visited.add(info["position"])
        done = False
        while not done:
            # locked 中の留まりは外壁衝突に数えない。
            before = env.position
            action = env.sample_action()["action"]
            next_obs, reward, done, step_info = env.step(action)
            _append_step(episode, action, next_obs, reward, done)
            visited.add(step_info["position"])
            if step_info.get("bumped") and not step_info.get("locked"):
                idx = int(np.argmax(action))
                bumps.add((before[0], before[1], idx))
        episodes.append(episode)
    return episodes


def _shortcut_path(side: str) -> list[np.ndarray]:
    """スタートから報酬マスへの近道。L は上上右、R は下下右。"""
    if side == "L":
        return [_one_hot(2), _one_hot(2), _one_hot(1)]
    return [_one_hot(3), _one_hot(3), _one_hot(1)]


def _cue_then_side_path(side: str) -> list[np.ndarray]:
    """手がかり経由で報酬マスへ。右右のあと、L なら上上左、R なら下下左。"""
    head = [_one_hot(1), _one_hot(1)]
    if side == "L":
        # CUE (2,0) -> (2,1) -> (2,2) -> (1,2)
        return head + [_one_hot(2), _one_hot(2), _one_hot(0)]
    # CUE (2,0) -> (2,-1) -> (2,-2) -> (1,-2)
    return head + [_one_hot(3), _one_hot(3), _one_hot(0)]


def collect_scripted_reward(
    env: InfoGridEnv,
    n_episodes: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    """半分は近道、半分は手がかり経由。どちらも左右を等確率で入る。"""
    if env.mode != "reward":
        raise ValueError("collect_scripted_reward は mode='reward' 用です。")
    if n_episodes < 2 or n_episodes % 2 != 0:
        raise ValueError("n_episodes は 2 以上の偶数にしてください。")
    half = n_episodes // 2
    episodes: list[dict[str, Any]] = []
    for _ in range(half):
        side = "L" if rng.random() < 0.5 else "R"
        # 正しい側は別抽選。入る側と独立なので、当たりと外れが両方残る。
        correct = "L" if rng.random() < 0.5 else "R"
        ep, _ = _run_actions(env, _shortcut_path(side), force_side=correct)
        episodes.append(ep)
    for _ in range(half):
        side = "L" if rng.random() < 0.5 else "R"
        correct = "L" if rng.random() < 0.5 else "R"
        ep, _ = _run_actions(env, _cue_then_side_path(side), force_side=correct)
        episodes.append(ep)
    return episodes


def collect_scripted_no_reward(
    env: InfoGridEnv,
    n_episodes: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    """半分は K へ、半分は C へ。C の二つの記号は等確率。"""
    if env.mode != "no_reward":
        raise ValueError("collect_scripted_no_reward は mode='no_reward' 用です。")
    if n_episodes < 2 or n_episodes % 2 != 0:
        raise ValueError("n_episodes は 2 以上の偶数にしてください。")
    half = n_episodes // 2
    episodes: list[dict[str, Any]] = []
    to_k = [_one_hot(2), _one_hot(2)]
    to_c = [_one_hot(1), _one_hot(1)]
    for _ in range(half):
        ep, _ = _run_actions(env, to_k)
        assert env.position == CELL_K or ep["done"][-1]
        episodes.append(ep)
    for _ in range(half):
        cue = SYM_A if rng.random() < 0.5 else SYM_B
        ep, _ = _run_actions(env, to_c, force_cue=cue)
        assert env.position == CELL_C or ep["done"][-1]
        episodes.append(ep)
    return episodes
