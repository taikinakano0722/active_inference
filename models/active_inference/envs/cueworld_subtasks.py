"""cueworld/subtasks_for_cueworld の環境を、このモデルの Env 契約へつなぐアダプタ。

subtasks 側は Gymnasium の契約である。
  reset() -> (obs: ndarray, info)
  step(int) -> (obs, reward, terminated, truncated, info)
このモデルの train() が求める契約は次のとおり。
  obs_space / act_space: dict[str, Space]
  reset() -> (dict[str, ndarray], info)
  step(dict[str, one-hot]) -> (dict[str, ndarray], reward, done, info)
  sample_action() -> dict[str, one-hot]

環境のコードは subtasks 側が正本で、ここへは写さない。ファイルを直接読み込む。
subtasks のパッケージ名はトップレベルの `envs` で、このパッケージの
`active_inference.envs` と紛らわしいため、sys.path には足さず、
モジュールを別名で読み込む。

置き場所の探し方（先に見つかったもの）。
  1. 環境変数 CUEWORLD_SUBTASKS_DIR
  2. /cueworld/subtasks_for_cueworld   （コンテナ内。docker-compose.yml でマウント）
  3. <proposed_model の親>/cueworld/subtasks_for_cueworld   （ホストで直接動かすとき）
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from active_inference.spaces import Space

OBS_KEY = "obs"
ACT_KEY = "action"

# subtasks のファイル名と、そこにあるクラス名。
_ENV_FILES = {
    "sanity": ("sanity.py", "SanityEnv"),
    "communication": ("communication.py", "CommunicationEnv"),
}


def subtasks_dir() -> Path:
    """subtasks_for_cueworld のディレクトリを返す。見つからなければ FileNotFoundError。"""
    candidates: list[Path] = []
    env_dir = os.environ.get("CUEWORLD_SUBTASKS_DIR")
    if env_dir:
        candidates.append(Path(env_dir))
    candidates.append(Path("/cueworld/subtasks_for_cueworld"))
    # このファイル: proposed_model/models/active_inference/envs/cueworld_subtasks.py
    candidates.append(Path(__file__).resolve().parents[4] / "cueworld" / "subtasks_for_cueworld")
    for c in candidates:
        if (c / "envs").is_dir():
            return c
    raise FileNotFoundError(
        "subtasks_for_cueworld が見つかりません。探した場所: "
        + ", ".join(str(c) for c in candidates)
        + "。コンテナならdocker-compose.yml の cueworld のマウントを確認するか、"
        " CUEWORLD_SUBTASKS_DIR を指定してください。"
    )


def _load_module(name: str) -> ModuleType:
    """subtasks/envs/<file> を `cueworld_subtasks_<name>` という名前で読み込む。"""
    file_name, _ = _ENV_FILES[name]
    mod_name = f"cueworld_subtasks_{name}"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    path = subtasks_dir() / "envs" / file_name
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"{path} を読み込めません。")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


def load_env_class(name: str) -> type:
    """'sanity' か 'communication' のクラスを返す。"""
    if name not in _ENV_FILES:
        raise ValueError(f"name は {sorted(_ENV_FILES)} のどれかにしてください: {name!r}")
    return getattr(_load_module(name), _ENV_FILES[name][1])


class GymSubtaskEnv:
    """Gymnasium の離散行動・ベクトル観測の環境を、このモデルの Env にする。

    - 観測は {"obs": (D,) float32}。
    - 行動は {"action": one-hot (n,)}。argmax を int にして渡す。
    - done は terminated or truncated。subtasks の環境は max_steps で truncated になる。
    - info は subtasks のものに is_first / is_last / terminated / truncated / action を足す。
    """

    def __init__(self, gym_env: Any, seed: int | None = None):
        self.gym_env = gym_env
        n = int(gym_env.action_space.n)
        dim = int(np.prod(gym_env.observation_space.shape))
        self.n_actions = n
        self.obs_space: dict[str, Space] = {OBS_KEY: Space(np.float32, (dim,))}
        self.act_space: dict[str, Space] = {
            ACT_KEY: Space(np.float32, (n,), discrete=True, n=n)
        }
        self._rng = np.random.default_rng(seed)

    def _wrap_obs(self, obs: np.ndarray) -> dict[str, np.ndarray]:
        return {OBS_KEY: np.asarray(obs, dtype=np.float32).reshape(-1)}

    def _action_to_int(self, action: Any) -> int:
        if isinstance(action, dict):
            action = action[ACT_KEY]
        a = np.asarray(action)
        if a.ndim == 0:
            return int(a)
        return int(np.argmax(a.reshape(-1)))

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        obs, info = self.gym_env.reset(seed=seed, options=options)
        info = dict(info)
        info["is_first"] = True
        return self._wrap_obs(obs), info

    def step(
        self, action: Any
    ) -> tuple[dict[str, np.ndarray], float, bool, dict[str, Any]]:
        a = self._action_to_int(action)
        obs, reward, terminated, truncated, info = self.gym_env.step(a)
        done = bool(terminated or truncated)
        info = dict(info)
        info.update(
            is_first=False,
            is_last=done,
            terminated=bool(terminated),
            truncated=bool(truncated),
            action=a,
        )
        return self._wrap_obs(obs), float(reward), done, info

    def sample_action(self) -> dict[str, np.ndarray]:
        return {k: v.sample(self._rng) for k, v in self.act_space.items()}


def make_sanity_env(
    *,
    max_steps: int,
    n_actions: int = 3,
    n_symbols: int = 4,
    reward_mode: bool = False,
    symbol_noise: float = 0.0,
    obs_noise: float = 0.0,
    seed: int | None = None,
) -> GymSubtaskEnv:
    cls = load_env_class("sanity")
    env = cls(
        max_steps=max_steps,
        n_actions=n_actions,
        n_symbols=n_symbols,
        reward_mode=reward_mode,
        symbol_noise=symbol_noise,
        obs_noise=obs_noise,
        seed=seed,
    )
    return GymSubtaskEnv(env, seed=seed)


def make_communication_env(
    *,
    max_steps: int,
    n_meaningful: int = 1,
    dependency: str = "short",
    obs_noise: float = 0.0,
    seed: int | None = None,
) -> GymSubtaskEnv:
    cls = load_env_class("communication")
    env = cls(
        max_steps=max_steps,
        n_meaningful=n_meaningful,
        dependency=dependency,
        obs_noise=obs_noise,
        seed=seed,
    )
    return GymSubtaskEnv(env, seed=seed)
