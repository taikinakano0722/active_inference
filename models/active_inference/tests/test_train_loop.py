"""収集と学習の周回。1周目と2周目以降で収集量が違ってよい。"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from active_inference.agent import Agent
from active_inference.config import Config
from active_inference.envs.point1d import Point1DEnv
from active_inference.train import TrainRatio, train


class _CountingEnv:
    """reset と一様サンプルの回数を数える。学習ループがどちらの収集を呼んだかの確認用。"""

    def __init__(self, env: Point1DEnv):
        self._env = env
        self.obs_space = env.obs_space
        self.act_space = env.act_space
        self.resets = 0
        self.steps = 0
        self.random_actions = 0

    def reset(self, seed: int | None = None):
        self.resets += 1
        return self._env.reset(seed=seed)

    def step(self, action):
        self.steps += 1
        return self._env.step(action)

    def sample_action(self):
        self.random_actions += 1
        return self._env.sample_action()


def _tiny(**kwargs) -> Config:
    values = dict(
        deter_dim=32,
        stoch_dim=8,
        hidden_dim=32,
        vec_units=16,
        batch_size=4,
        seq_len=4,
        horizon=2,
        candidate_count=4,
        efe_mode="entropy",
        action_mode="policy",
        n_rounds=2,
        # batch_size * seq_len = 16。比 1 なら 16 環境ステップで更新 1 回。
        train_ratio=1.0,
        seed=0,
    )
    values.update(kwargs)
    return Config(**values)


def test_train_second_round_uses_its_own_budget_and_the_model():
    # 1周目はステップ数、2周目はエピソード数。単位も本数も違ってよい。
    # max_steps=6 なので、10 ステップは 6+4 に割れ、そのあと 1 エピソードで 6 歩。
    env = Point1DEnv(discrete=True, max_steps=6, seed=0)
    wrapped = _CountingEnv(env)
    agent = train(
        wrapped,
        _tiny(
            random_initial=True,
            random_episodes=0,
            random_steps=10,
            collect_episodes=1,
            collect_steps=0,
        ),
    )
    assert isinstance(agent, Agent)
    assert wrapped.steps == 16
    assert wrapped.resets == 3
    # 一様サンプルは1周目の 10 歩だけ。2周目は select_action。
    assert wrapped.random_actions == 10


def test_train_without_random_initial_uses_select_action():
    env = Point1DEnv(discrete=True, max_steps=6, seed=0)
    wrapped = _CountingEnv(env)
    train(
        wrapped,
        _tiny(
            random_initial=False,
            random_episodes=0,
            random_steps=10,
            collect_episodes=1,
            collect_steps=0,
        ),
    )
    assert wrapped.steps == 16
    assert wrapped.random_actions == 0


def test_train_candidate_mode_skips_policy_and_uses_model():
    env = Point1DEnv(discrete=True, max_steps=4, seed=1)
    wrapped = _CountingEnv(env)
    agent = train(
        wrapped,
        _tiny(
            action_mode="candidate",
            random_initial=True,
            horizon=1,
            candidate_count=3,
            random_episodes=1,
            collect_episodes=1,
            seq_len=4,
        ),
    )
    assert isinstance(agent, Agent)
    assert wrapped.steps == 8
    assert wrapped.resets == 2
    assert wrapped.random_actions == 4


def test_missing_collect_budget_raises_before_training():
    env = Point1DEnv(discrete=True, max_steps=5, seed=0)
    cfg = Config(
        n_rounds=2,
        random_episodes=1,
        collect_episodes=0,
        collect_steps=0,
    )
    try:
        train(env, cfg)
    except ValueError as exc:
        assert "collect_steps" in str(exc)
    else:
        raise AssertionError("2周目の収集量が無いときはエラーになるべき")


def test_n_rounds_must_be_positive():
    env = Point1DEnv(discrete=True, max_steps=5, seed=0)
    cfg = Config(n_rounds=0, random_episodes=1)
    try:
        train(env, cfg)
    except ValueError as exc:
        assert "n_rounds" in str(exc)
    else:
        raise AssertionError("n_rounds が 0 のときはエラーになるべき")


def test_train_ratio_matches_replay_steps():
    # 32 環境ステップにつき、1024 ステップ分のリプレイを使う更新が 1 回。
    credit = TrainRatio(32.0, 1024)
    assert credit.updates_for(32) == 1
    assert credit.updates_for(31) == 0
    assert credit.updates_for(1) == 1
    # 系列を切る前のステップは、呼び出し側が updates_for を呼ばない。
    fresh = TrainRatio(32.0, 1024)
    assert fresh.updates_for(0) == 0


def test_initial_policy_is_broad():
    cfg = Config(deter_dim=32, stoch_dim=8, hidden_dim=32, vec_units=16)
    key = jax.random.PRNGKey(0)
    discrete = Point1DEnv(discrete=True, max_steps=4, seed=0)
    agent = Agent.create(discrete.obs_space, discrete.act_space, cfg, key)
    feat = jnp.zeros((agent.rssm.feature_dim(),))
    ent = float(agent.policy(feat).entropy())
    assert ent > 0.9 * float(jnp.log(3.0))

    key, k_cont = jax.random.split(key)
    continuous = Point1DEnv(discrete=False, max_speed=1.0, max_steps=4, seed=0)
    agent_c = Agent.create(continuous.obs_space, continuous.act_space, cfg, k_cont)
    std = float(agent_c.policy(feat).stddev().mean())
    assert std > 0.5


def test_policy_mode_requires_entropy_efe():
    env = Point1DEnv(discrete=True, max_steps=5, seed=0)
    cfg = Config(
        n_rounds=1,
        random_episodes=1,
        action_mode="policy",
        efe_mode="kl",
    )
    try:
        train(env, cfg)
    except ValueError as exc:
        assert "entropy" in str(exc)
    else:
        raise AssertionError("方策モードで efe_mode が kl のときはエラーになるべき")


if __name__ == "__main__":
    test_train_ratio_matches_replay_steps()
    test_initial_policy_is_broad()
    test_missing_collect_budget_raises_before_training()
    test_n_rounds_must_be_positive()
    test_policy_mode_requires_entropy_efe()
    test_train_second_round_uses_its_own_budget_and_the_model()
    test_train_without_random_initial_uses_select_action()
    test_train_candidate_mode_skips_policy_and_uses_model()
    print("test_train_loop: ok")
