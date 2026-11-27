"""学習ループ。収集して世界モデルを更新し、方策モードなら方策も更新する。

このファイルが実行の入口である。環境との対話は collect_episode() だけが行う。
重みの更新そのものは agent.py の make_wm_step / make_joint_step に委譲する。

train() は環境と設定を受け取る繰り返しで、環境ごとの数値は持たない。

行動の中身は select_action と同じ2モード。収集では事後更新とまとめて
step_latent_and_action を JIT する。random_initial が真のときだけ、
1周目の行動を環境の一様サンプルにする。

更新回数は固定エポックではなく train_ratio で決める。新しい環境ステップに比例し、
チャンクをバッファへ入れた直後に行う。系列を切れる前は更新しない。
1回の更新は、方策モードなら世界モデルと方策を同じバッチ・1回の observe で進め、
候補モードなら世界モデルだけを進める。

1周目の量は random_episodes / random_steps、2周目以降は collect_episodes / collect_steps。
一般に1周目のほうが多い。episodes が 1 以上ならその本数、0 なら steps ステップ。
done が来たチャンクはそこで区切る。
"""

from __future__ import annotations

from typing import Any, Protocol

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from active_inference.agent import (
    Agent,
    _actions_to_array,
    _batch_encode,
    make_joint_step,
    make_wm_step,
    step_latent_and_action,
)
from active_inference.config import Config
from active_inference.replay_buffer import ReplayBuffer, StreamReplayBuffer
from active_inference.spaces import Space, action_dim, flatten_action


def _should_log_update(done: int) -> bool:
    """ログを出す更新番号か。ここでだけ metrics をホストへ戻す。"""
    return done == 1 or done % 10 == 0


class Env(Protocol):
    """収集と学習が環境に求めるもの。reset / step / sample_action と二つの空間。"""

    obs_space: dict[str, Space]
    act_space: dict[str, Space]

    def reset(
        self, seed: int | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]: ...

    def step(
        self, action: dict[str, np.ndarray]
    ) -> tuple[dict[str, np.ndarray], float, bool, dict[str, Any]]: ...

    def sample_action(self) -> dict[str, np.ndarray]: ...


def _require_collection(
    episodes: int,
    steps: int,
    episodes_name: str,
    steps_name: str,
) -> tuple[int, int]:
    """episodes が 1 以上ならその本数。0 なら steps。

    戻り値は (使うエピソード数, 使うステップ数)。使わない側は 0。
    """
    if episodes < 0:
        raise ValueError(f"{episodes_name} は 0 以上にしてください。")
    if episodes == 0:
        if steps <= 0:
            raise ValueError(
                f"{episodes_name} が 0 のときは {steps_name} に"
                " 1 以上のステップ数を指定してください。"
            )
        return 0, int(steps)
    return int(episodes), 0


class TrainRatio:
    """集めた環境ステップから、勾配更新の回数を決める。

    DreamerV3 の train_ratio と同じ比である。
    更新 1 回は batch_steps（batch_size * seq_len）ステップをリプレイから使う。
    したがって新しいステップ n に対する更新回数は train_ratio * n / batch_steps。
    端数は切り捨て、余りは次の呼び出しへ持ち越す。
    """

    def __init__(self, train_ratio: float, batch_steps: int):
        if train_ratio < 0:
            raise ValueError("train_ratio は 0 以上にしてください。")
        if batch_steps < 1:
            raise ValueError("batch_steps は 1 以上にしてください。")
        self._rate = float(train_ratio) / float(batch_steps)
        self._step = 0

    def updates_for(self, new_steps: int) -> int:
        if new_steps < 0:
            raise ValueError("new_steps は 0 以上にしてください。")
        prev = self._step
        self._step += int(new_steps)
        return int(self._rate * self._step) - int(self._rate * prev)


def _validate_train_config(config: Config) -> None:
    """学習を始める前に、周回数と二つの収集量を確かめる。"""
    if config.n_rounds < 1:
        raise ValueError("n_rounds は 1 以上にしてください。")
    if config.train_ratio < 0:
        raise ValueError("train_ratio は 0 以上にしてください。")
    if config.batch_size < 1 or config.seq_len < 1:
        raise ValueError("batch_size と seq_len は 1 以上にしてください。")
    if config.action_mode not in ("policy", "candidate"):
        raise ValueError(
            "action_mode は 'policy' か 'candidate' にしてください。"
            f" 受け取った値: {config.action_mode}"
        )
    if config.replay_mode not in ("episode", "stream"):
        raise ValueError(
            "replay_mode は 'episode' か 'stream' にしてください。"
            f" 受け取った値: {config.replay_mode}"
        )
    if config.replay_mode == "stream" and config.stream_capacity < 1:
        raise ValueError("stream_capacity は 1 以上にしてください。")
    # 方策の損失は想像中に KL を渡さない。efe_mode が kl だとそこで落ちる。
    if config.action_mode == "policy" and config.efe_mode != "entropy":
        raise ValueError(
            "方策の更新は想像中に KL を計算しないため、efe_mode は 'entropy' にしてください。"
        )
    _require_collection(
        config.random_episodes,
        config.random_steps,
        "random_episodes",
        "random_steps",
    )
    # 2周目があるのに収集量が無いと、1周目の学習が終わってから失敗する。
    if config.n_rounds > 1:
        _require_collection(
            config.collect_episodes,
            config.collect_steps,
            "collect_episodes",
            "collect_steps",
        )


def collect_episode(
    env: Env,
    agent: Agent | None,
    config: Config,
    key: jax.Array,
    random: bool = False,
    max_steps: int | None = None,
) -> tuple[dict[str, Any], float, jax.Array]:
    """環境と1エピソード対話し、遷移の辞書と合計報酬を返す。

    学習の更新はこの関数の外（バッファからのミニバッチ）で行う。
    ここがやるのは「観測を受け取り、行動を1つ出し、結果をリストに積む」ことだけ。

    行動の出どころは2通り。
    - random=True、または agent が None: 環境の一様サンプル。世界モデル学習用の収集。
    - それ以外: step_latent_and_action で事後状態と行動を1つの JIT から得る。
      config.action_mode が "candidate" なら候補列の比較、"policy" なら方策ネット。

    max_steps が None のときは、環境が done を返すまで続ける。
    数を渡したときは、done より先でもその歩数で打ち切る。継続する環境では
    done が来ないので、歩数を渡さないとこの関数は戻らない。
    打ち切ったチャンクの末尾 done は False になり得る。

    episode["obs"] は reset 直後の観測を先頭に含むので、長さは行動の数より1大きい。
    act / rew / done の長さはステップ数 T。後段の世界モデルは
    「行動前の観測 obs[t]」と「行動後の観測 obs[t+1]」を対にして使う。
    """
    # reset が返す観測が、このエピソードの obs[0] になる。まだ行動は出していない。
    obs, info = env.reset()
    episode = {
        "obs": {k: [v.copy()] for k, v in obs.items()},
        "act": {k: [] for k in env.act_space},
        "rew": [],
        "done": [],
    }
    # 潜在状態 {"deter", "stoch"}。ランダム収集ではエージェントを呼ばないので None。
    state = agent.rssm.initial() if agent is not None else None
    # 最初のステップには直前の行動が無い。事後の更新式は行動を受け取るので零ベクトルを置く。
    prev_action = jnp.zeros((action_dim(env.act_space),), dtype=jnp.float32)
    done = False
    total_reward = 0.0
    # 最初の1回だけ、事後更新は渡された状態を捨てて RSSM のゼロ状態から始める。
    # スカラーの jnp.bool_ にして JIT 内で jnp.where 分岐する。Python の bool だと
    # 先頭とそれ以外で別コンパイルになる。
    is_first = jnp.bool_(True)
    n_collected = 0

    # max_steps が None のときは done まで。数を渡したときは、その歩数か done の早い方で止める。
    while not done and (max_steps is None or n_collected < max_steps):
        # JAX の乱数キーは消費する。k_act は行動のサンプル用、k_post は事後分布のサンプル用。
        key, k_act, k_post = jax.random.split(key, 3)
        if random or agent is None:
            action = env.sample_action()
            action_jax = {k: jnp.asarray(v) for k, v in action.items()}
        else:
            # 事後更新と行動選択を1回の JIT にする。状態はデバイスに残し、
            # 環境へ渡す行動だけ numpy にする。
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
            # 環境の step は numpy を受け取る。
            action = {k: np.asarray(v) for k, v in action_jax.items()}

        next_obs, reward, done, info = env.step(action)
        # 積むのは「今出した行動」と「その結果」。観測は結果側（next_obs）を足していく。
        for k, v in action.items():
            episode["act"][k].append(np.asarray(v, dtype=np.float32))
        for k, v in next_obs.items():
            episode["obs"][k].append(v.copy())
        episode["rew"].append(float(reward))
        episode["done"].append(bool(done))
        total_reward += float(reward)
        # 次の事後更新に渡すため、行動辞書を1本のベクトルにする。
        prev_action = flatten_action(action_jax, env.act_space)
        obs = next_obs
        is_first = jnp.bool_(False)
        n_collected += 1

    return episode, total_reward, key


def collect_chunks(
    env: Env,
    agent: Agent | None,
    config: Config,
    key: jax.Array,
    *,
    episodes: int,
    steps: int,
    random: bool,
    episodes_name: str = "episodes",
    steps_name: str = "steps",
) -> tuple[list[dict[str, Any]], list[float], jax.Array]:
    """チャンクを集め、チャンクのリストと各チャンクの合計報酬を返す。

    episodes が 1 以上のときは、その本数だけエピソードが終わるまで集める。
    0 のときは steps ステップ集める。done が先に来たチャンクはそこで区切り、
    残りの歩数は新しいエピソードとして続ける。各チャンクは reset から始まる。

    random=True のときは一様な行動。False のときは step_latent_and_action。
    行動の中身は select_action と同じで、事後更新とまとめて JIT されている。
    戻り値の形は、どちらの集め方でも、どの環境でも同じ。
    """
    n_episodes, n_steps = _require_collection(
        episodes, steps, episodes_name, steps_name
    )
    if not random and agent is None:
        raise ValueError("モデルで集めるときは agent が必要です。")
    chunks: list[dict[str, Any]] = []
    returns: list[float] = []
    # random のときは agent に None を渡し、collect_episode は一様サンプルの枝だけを通る。
    actor = None if random else agent
    if n_episodes == 0:
        remaining = n_steps
        while remaining > 0:
            key, k_ep = jax.random.split(key)
            ep, ret, key = collect_episode(
                env, actor, config, k_ep, random=random, max_steps=remaining
            )
            taken = len(ep["rew"])
            if taken <= 0:
                raise RuntimeError("ステップを1つも集められませんでした。")
            chunks.append(ep)
            returns.append(ret)
            remaining -= taken
    else:
        for _ in range(n_episodes):
            key, k_ep = jax.random.split(key)
            ep, ret, key = collect_episode(
                env, actor, config, k_ep, random=random
            )
            chunks.append(ep)
            returns.append(ret)
    return chunks, returns, key


def collect_random(
    env: Env,
    config: Config,
    key: jax.Array,
) -> tuple[list[dict[str, Any]], list[float], jax.Array]:
    """ランダム行動のデータを集め、チャンクのリストと各チャンクの合計報酬を返す。

    random_episodes が 1 以上のときは、その本数だけエピソードが終わるまで集める。
    0 のときは random_steps ステップ集める。done が先に来たチャンクはそこで区切り、
    残りの歩数は新しいエピソードとして続ける。各チャンクは reset から始まる。
    """
    return collect_chunks(
        env,
        None,
        config,
        key,
        episodes=config.random_episodes,
        steps=config.random_steps,
        random=True,
        episodes_name="random_episodes",
        steps_name="random_steps",
    )


def batch_to_jax(batch: dict[str, Any]) -> dict[str, Any]:
    """リプレイバッファの numpy 配列を、損失関数が受け取る JAX 配列にする。

    sample_paths が保証する形は次のとおり。B はバッチ本数、T は系列長。
      obs[key]: (B, T+1, ...)  先頭は、その区間が始まる直前の観測
      act[key]: (B, T, ...)
      rew:      (B, T)
      reset:    (B, T)  True の時刻はエピソード境界。RSSM はその時刻の状態をゼロに戻す
    """
    out = {
        "obs": {k: jnp.asarray(v) for k, v in batch["obs"].items()},
        "act": {k: jnp.asarray(v) for k, v in batch["act"].items()},
        "rew": jnp.asarray(batch["rew"]),
        "reset": jnp.asarray(batch["reset"]),
    }
    return out


@eqx.filter_jit
def extract_start_states(agent: Agent, batch: dict[str, Any], key: jax.Array):
    """世界モデルの事後から方策学習の開始状態を取る。

    方策は「ある潜在状態から horizon 手先まで想像し、期待自由エネルギーが
    小さくなる行動」を学ぶ。開始点をゼロ状態にすると、バッファにある軌道と
    無関係な未来を想像してしまう。ここでは実データの系列を observe し、
    その末尾の事後状態（deter と stoch）を開始点にする。

    返した状態には stop_gradient を掛ける。方策の損失から世界モデルの
    重みへ勾配が流れないようにするためである。

    エンコードと observe は毎回同じ形なので JIT する。JIT しないと、呼び出しの
    たびにグラフを組み直し、Point1D の幅でも1回あたり約1秒かかる。
    """
    obs = batch["obs"]
    act = batch["act"]
    reset = batch["reset"]
    B = batch["rew"].shape[0]

    # 行動の結果として得た観測は obs の2件目以降。先頭は行動前なのでエンコードしない。
    obs_post = {k: v[:, 1:] for k, v in obs.items()}
    embeds = _batch_encode(agent.encoder, obs_post)
    actions = _actions_to_array(act, agent.act_space)
    init = agent.rssm.initial()
    # initial() はサンプル1個分。バッチ B 本へ同じゼロ状態を並べる。
    init_batch = jax.tree.map(lambda x: jnp.broadcast_to(x, (B,) + x.shape), init)
    keys = jax.random.split(key, B)

    def observe_one(init_state, actions_t, embeds_t, resets_t, k):
        return agent.rssm.observe(init_state, actions_t, embeds_t, resets_t, k)

    # observe は系列1本用。vmap でバッチ次元に広げる。
    _, outs = jax.vmap(observe_one)(init_batch, actions, embeds, reset, keys)
    # 末尾状態を開始点にする。
    start = {
        "deter": outs["deter"][:, -1],
        "stoch": outs["stoch"][:, -1],
    }
    return jax.tree.map(jax.lax.stop_gradient, start)


def _store_episode(buffer, episode: dict[str, Any], replay_mode: str) -> None:
    """集めた1本をバッファに入れる。stream では1ステップずつ列へ足す。

    collect_episode は毎回、環境をやり直した観測から始まる。
    そのため列に足すとき、その1本の最初のステップだけ is_first を真にする。
    """
    if replay_mode != "stream":
        buffer.add_episode(episode)
        return
    obs_keys = list(episode["obs"])
    act_keys = list(episode["act"])
    for t, (rew, done) in enumerate(zip(episode["rew"], episode["done"])):
        buffer.add_step(
            {k: episode["obs"][k][t] for k in obs_keys},
            {k: episode["act"][k][t] for k in act_keys},
            rew,
            done,
            t == 0,
            {k: episode["obs"][k][t + 1] for k in obs_keys},
        )


def _train_on_chunk(
    agent: Agent,
    buffer: ReplayBuffer | StreamReplayBuffer,
    config: Config,
    taken: int,
    credit: TrainRatio,
    wm_step,
    wm_opt_state,
    joint_step,
    pol_opt_state,
    key: jax.Array,
    rng: np.random.Generator,
    done_updates: int,
) -> tuple[Agent, Any, Any, jax.Array, int]:
    """チャンクを入れたあと、train_ratio が指定する回数だけ更新する。

    系列を切れるようになるまでは回数を進めない。そのステップは比に含めない。
    方策モードでは joint_step で、同じバッチ・1回の observe から世界モデルと
    方策をまとめて更新する。候補モードでは世界モデルだけを更新する。
    """
    if not buffer.can_sample(config.seq_len):
        return agent, wm_opt_state, pol_opt_state, key, done_updates
    n_updates = credit.updates_for(taken)
    if n_updates <= 0:
        return agent, wm_opt_state, pol_opt_state, key, done_updates
    for i in range(n_updates):
        batch = buffer.sample_paths(config.batch_size, config.seq_len, rng)
        batch = batch_to_jax(batch)
        key, k_step = jax.random.split(key)
        done = done_updates + i + 1
        if config.action_mode == "policy":
            # DreamerV3 の train 1 回に対応する。observe は1回。
            agent, wm_opt_state, pol_opt_state, _loss, metrics = joint_step(
                agent, wm_opt_state, pol_opt_state, batch, k_step
            )
            if _should_log_update(done):
                m = {k: float(v) for k, v in metrics.items()}
                print(
                    f"  update {done:03d} "
                    f"total={m['loss/total']:.4f} "
                    f"recon={m['loss/recon']:.4f} "
                    f"rew={m['loss/reward']:.4f} "
                    f"kl={m['loss/kl']:.4f} "
                    f"policy={m['policy/loss']:.4f} "
                    f"efe={m['policy/efe']:.4f}"
                )
        else:
            agent, wm_opt_state, _loss, metrics = wm_step(
                agent, wm_opt_state, batch, k_step
            )
            if _should_log_update(done):
                m = {k: float(v) for k, v in metrics.items()}
                print(
                    f"  update {done:03d} "
                    f"total={m['loss/total']:.4f} "
                    f"recon={m['loss/recon']:.4f} "
                    f"rew={m['loss/reward']:.4f} "
                    f"kl={m['loss/kl']:.4f}"
                )
    return agent, wm_opt_state, pol_opt_state, key, done_updates + n_updates


def train(
    env: Env,
    config: Config | None = None,
    agent: Agent | None = None,
) -> Agent:
    """環境と設定を受け取り、収集と更新を n_rounds 周繰り返す。

    行動は step_latent_and_action。中身は select_action と同じ2モードで、
    事後更新とまとめて JIT する。random_initial が真のときだけ、1周目を一様サンプルにする。
    チャンクをバッファに入れるたびに、train_ratio に従って更新する。
    方策モードでは、同じバッチ・1回の observe で世界モデルと方策を更新し、
    次のチャンクの行動選択に使う。

    agent を渡したときは Agent.create をせず、その重みから学習を続ける。
    オプティマイザ状態は毎回作り直す（途中の momentum は捨てる）。

    1周目の量は random_episodes / random_steps、
    2周目以降の量は collect_episodes / collect_steps。
    環境ごとの次元や離散・連続の切り替えは、この関数には書かない。
    """
    config = config or Config()
    _validate_train_config(config)

    # JAX の乱数はキーを split して配る。同じキーを二つの用途に使い回さない。
    key = jax.random.PRNGKey(config.seed)
    key, k_agent = jax.random.split(key)
    # 入出力の次元は環境が宣言する obs_space / act_space から決まる。
    # 評価用に段階を分けるときは、学習済みの agent を受け取り直す。
    if agent is None:
        agent = Agent.create(env.obs_space, env.act_space, config, k_agent)

    if config.replay_mode == "stream":
        buffer = StreamReplayBuffer(capacity=config.stream_capacity)
    else:
        buffer = ReplayBuffer(capacity=config.buffer_capacity)
    # Equinox のモジュールには配列でない設定値も入っている。勾配の対象は配列だけ。
    params = eqx.filter(agent, eqx.is_inexact_array)
    wm_step = None
    joint_step = None
    pol_opt_state = None
    if config.action_mode == "policy":
        # 方策モードは、世界モデルと方策を同じバッチ・1回の observe で更新する。
        wm_opt, pol_opt, joint_step = make_joint_step(config)
        wm_opt_state = wm_opt.init(params)
        pol_opt_state = pol_opt.init(params)
    else:
        # 候補モードは世界モデルだけを更新する。
        wm_opt, wm_step = make_wm_step(config)
        wm_opt_state = wm_opt.init(params)
    # ミニバッチを切り出す乱数は numpy。上の JAX キーとは別系統。
    rng = np.random.default_rng(config.seed)
    credit = TrainRatio(config.train_ratio, config.batch_size * config.seq_len)
    done_updates = 0

    for round_i in range(config.n_rounds):
        # 1周目と2周目以降で、集める本数（またはステップ数）を別に取る。
        # 一様サンプルにするのは、オプションを立てた1周目だけ。
        if round_i == 0:
            episodes = config.random_episodes
            steps = config.random_steps
            episodes_name = "random_episodes"
            steps_name = "random_steps"
            random = bool(config.random_initial)
        else:
            episodes = config.collect_episodes
            steps = config.collect_steps
            episodes_name = "collect_episodes"
            steps_name = "collect_steps"
            random = False
        if random:
            how = "ランダムに"
        elif config.action_mode == "policy":
            how = "方策で"
        else:
            how = "候補で"
        n_episodes, n_steps = _require_collection(
            episodes, steps, episodes_name, steps_name
        )
        if n_episodes == 0:
            amount = f"{n_steps} ステップ"
        else:
            amount = f"{n_episodes} エピソード"
        print(f"周回 {round_i + 1}/{config.n_rounds}: {how} {amount}集めます...")

        returns: list[float] = []
        # collect_chunks と同じ区切り。学習をチャンクのあいだに挟むため、ここでは1本ずつ進める。
        if n_episodes == 0:
            remaining = n_steps
            while remaining > 0:
                key, k_ep = jax.random.split(key)
                actor = None if random else agent
                ep, ret, key = collect_episode(
                    env, actor, config, k_ep, random=random, max_steps=remaining
                )
                taken = len(ep["rew"])
                if taken <= 0:
                    raise RuntimeError("ステップを1つも集められませんでした。")
                _store_episode(buffer, ep, config.replay_mode)
                returns.append(ret)
                agent, wm_opt_state, pol_opt_state, key, done_updates = _train_on_chunk(
                    agent,
                    buffer,
                    config,
                    taken,
                    credit,
                    wm_step,
                    wm_opt_state,
                    joint_step,
                    pol_opt_state,
                    key,
                    rng,
                    done_updates,
                )
                remaining -= taken
        else:
            for _ in range(n_episodes):
                key, k_ep = jax.random.split(key)
                actor = None if random else agent
                ep, ret, key = collect_episode(
                    env, actor, config, k_ep, random=random
                )
                taken = len(ep["rew"])
                if taken <= 0:
                    raise RuntimeError("ステップを1つも集められませんでした。")
                _store_episode(buffer, ep, config.replay_mode)
                returns.append(ret)
                agent, wm_opt_state, pol_opt_state, key, done_updates = _train_on_chunk(
                    agent,
                    buffer,
                    config,
                    taken,
                    credit,
                    wm_step,
                    wm_opt_state,
                    joint_step,
                    pol_opt_state,
                    key,
                    rng,
                    done_updates,
                )
        print(f"  mean return: {float(np.mean(returns)):.3f}")
        print(f"  累計の更新: {done_updates} 回")

    if config.train_ratio > 0 and not buffer.can_sample(config.seq_len):
        raise RuntimeError(
            f"長さ {config.seq_len} 以上のチャンクが無く、学習できません。"
        )
    return agent


def main() -> None:
    """Point1D の実験を起動する。数値と離散・連続の切り替えは experiments/point1d.py にある。"""
    from active_inference.experiments.point1d import main as run_point1d

    run_point1d()


if __name__ == "__main__":
    main()
