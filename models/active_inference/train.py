"""学習ループ。ランダム収集 → 世界モデル → 候補/方策。

このファイルが実行の入口である。処理は train() に上から順に書いてあり、
環境との対話は collect_episode() だけが行う。重みの更新そのものは
agent.py の make_wm_step / make_policy_step に委譲する。

一段落で通る流れは次の5段階。

1. ランダム行動でデータを集め、リプレイバッファに入れる。
   既定はエピソード本数。random_episodes が 0 のときは random_steps ステップ。
2. そのデータで世界モデル（エンコーダ、RSSM、観測デコーダ、報酬デコーダ）を学ぶ。
3. 候補モードで行動し、ランダムの平均リターンと比べる。方策の重みはまだ更新しない。
4. バッファ上の事後状態を開始点にして、方策ネットを学ぶ。
5. 方策モードで行動し、再びランダムの平均リターンと比べる。
"""

from __future__ import annotations

import argparse
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from active_inference.agent import (
    Agent,
    get_posterior_state,
    make_policy_step,
    make_wm_step,
    select_action,
    world_model_loss,
)
from active_inference.config import Config
from active_inference.envs.point1d import Point1DEnv
from active_inference.replay_buffer import ReplayBuffer
from active_inference.spaces import action_dim, flatten_action


def collect_episode(
    env: Point1DEnv,
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
    - それ以外: 今の観測を事後状態に畳み込み、select_action で1手選ぶ。
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
    is_first = True
    n_collected = 0

    # max_steps が None のときは done まで。数を渡したときは、その歩数か done の早い方で止める。
    while not done and (max_steps is None or n_collected < max_steps):
        # JAX の乱数キーは消費する。k_act は行動のサンプル用、k_post は事後分布のサンプル用。
        key, k_act, k_post = jax.random.split(key, 3)
        if random or agent is None:
            action = env.sample_action()
            action_jax = {k: jnp.asarray(v) for k, v in action.items()}
        else:
            # 直前の行動と今の観測から、RSSM の事後状態を1歩進める。
            if state is not None:
                state = get_posterior_state(
                    agent,
                    state,
                    prev_action,
                    {k: jnp.asarray(v) for k, v in obs.items()},
                    k_post,
                    is_first=is_first,
                )
            # 候補モードなら EFE 最小の列の先頭、方策モードなら方策分布から1サンプル。
            action_jax, _ = select_action(agent, state, config, k_act)
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
        is_first = False
        n_collected += 1

    return episode, total_reward, key


def collect_random(
    env: Point1DEnv,
    config: Config,
    key: jax.Array,
) -> tuple[list[dict[str, Any]], list[float], jax.Array]:
    """ランダム行動のデータを集め、チャンクのリストと各チャンクの合計報酬を返す。

    random_episodes が 1 以上のときは、その本数だけエピソードが終わるまで集める。
    0 のときは random_steps ステップ集める。done が先に来たチャンクはそこで区切り、
    残りの歩数は新しいエピソードとして続ける。各チャンクは reset から始まる。
    """
    chunks: list[dict[str, Any]] = []
    returns: list[float] = []
    if config.random_episodes == 0:
        if config.random_steps <= 0:
            raise ValueError(
                "random_episodes が 0 のときは random_steps に"
                " 1 以上のステップ数を指定してください。"
            )
        remaining = int(config.random_steps)
        while remaining > 0:
            key, k_ep = jax.random.split(key)
            # agent に None を渡すと、collect_episode はランダム行動の枝だけを通る。
            ep, ret, key = collect_episode(
                env, None, config, k_ep, random=True, max_steps=remaining
            )
            taken = len(ep["rew"])
            if taken <= 0:
                raise RuntimeError("ステップを1つも集められませんでした。")
            chunks.append(ep)
            returns.append(ret)
            remaining -= taken
    else:
        for _ in range(config.random_episodes):
            key, k_ep = jax.random.split(key)
            # agent に None を渡すと、collect_episode はランダム行動の枝だけを通る。
            ep, ret, key = collect_episode(env, None, config, k_ep, random=True)
            chunks.append(ep)
            returns.append(ret)
    return chunks, returns, key


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


def extract_start_states(agent: Agent, batch: dict[str, Any], key: jax.Array):
    """世界モデルの事後から方策学習の開始状態を取る。

    方策は「ある潜在状態から horizon 手先まで想像し、期待自由エネルギーが
    小さくなる行動」を学ぶ。開始点をゼロ状態にすると、バッファにある軌道と
    無関係な未来を想像してしまう。ここでは実データの系列を observe し、
    その末尾の事後状態（deter と stoch）を開始点にする。

    返した状態には stop_gradient を掛ける。方策の損失から世界モデルの
    重みへ勾配が流れないようにするためである。
    """
    obs = batch["obs"]
    act = batch["act"]
    reset = batch["reset"]
    B = batch["rew"].shape[0]
    T = batch["rew"].shape[1]
    from active_inference.agent import _actions_to_array, _batch_encode

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


def train(config: Config | None = None, discrete: bool = True) -> Agent:
    """第一段の閉じたループを一度走らせ、学習済みエージェントを返す。

    discrete=True は行動が「負方向・停止・正方向」の3値。
    False は同じ力学のまま、行動が速度そのもの（連続値）になる。
    """
    config = config or Config()
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
    # 呼び出し側が random_episodes=0 にしたときだけ、random_steps による収集を残す。
    # それ以外は、この Point1D 実験の既定であるエピソード本数にする。
    if config.random_episodes != 0:
        config.random_episodes = 30
    config.wm_epochs = 40
    config.policy_epochs = 30
    # 情報利得は、想像した stoch の事前エントロピー。観測をデコードし直す KL は使わない。
    config.efe_mode = "entropy"
    # 実用項（予測報酬の符号反転）を主にし、情報利得の寄与は小さくしておく。
    config.epistemic_weight = 0.01
    config.action_mode = "candidate"

    # 数直線上の位置を速度で動かす環境。1ステップの移動は max_speed までなので、
    # 目標へは複数ステップかかる。エピソードは 40 ステップで打ち切る。
    env = Point1DEnv(discrete=discrete, max_speed=0.1, max_steps=40, seed=config.seed)
    # JAX の乱数はキーを split して配る。同じキーを二つの用途に使い回さない。
    key = jax.random.PRNGKey(config.seed)
    key, k_agent = jax.random.split(key)
    # 入出力の次元は環境が宣言する obs_space / act_space から決まる。
    agent = Agent.create(env.obs_space, env.act_space, config, k_agent)

    # --- 1. ランダム収集 ---------------------------------------------------
    # 世界モデルは、自分が選んだ行動ではなく、一様な行動の軌道から先に学ぶ。
    # 既定はエピソード本数。random_episodes が 0 のときだけステップ数で集める。
    buffer = ReplayBuffer(capacity=500)
    if config.random_episodes == 0:
        print(f"Collecting {config.random_steps} random steps...")
    else:
        print(f"Collecting {config.random_episodes} random episodes...")
    chunks, returns, key = collect_random(env, config, key)
    for ep in chunks:
        buffer.add_episode(ep)
    print(f"  mean return: {np.mean(returns):.3f}")

    # --- 2. 世界モデルの学習 -----------------------------------------------
    # wm_step は world_model_loss（再構成 + 報酬の対数尤度 + 事前と事後の KL）の
    # 勾配でエージェントを1回更新する。Adam のモーメントは wm_opt_state に別に持つ。
    wm_opt, wm_step = make_wm_step(config)
    # Equinox のモジュールには配列でない設定値も入っている。勾配の対象は配列だけ。
    wm_opt_state = wm_opt.init(eqx.filter(agent, eqx.is_inexact_array))
    # ミニバッチを切り出す乱数は numpy。上の JAX キーとは別系統。
    rng = np.random.default_rng(config.seed)

    print(f"Training world model for {config.wm_epochs} epochs...")
    last_metrics = {}
    for epoch in range(config.wm_epochs):
        # バッファ内のエピソードから長さ seq_len の区間を batch_size 本切り出す。
        batch = buffer.sample_paths(config.batch_size, config.seq_len, rng)
        batch = batch_to_jax(batch)
        key, k_step = jax.random.split(key)
        agent, wm_opt_state, loss, metrics = wm_step(
            agent, wm_opt_state, batch, k_step
        )
        last_metrics = {k: float(v) for k, v in metrics.items()}
        # total = recon（観測の再構成）+ rew（報酬の予測）+ kl（事前と事後のずれ）
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"  epoch {epoch+1:03d} "
                f"total={last_metrics['loss/total']:.4f} "
                f"recon={last_metrics['loss/recon']:.4f} "
                f"rew={last_metrics['loss/reward']:.4f} "
                f"kl={last_metrics['loss/kl']:.4f}"
            )

    # 候補モード評価
    # 方策ネットは使わない。horizon 手先までの行動列を作り、期待自由エネルギーの
    # 和が最小の列の先頭1手だけを環境で実行する。離散行動のときは 3^4 = 81 本を
    # 全列挙し、連続行動のときは行動範囲から 81 本をサンプルする。
    # ここで集めた軌道もバッファへ足し、次の方策学習の材料にする。
    print("Evaluating candidate planner...")
    config.action_mode = "candidate"
    cand_returns = []
    for i in range(10):
        key, k_ep = jax.random.split(key)
        ep, ret, key = collect_episode(env, agent, config, k_ep, random=False)
        cand_returns.append(ret)
        buffer.add_episode(ep)
    print(f"  candidate mean return: {np.mean(cand_returns):.3f}")

    # 方策モード学習
    print(f"Training policy for {config.policy_epochs} epochs...")
    config.action_mode = "policy"
    # 学習中の efe は entropy（重い KL は使わない）
    config.efe_mode = "entropy"
    # pol_step は方策ネットだけを更新する。世界モデル側のパラメータには勾配を流さない。
    pol_opt, pol_step = make_policy_step(config)
    pol_opt_state = pol_opt.init(eqx.filter(agent, eqx.is_inexact_array))
    for epoch in range(config.policy_epochs):
        batch = buffer.sample_paths(config.batch_size, config.seq_len, rng)
        batch = batch_to_jax(batch)
        key, k_state, k_step = jax.random.split(key, 3)
        # 系列を世界モデルで通し、末尾の事後状態を想像の開始点にする。
        starts = extract_start_states(agent, batch, k_state)
        agent, pol_opt_state, loss, metrics = pol_step(
            agent, pol_opt_state, starts, k_step
        )
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"  epoch {epoch+1:03d} "
                f"loss={float(metrics['policy/loss']):.4f} "
                f"efe={float(metrics['policy/efe']):.4f}"
            )

    # 方策モード評価。collect_episode は候補モードと同じ関数で、
    # action_mode が "policy" なので select_action が方策ネットから行動を引く。
    print("Evaluating policy mode...")
    config.action_mode = "policy"
    pol_returns = []
    for i in range(10):
        key, k_ep = jax.random.split(key)
        ep, ret, key = collect_episode(env, agent, config, k_ep, random=False)
        pol_returns.append(ret)
    print(f"  policy mean return: {np.mean(pol_returns):.3f}")
    print(f"  random mean return: {np.mean(returns):.3f}")
    return agent


def main():
    """コマンドラインから train を起動する。

    既定は離散の3値行動。--continuous を付けると、行動が速度の連続値になる。
    どちらの場合も train() の5段階は同じ順序で走る。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--continuous", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    config = Config(seed=args.seed)
    train(config, discrete=not args.continuous)


if __name__ == "__main__":
    main()
