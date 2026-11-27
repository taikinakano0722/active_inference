"""世界モデルと方策をまとめるエージェント。

train.py が呼ぶ処理は、このファイルに関数として並んでいる。

- 学習（世界モデル）: world_model_loss を make_wm_step が1回更新する。
  実データの系列を事後状態にし、観測の再構成・報酬の予測・事前と事後の KL を足す。
- 学習（方策）: policy_loss を make_policy_step が1回更新する。
  事後状態から未来を想像し、期待自由エネルギーが小さくなるよう方策だけを動かす。
- 環境との1ステップ: get_posterior_state で観測を状態に畳み、select_action で1手出す。
  候補モードと方策モードの分岐は select_action にある。
  収集ループは step_latent_and_action でこの2つを1つの JIT にする。毎ステップ
  グラフを組み直さないため。

ネットワークの中身（RSSM の prior / posterior、EFE の式）は rssm.py と efe.py にある。
ここはそれらをバッチと時間に広げ、損失と更新にまとめる場所である。
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from .config import Config
from .decoder import ObsDecoder, RewardDecoder
from .efe import compute_efe_from_parts, evaluate_action_sequence
from .encoder import Encoder
from .policy import CandidatePlanner, PolicyNet
from .rssm import RSSM
from .spaces import Space, action_dim, flatten_action


class Agent(eqx.Module):
    """1体のエージェント。学習する部品は次の5つ。

    encoder:        観測辞書 → 1本の埋め込み。報酬は入れない。
    rssm:           埋め込みと行動から潜在状態 {"deter", "stoch"} を進める。
    obs_decoder:    潜在状態 → 観測の分布。世界モデル損失の再構成項。
    reward_decoder: 潜在状態 → 報酬の分布。損失にも、行動選択の実用項にも使う。
    policy:         潜在状態 → 行動分布。方策モードのときだけ行動の出どころになる。

    act_space と obs_space は配列ではないので static。勾配の対象に入らない。
    """

    encoder: Encoder
    rssm: RSSM
    obs_decoder: ObsDecoder
    reward_decoder: RewardDecoder
    policy: PolicyNet
    act_space: dict[str, Space] = eqx.field(static=True)
    obs_space: dict[str, Space] = eqx.field(static=True)

    @classmethod
    def create(
        cls,
        obs_space: dict[str, Space],
        act_space: dict[str, Space],
        config: Config,
        key: jax.Array,
    ) -> "Agent":
        """空間の形と Config から、未学習のエージェントを組み立てる。

        次元の依存順に作る。Encoder の出力幅が RSSM の入力になり、
        RSSM の feature_dim（deter と stoch を結合した長さ）が
        両デコーダと方策ネットの入力になる。乱数キーは部品ごとに分ける。
        """
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        encoder = Encoder(
            obs_space,
            vec_units=config.vec_units,
            vec_layers=config.vec_layers,
            img_depths=config.img_depths,
            img_kernel=config.img_kernel,
            key=k1,
        )
        adim = action_dim(act_space)
        rssm = RSSM(
            action_dim=adim,
            embed_dim=encoder.embed_dim,
            deter_dim=config.deter_dim,
            stoch_dim=config.stoch_dim,
            hidden_dim=config.hidden_dim,
            stoch_type=config.stoch_type,
            stoch_classes=config.stoch_classes,
            key=k2,
        )
        feat_dim = rssm.feature_dim()
        obs_decoder = ObsDecoder(
            obs_space,
            feat_dim=feat_dim,
            vec_units=config.vec_units,
            vec_layers=config.vec_layers,
            img_depths=config.img_depths,
            img_kernel=config.img_kernel,
            key=k3,
        )
        reward_decoder = RewardDecoder(
            feat_dim, hidden_dim=config.hidden_dim, layers=2, key=k4
        )
        policy = PolicyNet(
            feat_dim,
            act_space,
            hidden_dim=config.hidden_dim,
            layers=3,
            min_std=config.policy_min_std,
            max_std=config.policy_max_std,
            unimix=config.policy_unimix,
            outscale=config.policy_outscale,
            key=k5,
        )
        return cls(
            encoder=encoder,
            rssm=rssm,
            obs_decoder=obs_decoder,
            reward_decoder=reward_decoder,
            policy=policy,
            act_space=act_space,
            obs_space=obs_space,
        )


def _batch_encode(encoder: Encoder, obs: dict[str, jnp.ndarray]) -> jnp.ndarray:
    """obs 各値は (B, T, ...)。返り値 (B, T, embed)。

    Encoder は観測1件用なので、バッチ B と時刻 T の両方に vmap を掛ける。
    時刻を先頭に移してから、外側を時間、内側をバッチにしてエンコードし、
    (B, T, E) に戻す。
    """
    # obs_t: 各キー (T, B, ...)
    obs_t = {k: jnp.swapaxes(v, 0, 1) for k, v in obs.items()}

    def encode_time(obs_b):
        # obs_b: 各キー (B, ...)
        return jax.vmap(encoder)(obs_b)

    encoded = jax.vmap(encode_time)(obs_t)  # (T, B, E)
    return jnp.swapaxes(encoded, 0, 1)


def _actions_to_array(
    act: dict[str, jnp.ndarray], act_space: dict[str, Space]
) -> jnp.ndarray:
    """(B, T, ...) の行動辞書 → (B, T, action_dim)。

    RSSM は行動を1本のベクトルとして受け取る。キーが複数でも、
    ソート順に平坦化して結合すれば、どの呼び出しでも同じ並びになる。
    離散行動は one-hot のまま連結する。
    """
    parts = []
    for key in sorted(act_space.keys()):
        a = act[key].astype(jnp.float32)
        # (B, T, *shape) → (B, T, flat)
        flat = a.reshape(a.shape[0], a.shape[1], -1)
        parts.append(flat)
    return jnp.concatenate(parts, axis=-1)


def _world_model_forward(
    agent: Agent,
    obs: dict[str, jnp.ndarray],
    act: dict[str, jnp.ndarray],
    rew: jnp.ndarray,
    reset: jnp.ndarray,
    config: Config,
    key: jax.Array,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray], dict[str, jnp.ndarray]]:
    """実系列を1回 observe し、損失と各時刻の outs を返す。

    損失は3項の和。
      recon: 事後状態から観測を再構成した対数尤度の符号反転。
      reward: 同じ状態から報酬を予測した対数尤度の符号反転。
      kl: 事前側（dyn）と事後側（rep）で勾配を分けた KL を、それぞれの係数で足したもの。
    outs の末尾は、方策学習の想像の開始状態に使える。
    """
    B = rew.shape[0]
    # 行動後の観測は obs[:, 1:]
    # 先頭の obs[:, 0] は行動前の観測なので、再構成の対象にはしない。
    obs_post = {k: v[:, 1:] for k, v in obs.items()}
    embeds = _batch_encode(agent.encoder, obs_post)  # (B, T, E)
    actions = _actions_to_array(act, agent.act_space)  # (B, T, A)

    init = agent.rssm.initial()
    # バッチ分に拡張
    init_batch = jax.tree.map(lambda x: jnp.broadcast_to(x, (B,) + x.shape), init)
    keys = jax.random.split(key, B)

    def observe_one(init_state, actions_t, embeds_t, resets_t, k):
        # 時間軸先頭へ
        # observe が時刻順に posterior を進め、各時刻の deter / stoch / KL を返す。
        # reset が True の時刻は、その直前で潜在状態をゼロに戻す。
        return agent.rssm.observe(init_state, actions_t, embeds_t, resets_t, k)

    # observe は系列1本用。vmap で B 本を同時に進める。
    _finals, outs = jax.vmap(observe_one)(
        init_batch, actions, embeds, reset, keys
    )
    # outs: 各値 (B, T, ...)
    # kl_dyn は事後を止めた KL、kl_rep は事前を止めた KL。順伝播の値は同じ。
    # free_nats 未満の時刻は定数になり、その項から勾配は出ない。
    kl_dyn = jnp.maximum(outs["kl_dyn"], config.free_nats)
    kl_rep = jnp.maximum(outs["kl_rep"], config.free_nats)
    kl_loss = config.kl_scale * (
        config.dyn_scale * kl_dyn + config.rep_scale * kl_rep
    )

    def decode_loss_t(deter, stoch, obs_t, rew_t):
        # 1時刻分。潜在状態を1本の特徴にし、観測と報酬の負の対数尤度を返す。
        state = {"deter": deter, "stoch": stoch}
        feat = agent.rssm.feature(state)
        recon = -agent.obs_decoder.log_prob(feat, obs_t)
        rew_lp = -agent.reward_decoder(feat).log_prob(rew_t)
        return recon, rew_lp

    def decode_batch(deter_seq, stoch_seq, obs_seq, rew_seq):
        # 各配列は (T, ...)。vmap が時間を1時刻ずつ decode_loss_t に渡す。
        return jax.vmap(decode_loss_t)(deter_seq, stoch_seq, obs_seq, rew_seq)

    # outs deter/stoch: (B, T, ...)
    # obs_post: (B, T, ...)
    # 内側の vmap が時間、外側の vmap がバッチ。
    recon_losses, rew_losses = jax.vmap(decode_batch)(
        outs["deter"],
        outs["stoch"],
        obs_post,
        rew,
    )
    recon_loss = recon_losses.mean()
    rew_loss = rew_losses.mean()
    kl_mean = kl_loss.mean()
    total = recon_loss + rew_loss + kl_mean
    metrics = {
        "loss/total": total,
        "loss/recon": recon_loss,
        "loss/reward": rew_loss,
        "loss/kl": kl_mean,
        "loss/dyn": (config.kl_scale * config.dyn_scale * kl_dyn).mean(),
        "loss/rep": (config.kl_scale * config.rep_scale * kl_rep).mean(),
        "state/prior_entropy": outs["prior_entropy"].mean(),
        "state/post_entropy": outs["post_entropy"].mean(),
    }
    return total, metrics, outs


def world_model_loss(
    agent: Agent,
    obs: dict[str, jnp.ndarray],
    act: dict[str, jnp.ndarray],
    rew: jnp.ndarray,
    reset: jnp.ndarray,
    config: Config,
    key: jax.Array,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """実データの系列に対する世界モデルの損失。小さいほどよい。

    obs: (B, T+1, ...)  reset 観測を含む
    act: (B, T, ...)
    rew: (B, T)
    reset: (B, T)  各ステップ開始時に状態をリセットするか

    この和が、変分自由エネルギーに相当する学習目標である。
    """
    total, metrics, _outs = _world_model_forward(
        agent, obs, act, rew, reset, config, key
    )
    return total, metrics


def world_model_loss_and_starts(
    agent: Agent,
    obs: dict[str, jnp.ndarray],
    act: dict[str, jnp.ndarray],
    rew: jnp.ndarray,
    reset: jnp.ndarray,
    config: Config,
    key: jax.Array,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray], dict[str, jnp.ndarray]]:
    """世界モデル損失と、同じ observe の末尾事後状態を返す。

    末尾は stop_gradient 済み。方策の想像の開始点に使い、世界モデルへ
    方策勾配を流さない。
    """
    total, metrics, outs = _world_model_forward(
        agent, obs, act, rew, reset, config, key
    )
    starts = {
        "deter": jax.lax.stop_gradient(outs["deter"][:, -1]),
        "stoch": jax.lax.stop_gradient(outs["stoch"][:, -1]),
    }
    return total, metrics, starts


def policy_loss(
    agent: Agent,
    start_states: dict[str, jnp.ndarray],
    config: Config,
    key: jax.Array,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """想像した未来の期待自由エネルギーを、方策の損失にする。

    start_states: 各値 (B, ...)
    想像ホライズンで EFE 和を小さくする。

    各開始状態から、観測は使わずに次を horizon 回繰り返す。
      1. 方策が今の潜在状態から行動を1つサンプルする。
      2. RSSM の prior がその行動で状態を1歩進める。
      3. 進んだ状態の予測報酬と事前エントロピーから、その時刻の EFE を出す。
    損失に入れるのは、この EFE をホライズン分足した値。

    ここは常に epistemic_kl=None で EFE を計算する。efe_mode が "kl" のときは
    ここで止める。想像中に KL を毎回入れると重いため、kl は候補モード向け。
    """
    if config.efe_mode == "kl":
        raise ValueError(
            "policy_loss は想像の各歩で KL を計算しないため、"
            "efe_mode は 'entropy' にしてください。"
            "kl は候補モードの行動選択で使います。"
        )
    B = start_states["deter"].shape[0]
    keys = jax.random.split(key, B)

    def one(state, k):
        k, k_roll = jax.random.split(k)

        def step(carry, _):
            st, key = carry
            key, k_act, k_prior = jax.random.split(key, 3)
            feat = agent.rssm.feature(st)
            dist = agent.policy(feat)
            action = dist.sample(seed=k_act)
            logp = dist.log_prob(action)
            ent = dist.entropy()
            # 想像なので posterior ではなく prior。観測の埋め込みは渡さない。
            st2, prior_dist = agent.rssm.prior(st, action, k_prior)
            feat2 = agent.rssm.feature(st2)
            rew_mean = agent.reward_decoder(feat2).mean()
            efe = compute_efe_from_parts(
                rew_mean, prior_dist.entropy(), None, config
            )
            return (st2, key), (efe, logp, ent)

        # scan の入力系列は無い。length=horizon の回数だけ step を回す。
        (_, _), (efes, logps, ents) = jax.lax.scan(
            step, (state, k_roll), None, length=config.horizon
        )
        total_efe = jnp.sum(efes)
        # REINFORCE 風: EFE * log_prob。離散はこれ、連続も同じ形で可。
        # ただし連続で reparameterize したい場合は efe を直接微分する。
        # 離散のサンプルは行動へ勾配が通らない。EFE を定数にして対数確率へ掛ける。
        # EFE が大きい行動ほど log_prob を下げる勾配が強くなる。
        if agent.policy.discrete:
            loss = jnp.sum(jax.lax.stop_gradient(efes) * logps)
        else:
            # 連続のサンプルは mean + std * noise なので、EFE から行動を経由して
            # 方策の平均と分散へ勾配が戻る。
            loss = total_efe
        # エントロピー項は行動分布を少し広げ、同じ行動ばかり出すのを抑える。
        loss = loss - config.actor_entropy * jnp.sum(ents)
        return loss, total_efe

    losses, efes = jax.vmap(one)(start_states, keys)
    loss = losses.mean()
    metrics = {
        "policy/loss": loss,
        "policy/efe": efes.mean(),
    }
    return loss, metrics


def make_wm_step(config: Config):
    """世界モデル用の最適化器と、1回分の更新関数を返す。

    更新関数 step(agent, opt_state, batch, key) は、world_model_loss の
    勾配をエージェントへ適用して新しい agent を返す。損失が使うのは
    エンコーダ、RSSM、二つのデコーダだけなので、方策ネットの勾配はゼロのまま残る。
    """
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.grad_clip),
        optax.adam(config.lr),
    )

    def loss_fn(agent, batch, key):
        return world_model_loss(
            agent,
            batch["obs"],
            batch["act"],
            batch["rew"],
            batch["reset"],
            config,
            key,
        )

    @eqx.filter_jit
    def step(agent, opt_state, batch, key):
        # has_aux=True なので、勾配は total だけから作り、metrics は添えるだけ。
        (loss, metrics), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
            agent, batch, key
        )
        updates, opt_state = optimizer.update(
            grads, opt_state, eqx.filter(agent, eqx.is_inexact_array)
        )
        agent = eqx.apply_updates(agent, updates)
        return agent, opt_state, loss, metrics

    return optimizer, step


def make_policy_step(config: Config):
    """方策用の最適化器と、1回分の更新関数を返す。

    step(agent, opt_state, start_states, key) の勾配は policy の配列だけに残る。
    世界モデルの部品は stop_gradient 済みのコピーで損失を計算する。
    連続行動のとき、想像の中の行動から EFE へ戻る勾配は方策まで届く。
    その途中の RSSM と報酬デコーダの重みは更新されない。
    """
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.grad_clip),
        optax.adam(config.policy_lr),
    )

    @eqx.filter_jit
    def step(agent, opt_state, start_states, key):
        # 進め方では方策だけ更新する。世界モデル側へは勾配を流さない。
        def wrapped(agent_):
            return policy_loss(
                _stop_world_model(agent_), start_states, config, key
            )

        (loss, metrics), grads = eqx.filter_value_and_grad(wrapped, has_aux=True)(
            agent
        )
        updates, opt_state = optimizer.update(
            grads, opt_state, eqx.filter(agent, eqx.is_inexact_array)
        )
        agent = eqx.apply_updates(agent, updates)
        return agent, opt_state, loss, metrics

    return optimizer, step


def _stop_world_model(agent: Agent) -> Agent:
    """方策損失用に、世界モデル側の配列の勾配を止めたコピーを返す。"""
    return eqx.tree_at(
        lambda a: (
            a.encoder,
            a.rssm,
            a.obs_decoder,
            a.reward_decoder,
        ),
        agent,
        (
            jax.tree.map(jax.lax.stop_gradient, agent.encoder),
            jax.tree.map(jax.lax.stop_gradient, agent.rssm),
            jax.tree.map(jax.lax.stop_gradient, agent.obs_decoder),
            jax.tree.map(jax.lax.stop_gradient, agent.reward_decoder),
        ),
    )


def make_joint_step(config: Config):
    """世界モデルと方策を、同じバッチ・1回の observe で更新する関数を返す。

    step(agent, wm_opt_state, pol_opt_state, batch, key) は次を1つの JIT で行う。
      1. 系列を observe し、世界モデル損失と末尾の事後状態を得る。
      2. 世界モデルを Adam で1回更新する。
      3. その末尾状態から想像し、方策だけを Adam で1回更新する。
    世界モデルと方策のオプティマイザ状態は別々に持つ。
    """
    wm_optimizer = optax.chain(
        optax.clip_by_global_norm(config.grad_clip),
        optax.adam(config.lr),
    )
    pol_optimizer = optax.chain(
        optax.clip_by_global_norm(config.grad_clip),
        optax.adam(config.policy_lr),
    )

    @eqx.filter_jit
    def step(agent, wm_opt_state, pol_opt_state, batch, key):
        k_wm, k_pol = jax.random.split(key)

        def wm_fn(agent_):
            loss, metrics, starts = world_model_loss_and_starts(
                agent_,
                batch["obs"],
                batch["act"],
                batch["rew"],
                batch["reset"],
                config,
                k_wm,
            )
            # has_aux は (loss, aux) の2要素。metrics と starts は1つの aux にまとめる。
            return loss, (metrics, starts)

        (wm_loss, (wm_metrics, starts)), wm_grads = eqx.filter_value_and_grad(
            wm_fn, has_aux=True
        )(agent)
        wm_updates, wm_opt_state = wm_optimizer.update(
            wm_grads, wm_opt_state, eqx.filter(agent, eqx.is_inexact_array)
        )
        agent = eqx.apply_updates(agent, wm_updates)

        def pol_fn(agent_):
            return policy_loss(_stop_world_model(agent_), starts, config, k_pol)

        (pol_loss, pol_metrics), pol_grads = eqx.filter_value_and_grad(
            pol_fn, has_aux=True
        )(agent)
        pol_updates, pol_opt_state = pol_optimizer.update(
            pol_grads, pol_opt_state, eqx.filter(agent, eqx.is_inexact_array)
        )
        agent = eqx.apply_updates(agent, pol_updates)
        metrics = {
            **wm_metrics,
            **pol_metrics,
            "loss/joint": wm_loss + pol_loss,
        }
        return agent, wm_opt_state, pol_opt_state, wm_loss + pol_loss, metrics

    return wm_optimizer, pol_optimizer, step


def get_posterior_state(
    agent: Agent,
    prev_state: dict[str, jnp.ndarray],
    action: jnp.ndarray,
    obs: dict[str, jnp.ndarray],
    key: jax.Array,
    is_first: bool | jnp.ndarray = False,
) -> dict[str, jnp.ndarray]:
    """環境ループの1ステップ分。直前の状態・行動・今の観測から事後状態を返す。

    学習時の observe は系列をまとめて進めるが、行動中は観測が1件ずつ届く。
    ここで posterior を1回だけ呼ぶ。is_first のときは渡された状態と行動を捨て、
    ゼロ状態と零行動から始める。エピソード先頭には履歴が無いため。

    is_first は bool か、スカラーの jnp.bool_。JIT の中では jnp.where で分岐し、
    先頭とそれ以外で別コンパイルにならないようにする。
    """
    is_first = jnp.asarray(is_first, dtype=bool)
    zero_state = agent.rssm.initial()
    prev_state = jax.tree.map(
        lambda s, z: jnp.where(is_first, z, s), prev_state, zero_state
    )
    action = jnp.where(is_first, jnp.zeros_like(action), action)
    embed = agent.encoder(obs)
    post, _, _ = agent.rssm.posterior(prev_state, action, embed, key)
    return post


class _ActionSettings:
    """行動選択が読む設定だけを持つ。

    Config は dataclass なので JAX が pytree とみなし、extras の dict で
    JIT のトレースが止まる。ここは pytree にしない。
    """

    def __init__(
        self,
        action_mode: str,
        efe_mode: str,
        horizon: int,
        candidate_count: int,
        epistemic_weight: float,
        pragmatic_weight: float,
    ):
        self.action_mode = action_mode
        self.efe_mode = efe_mode
        self.horizon = horizon
        self.candidate_count = candidate_count
        self.epistemic_weight = epistemic_weight
        self.pragmatic_weight = pragmatic_weight


def _action_vector(
    agent: Agent,
    state: dict[str, jnp.ndarray],
    config: Config | _ActionSettings,
    key: jax.Array,
) -> tuple[jnp.ndarray, dict[str, Any]]:
    """事後状態から行動ベクトルを1つ返す。numpy には変換しない。"""
    if config.action_mode == "candidate":
        planner = CandidatePlanner()
        action_vec, info = planner.plan(
            agent.rssm,
            agent.reward_decoder,
            agent.encoder,
            agent.obs_decoder,
            state,
            agent.act_space,
            config,
            key,
        )
        return action_vec.astype(jnp.float32), info

    feat = agent.rssm.feature(state)
    dist = agent.policy(feat)
    action = dist.sample(seed=key)
    return action.astype(jnp.float32), {"efe": None}


def select_action(
    agent: Agent,
    state: dict[str, jnp.ndarray],
    config: Config,
    key: jax.Array,
) -> tuple[dict[str, jnp.ndarray], dict[str, Any]]:
    """現在の事後状態から行動を1つ選ぶ。

    config.action_mode で出どころが分かれる。評価に使う EFE の関数は同じで、
    行動列を誰が作るかだけが違う。
    - "candidate": 行動空間から horizon 長の列を作り、EFE の和が最小の列の先頭を返す。
      方策ネットの重みは見ない。
    - "policy": 方策ネットの分布から行動を1つサンプルする。想像はここでは回さない。
    返り値は環境が受け取る行動辞書。最初のモデルは行動キーが1つ。
    収集ループは、この関数ではなく step_latent_and_action を使う。
    """
    act_key = next(iter(agent.act_space.keys()))
    action_vec, info = _action_vector(agent, state, config, key)
    return {act_key: np_from_jax(action_vec)}, info


@eqx.filter_jit
def step_latent_and_action(
    agent: Agent,
    prev_state: dict[str, jnp.ndarray],
    prev_action: jnp.ndarray,
    obs: dict[str, jnp.ndarray],
    key_post: jax.Array,
    key_act: jax.Array,
    is_first: jnp.ndarray,
    action_mode: str,
    efe_mode: str,
    horizon: int,
    candidate_count: int,
    epistemic_weight: float,
    pragmatic_weight: float,
) -> tuple[dict[str, jnp.ndarray], dict[str, jnp.ndarray]]:
    """観測1件から事後状態と行動を返す。収集ループの1手。

    モード、ホライズン、重みはコンパイルのキーに入る。is_first はスカラーの
    jnp.bool_ で、jnp.where で分岐するため、先頭とそれ以外で別コンパイルに
    ならない。重みが変わったあとのエージェントは、同じコンパイル結果を使う。
    """
    state = get_posterior_state(
        agent, prev_state, prev_action, obs, key_post, is_first=is_first
    )
    settings = _ActionSettings(
        action_mode=action_mode,
        efe_mode=efe_mode,
        horizon=int(horizon),
        candidate_count=int(candidate_count),
        epistemic_weight=float(epistemic_weight),
        pragmatic_weight=float(pragmatic_weight),
    )
    action_vec, _ = _action_vector(agent, state, settings, key_act)
    act_key = next(iter(agent.act_space.keys()))
    return state, {act_key: action_vec}


def np_from_jax(x: jnp.ndarray):
    """JAX 配列を numpy にする。環境の step は numpy を受け取る。"""
    import numpy as np

    return np.asarray(x)
