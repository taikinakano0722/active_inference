"""行動選択1手の時間を、方策・候補列・DreamerV3 で比べる。

Point1D の幅（deter 64、確率状態の平坦次元 16、隠れ 64）に揃える。
DreamerV3 のセルはブロック GRU とカテゴリカルのまま、幅だけ合わせる。
計算 dtype は両方 float32。バッチは 1。観測は position と goal のベクトル。

2種類の時間を取る。
- call: コンパイル済みの関数を Python から1回呼び、行動がホストに戻るまで。
- device: 同じ処理を 100 手、1回の JIT の中で進め、1手あたりに割る。
"""

from __future__ import annotations

import json
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

# Dreamer の計算型は既定が bfloat16。モジュールを読む前に float32 へ変える。
import embodied.jax.nets as nn

nn.COMPUTE_DTYPE = jnp.float32

import elements
import embodied.jax
import ninjax as nj

from dreamerv3 import rssm as dv3_rssm

from active_inference.agent import Agent, get_posterior_state
from active_inference.config import Config
from active_inference.policy import sample_action_sequences
from active_inference.spaces import Space
from active_inference.efe import evaluate_action_sequences


N_STEPS = 100
N_WARM = 10
N_ITER = 50
DETER = 64
STOC_FLAT = 16
HIDDEN = 64
HORIZON = 4
N_CANDIDATES = 81  # 3^4


def _stats(samples_s: list[float]) -> dict[str, float]:
    xs = np.asarray(samples_s, dtype=np.float64) * 1e3
    return {
        "median_ms": float(np.median(xs)),
        "mean_ms": float(xs.mean()),
        "std_ms": float(xs.std()),
        "min_ms": float(xs.min()),
        "p90_ms": float(np.percentile(xs, 90)),
    }


def _measure(fn, n_warm: int, n_iter: int) -> tuple[float, dict[str, float]]:
    t0 = time.perf_counter()
    host = jax.device_get(fn())
    compile_ms = (time.perf_counter() - t0) * 1e3
    del host
    for _ in range(n_warm):
        jax.block_until_ready(fn())
    samples = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        jax.device_get(fn())
        samples.append(time.perf_counter() - t0)
    return compile_ms, _stats(samples)


def _nparams(tree) -> int:
    leaves = jax.tree.leaves(eqx.filter(tree, eqx.is_inexact_array))
    return int(sum(x.size for x in leaves))


def _nj_params(state) -> dict[str, int]:
    counts: dict[str, int] = {}
    leaves, _ = jax.tree_util.tree_flatten_with_path(state)
    for path, val in leaves:
        if not hasattr(val, "size") or not jnp.issubdtype(val.dtype, jnp.inexact):
            continue
        parts = []
        for p in path:
            parts.append(str(getattr(p, "key", p)))
        raw = parts[0] if parts else "other"
        head = raw.split("/")[0]
        counts[head] = counts.get(head, 0) + int(val.size)
    counts["total"] = int(sum(counts.values()))
    return counts


def main() -> None:
    key = jax.random.PRNGKey(0)
    config = Config()
    config.deter_dim = DETER
    config.stoch_dim = STOC_FLAT
    config.hidden_dim = HIDDEN
    config.vec_units = 32
    config.vec_layers = 2
    config.horizon = HORIZON
    config.candidate_count = N_CANDIDATES
    config.efe_mode = "entropy"
    config.epistemic_weight = 0.01
    config.pragmatic_weight = 1.0

    obs_space = {
        "position": Space(np.float32, (1,)),
        "goal": Space(np.float32, (1,)),
    }
    act_space = {
        "action": Space(np.float32, (3,), discrete=True, n=3),
    }
    key, k_agent = jax.random.split(key)
    agent = Agent.create(obs_space, act_space, config, k_agent)
    state0 = agent.rssm.initial()
    prev0 = jnp.zeros((3,), dtype=jnp.float32)
    obs0 = {
        "position": jnp.zeros((1,), dtype=jnp.float32),
        "goal": jnp.ones((1,), dtype=jnp.float32),
    }
    key, k_seq = jax.random.split(key)
    seq_t0 = time.perf_counter()
    action_seqs = sample_action_sequences(act_space, HORIZON, N_CANDIDATES, k_seq)
    action_seqs = jnp.asarray(action_seqs, dtype=jnp.float32)
    jax.block_until_ready(action_seqs)
    seq_build_ms = (time.perf_counter() - seq_t0) * 1e3

    @eqx.filter_jit
    def policy_call(agent, state, prev, obs, key):
        k_post, k_act = jax.random.split(key)
        state = get_posterior_state(
            agent, state, prev, obs, k_post, is_first=False
        )
        action = agent.policy(agent.rssm.feature(state)).sample(seed=k_act)
        return action

    @eqx.filter_jit
    def policy_device(agent, state, prev, obs, key):
        def body(carry, _):
            state, prev, key = carry
            key, k_use = jax.random.split(key)
            k_post, k_act = jax.random.split(k_use)
            state = get_posterior_state(
                agent, state, prev, obs, k_post, is_first=False
            )
            action = agent.policy(agent.rssm.feature(state)).sample(seed=k_act)
            action = action.astype(jnp.float32)
            return (state, action, key), action

        (_, _, _), actions = jax.lax.scan(
            body, (state, prev, key), None, length=N_STEPS
        )
        return actions

    @eqx.filter_jit
    def candidate_call(agent, state, prev, obs, seqs, key):
        k_post, k_eval = jax.random.split(key)
        state = get_posterior_state(
            agent, state, prev, obs, k_post, is_first=False
        )
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
        return seqs[jnp.argmin(efes), 0]

    @eqx.filter_jit
    def candidate_device(agent, state, prev, obs, seqs, key):
        def body(carry, _):
            state, prev, key = carry
            key, k_use = jax.random.split(key)
            k_post, k_eval = jax.random.split(k_use)
            state = get_posterior_state(
                agent, state, prev, obs, k_post, is_first=False
            )
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
            action = seqs[jnp.argmin(efes), 0]
            return (state, action, key), action

        (_, _, _), actions = jax.lax.scan(
            body, (state, prev, key), None, length=N_STEPS
        )
        return actions

    # --- DreamerV3。幅だけ合わせ、セルは元の実装。 ---
    dv3_obs = {
        "position": elements.Space(np.float32, (1,)),
        "goal": elements.Space(np.float32, (1,)),
    }
    dv3_act = {"action": elements.Space(np.int32, (), 0, 3)}

    def dreamer_step(obs, reset, prevact, enc_carry, dyn_carry):
        enc = dv3_rssm.Encoder(
            dv3_obs,
            units=64,
            layers=2,
            act="silu",
            norm="rms",
            symlog=True,
            name="enc",
        )
        dyn = dv3_rssm.RSSM(
            dv3_act,
            deter=DETER,
            hidden=HIDDEN,
            stoch=4,
            classes=4,
            act="silu",
            norm="rms",
            unimix=0.01,
            outscale=1.0,
            imglayers=2,
            obslayers=1,
            dynlayers=1,
            blocks=8,
            name="dyn",
        )
        pol = embodied.jax.MLPHead(
            dv3_act,
            {"action": "categorical"},
            layers=3,
            units=HIDDEN,
            act="silu",
            norm="rms",
            outscale=0.01,
            unimix=0.01,
            minstd=0.1,
            maxstd=1.0,
            name="pol",
        )
        enc_carry, _, tokens = enc(
            enc_carry, obs, reset, training=False, single=True
        )
        dyn_carry, _, feat = dyn.observe(
            dyn_carry, tokens, prevact, reset, training=False, single=True
        )
        flat = jnp.concatenate(
            [
                feat["deter"],
                feat["stoch"].reshape((*feat["stoch"].shape[:-2], -1)),
            ],
            -1,
        )
        act = jax.tree.map(lambda d: d.sample(seed=nj.seed()), pol(flat, bdims=1))
        return enc_carry, dyn_carry, act

    def dreamer_many(obs, reset, prevact, enc_carry, dyn_carry):
        def body(carry, _):
            enc_c, dyn_c, prev = carry
            enc_c, dyn_c, act = dreamer_step(obs, reset, prev, enc_c, dyn_c)
            return (enc_c, dyn_c, act), act["action"]

        _, actions = nj.scan(
            body, (enc_carry, dyn_carry, prevact), (), N_STEPS
        )
        return actions

    pure_step = nj.pure(dreamer_step)
    pure_many = nj.pure(dreamer_many)
    d_obs = {
        "position": jnp.zeros((1, 1), dtype=jnp.float32),
        "goal": jnp.ones((1, 1), dtype=jnp.float32),
    }
    d_reset = jnp.zeros((1,), dtype=bool)
    d_prev = {"action": jnp.zeros((1,), dtype=jnp.int32)}
    d_enc = {}
    d_dyn = {
        "deter": jnp.zeros((1, DETER), dtype=jnp.float32),
        "stoch": jnp.zeros((1, 4, 4), dtype=jnp.float32),
    }
    key, k_dv = jax.random.split(key)
    dv_params = nj.init(dreamer_step)(
        {}, d_obs, d_reset, d_prev, d_enc, d_dyn, seed=k_dv
    )

    @jax.jit
    def dreamer_call(params, seed, obs, reset, prevact, enc_carry, dyn_carry):
        _, (_, _, act) = pure_step(
            params, obs, reset, prevact, enc_carry, dyn_carry, seed=seed
        )
        return act["action"]

    @jax.jit
    def dreamer_device(params, seed, obs, reset, prevact, enc_carry, dyn_carry):
        _, actions = pure_many(
            params, obs, reset, prevact, enc_carry, dyn_carry, seed=seed
        )
        return actions

    def next_key():
        nonlocal key
        key, k = jax.random.split(key)
        return k

    print("計測を始めます。最初の呼び出しでコンパイルします。", flush=True)
    results = []

    def add(name, kind, compile_ms, stats, note):
        row = {"name": name, "kind": kind, "compile_ms": compile_ms, **stats, "note": note}
        results.append(row)
        print(
            f"{name:22} {kind:8} median {stats['median_ms']:.4f} ms"
            f"  compile {compile_ms:.1f} ms",
            flush=True,
        )

    compile_ms, stats = _measure(
        lambda: policy_call(agent, state0, prev0, obs0, next_key()),
        N_WARM,
        N_ITER,
    )
    add("ours_policy", "call", compile_ms, stats, "事後更新と方策サンプル1回")

    compile_ms, stats = _measure(
        lambda: policy_device(agent, state0, prev0, obs0, next_key()),
        N_WARM,
        N_ITER,
    )
    stats = {k: v / N_STEPS for k, v in stats.items()}
    add("ours_policy", "device", compile_ms, stats, f"{N_STEPS}手を1回のJITで割り戻し")

    compile_ms, stats = _measure(
        lambda: candidate_call(
            agent, state0, prev0, obs0, action_seqs, next_key()
        ),
        N_WARM,
        N_ITER,
    )
    add(
        "ours_candidate",
        "call",
        compile_ms,
        stats,
        "事後1回と候補81本・ホライズン4のEFE",
    )

    compile_ms, stats = _measure(
        lambda: candidate_device(
            agent, state0, prev0, obs0, action_seqs, next_key()
        ),
        N_WARM,
        N_ITER,
    )
    stats = {k: v / N_STEPS for k, v in stats.items()}
    add("ours_candidate", "device", compile_ms, stats, f"{N_STEPS}手を1回のJITで割り戻し")

    compile_ms, stats = _measure(
        lambda: dreamer_call(
            dv_params, next_key(), d_obs, d_reset, d_prev, d_enc, d_dyn
        ),
        N_WARM,
        N_ITER,
    )
    add("dreamerv3_policy", "call", compile_ms, stats, "エンコーダ、RSSM 1歩、方策サンプル")

    compile_ms, stats = _measure(
        lambda: dreamer_device(
            dv_params, next_key(), d_obs, d_reset, d_prev, d_enc, d_dyn
        ),
        N_WARM,
        N_ITER,
    )
    stats = {k: v / N_STEPS for k, v in stats.items()}
    add("dreamerv3_policy", "device", compile_ms, stats, f"{N_STEPS}手を1回のJITで割り戻し")

    ours_params = {
        "encoder": _nparams(agent.encoder),
        "rssm": _nparams(agent.rssm),
        "policy": _nparams(agent.policy),
        "reward_decoder": _nparams(agent.reward_decoder),
        "obs_decoder": _nparams(agent.obs_decoder),
    }
    ours_params["policy_path"] = (
        ours_params["encoder"] + ours_params["rssm"] + ours_params["policy"]
    )
    ours_params["candidate_path"] = (
        ours_params["encoder"] + ours_params["rssm"] + ours_params["reward_decoder"]
    )
    dv_counts = _nj_params(dv_params)

    device = jax.devices()[0]
    report = {
        "device": str(device),
        "jax": jax.__version__,
        "dtype": "float32",
        "batch": 1,
        "n_warm": N_WARM,
        "n_iter": N_ITER,
        "n_steps_inside_jit": N_STEPS,
        "widths": {
            "deter": DETER,
            "stoch_flat": STOC_FLAT,
            "hidden": HIDDEN,
            "ours_vec_units": 32,
            "ours_vec_layers": 2,
            "dreamer_stoch": 4,
            "dreamer_classes": 4,
            "dreamer_blocks": 8,
            "dreamer_encoder_units": 64,
            "dreamer_encoder_layers": 2,
            "policy_layers": 3,
        },
        "candidate": {
            "horizon": HORIZON,
            "n_candidates": int(action_seqs.shape[0]),
            "efe_mode": "entropy",
            "sequence_build_ms": seq_build_ms,
        },
        "params": {"ours": ours_params, "dreamerv3": dv_counts},
        "timings": results,
    }
    text = json.dumps(report, indent=2)
    print(text)
    out_path = "/workspace/models/active_inference/experiments/bench_action_select.json"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)


if __name__ == "__main__":
    main()
