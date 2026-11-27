"""方策モードの更新1回を区間に分けて測る。

Point1D の幅（deter 64、系列長 20、バッチ 16）。学習ループは変えない。
融合した更新は、この計測の中だけに置く。
"""

from __future__ import annotations

import json
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from active_inference.agent import (
    Agent,
    _actions_to_array,
    _batch_encode,
    make_policy_step,
    make_wm_step,
    policy_loss,
    world_model_loss,
)
from active_inference.config import Config
from active_inference.replay_buffer import ReplayBuffer
from active_inference.spaces import Space
from active_inference.train import batch_to_jax, extract_start_states


N_WARM = 5
N_ITER = 20
N_QUEUE = 20
SEQ_LEN = 20
BATCH = 16
EP_LEN = 40
N_EP = 8


def _stats(samples_s: list[float]) -> dict[str, float]:
    xs = np.asarray(samples_s, dtype=np.float64) * 1e3
    return {
        "median_ms": float(np.median(xs)),
        "mean_ms": float(xs.mean()),
        "std_ms": float(xs.std()),
        "min_ms": float(xs.min()),
        "p90_ms": float(np.percentile(xs, 90)),
    }


def _time_calls(fn, n_warm: int, n_iter: int) -> tuple[float, dict[str, float]]:
    t0 = time.perf_counter()
    jax.block_until_ready(fn())
    compile_ms = (time.perf_counter() - t0) * 1e3
    for _ in range(n_warm):
        jax.block_until_ready(fn())
    samples = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        jax.block_until_ready(fn())
        samples.append(time.perf_counter() - t0)
    return compile_ms, _stats(samples)


def _cpu_calls(fn, n_warm: int, n_iter: int) -> dict[str, float]:
    for _ in range(n_warm):
        fn()
    samples = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return _stats(samples)


def _episode(rng: np.random.Generator, image_size: int | None) -> dict:
    length = EP_LEN
    obs = {
        "position": [rng.normal(size=(1,)).astype(np.float32) for _ in range(length + 1)],
        "goal": [rng.normal(size=(1,)).astype(np.float32) for _ in range(length + 1)],
    }
    if image_size is not None:
        obs["image"] = [
            rng.integers(0, 256, size=(image_size, image_size, 1), dtype=np.uint8)
            for _ in range(length + 1)
        ]
    acts = []
    for _ in range(length):
        v = np.zeros((3,), dtype=np.float32)
        v[int(rng.integers(0, 3))] = 1.0
        acts.append(v)
    done = [False] * (length - 1) + [True]
    return {
        "obs": obs,
        "act": {"action": acts},
        "rew": [float(rng.random()) for _ in range(length)],
        "done": done,
    }


def _buffer(image_size: int | None) -> ReplayBuffer:
    rng = np.random.default_rng(0)
    buf = ReplayBuffer(capacity=N_EP)
    for _ in range(N_EP):
        buf.add_episode(_episode(rng, image_size))
    return buf


def _point_config() -> Config:
    config = Config(seed=0)
    config.deter_dim = 64
    config.stoch_dim = 16
    config.hidden_dim = 64
    config.vec_units = 32
    config.vec_layers = 2
    config.batch_size = BATCH
    config.seq_len = SEQ_LEN
    config.horizon = 4
    config.efe_mode = "entropy"
    config.action_mode = "policy"
    config.epistemic_weight = 0.01
    return config


def _observe_loss_and_starts(agent, batch, config, key):
    """再構成・報酬・KL と、想像の開始状態を、同じ observe から作る。"""
    obs = batch["obs"]
    act = batch["act"]
    rew = batch["rew"]
    reset = batch["reset"]
    B = rew.shape[0]
    obs_post = {k: v[:, 1:] for k, v in obs.items()}
    embeds = _batch_encode(agent.encoder, obs_post)
    actions = _actions_to_array(act, agent.act_space)
    init = agent.rssm.initial()
    init_batch = jax.tree.map(lambda x: jnp.broadcast_to(x, (B,) + x.shape), init)
    keys = jax.random.split(key, B)

    def observe_one(init_state, actions_t, embeds_t, resets_t, k):
        return agent.rssm.observe(init_state, actions_t, embeds_t, resets_t, k)

    _finals, outs = jax.vmap(observe_one)(init_batch, actions, embeds, reset, keys)
    kl_dyn = jnp.maximum(outs["kl_dyn"], config.free_nats)
    kl_rep = jnp.maximum(outs["kl_rep"], config.free_nats)
    kl_loss = config.kl_scale * (config.dyn_scale * kl_dyn + config.rep_scale * kl_rep)

    def decode_path(deter_seq, stoch_seq, obs_seq, rew_seq):
        def body(t):
            state = {"deter": deter_seq[t], "stoch": stoch_seq[t]}
            feat = agent.rssm.feature(state)
            obs_t = {k: obs_seq[k][t] for k in obs_seq}
            recon = -agent.obs_decoder.log_prob(feat, obs_t)
            rew_lp = -agent.reward_decoder(feat).log_prob(rew_seq[t])
            return recon, rew_lp

        return jax.vmap(body)(jnp.arange(rew_seq.shape[0]))

    recon_losses, rew_losses = jax.vmap(decode_path)(
        outs["deter"], outs["stoch"], obs_post, rew
    )
    total = recon_losses.mean() + rew_losses.mean() + kl_loss.mean()
    starts = {
        "deter": jax.lax.stop_gradient(outs["deter"][:, -1]),
        "stoch": jax.lax.stop_gradient(outs["stoch"][:, -1]),
    }
    return total, starts


def _stop_world(agent):
    return eqx.tree_at(
        lambda a: (a.encoder, a.rssm, a.obs_decoder, a.reward_decoder),
        agent,
        (
            jax.tree.map(jax.lax.stop_gradient, agent.encoder),
            jax.tree.map(jax.lax.stop_gradient, agent.rssm),
            jax.tree.map(jax.lax.stop_gradient, agent.obs_decoder),
            jax.tree.map(jax.lax.stop_gradient, agent.reward_decoder),
        ),
    )


def main() -> None:
    config = _point_config()
    obs_space = {
        "position": Space(np.float32, (1,)),
        "goal": Space(np.float32, (1,)),
    }
    act_space = {"action": Space(np.float32, (3,), discrete=True, n=3)}
    agent = Agent.create(obs_space, act_space, config, jax.random.PRNGKey(0))
    params = eqx.filter(agent, eqx.is_inexact_array)
    wm_opt, wm_step = make_wm_step(config)
    pol_opt, pol_step = make_policy_step(config)
    wm_state = wm_opt.init(params)
    pol_state = pol_opt.init(params)

    vec_buf = _buffer(None)
    img32_buf = _buffer(32)
    img64_buf = _buffer(64)
    rng = np.random.default_rng(1)
    key = jax.random.PRNGKey(1)

    def next_key():
        nonlocal key
        key, k = jax.random.split(key)
        return k

    rows: list[dict] = []

    def add(name: str, stats: dict, note: str, compile_ms: float | None = None) -> None:
        row = {"name": name, "note": note, **stats}
        if compile_ms is not None:
            row["compile_ms"] = compile_ms
        rows.append(row)
        print(
            f"{name:28} median {stats['median_ms']:.3f} ms",
            flush=True,
        )

    print("リプレイと転送", flush=True)
    add(
        "sample_vector",
        _cpu_calls(lambda: vec_buf.sample_paths(BATCH, SEQ_LEN, rng), N_WARM, N_ITER),
        "位置と目標。np.stack だけ",
    )
    add(
        "sample_image32",
        _cpu_calls(lambda: img32_buf.sample_paths(BATCH, SEQ_LEN, rng), N_WARM, N_ITER),
        "32x32x1 を同じ切り方",
    )
    add(
        "sample_image64",
        _cpu_calls(lambda: img64_buf.sample_paths(BATCH, SEQ_LEN, rng), N_WARM, N_ITER),
        "64x64x1 を同じ切り方",
    )

    vec_np = vec_buf.sample_paths(BATCH, SEQ_LEN, rng)
    img32_np = img32_buf.sample_paths(BATCH, SEQ_LEN, rng)
    img64_np = img64_buf.sample_paths(BATCH, SEQ_LEN, rng)

    def h2d(batch):
        out = batch_to_jax(batch)
        jax.block_until_ready(out["rew"])
        return out["rew"]

    for name, batch, note in (
        ("h2d_vector", vec_np, "ベクトルバッチをデバイスへ"),
        ("h2d_image32", img32_np, "32x32 を含むバッチをデバイスへ"),
        ("h2d_image64", img64_np, "64x64 を含むバッチをデバイスへ"),
    ):
        compile_ms, stats = _time_calls(lambda b=batch: h2d(b), N_WARM, N_ITER)
        add(name, stats, note, compile_ms)

    batch = batch_to_jax(vec_np)
    # 以降のデバイス計測は、このベクトルバッチを使い回す。
    # オプティマイザ状態は計測のたびに更新される。時間の比較が目的。

    print("世界モデル", flush=True)

    def wm_wait_loss():
        nonlocal agent, wm_state
        agent, wm_state, loss, _metrics = wm_step(agent, wm_state, batch, next_key())
        jax.block_until_ready(loss)
        return loss

    compile_ms, stats = _time_calls(wm_wait_loss, N_WARM, N_ITER)
    add("wm_step_sync_loss", stats, "JIT 済み更新。loss の完了だけ待つ", compile_ms)

    def wm_float_metrics():
        nonlocal agent, wm_state
        agent, wm_state, loss, metrics = wm_step(agent, wm_state, batch, next_key())
        return {k: float(v) for k, v in metrics.items()}

    t0 = time.perf_counter()
    wm_float_metrics()
    compile_ms = (time.perf_counter() - t0) * 1e3
    for _ in range(1):
        wm_float_metrics()
    samples = []
    for _ in range(N_ITER):
        t0 = time.perf_counter()
        wm_float_metrics()
        samples.append(time.perf_counter() - t0)
    add("wm_step_float_metrics", _stats(samples), "今のループ。metrics を毎回 float にする", compile_ms)

    def wm_queued():
        nonlocal agent, wm_state
        loss = None
        for _ in range(N_QUEUE):
            agent, wm_state, loss, _metrics = wm_step(
                agent, wm_state, batch, next_key()
            )
        jax.block_until_ready(loss)
        return loss

    t0 = time.perf_counter()
    jax.block_until_ready(wm_queued())
    for _ in range(3):
        jax.block_until_ready(wm_queued())
    samples = []
    for _ in range(5):
        t0 = time.perf_counter()
        jax.block_until_ready(wm_queued())
        samples.append((time.perf_counter() - t0) / N_QUEUE)
    add("wm_step_queued", _stats(samples), f"{N_QUEUE} 回積んで最後に1回待つ。1回あたり")

    print("開始状態と方策", flush=True)

    def extract_raw():
        starts = extract_start_states(agent, batch, next_key())
        jax.block_until_ready(starts["deter"])
        return starts["deter"]

    compile_ms, stats = _time_calls(extract_raw, 1, 4)
    add("extract_unjit", stats, "今の extract_start_states。JIT なし", compile_ms)

    @eqx.filter_jit
    def extract_jit(agent_, batch_, key_):
        return extract_start_states(agent_, batch_, key_)

    def extract_compiled():
        starts = extract_jit(agent, batch, next_key())
        jax.block_until_ready(starts["deter"])
        return starts["deter"]

    compile_ms, stats = _time_calls(extract_compiled, N_WARM, N_ITER)
    add("extract_jit", stats, "同じ observe を filter_jit で包む", compile_ms)

    def pol_wait():
        nonlocal agent, pol_state
        starts = extract_jit(agent, batch, next_key())
        agent, pol_state, loss, _metrics = pol_step(
            agent, pol_state, starts, next_key()
        )
        jax.block_until_ready(loss)
        return loss

    # 開始状態の計算は上で別に測る。ここは pol_step だけにしたいので、
    # starts を先に用意してから計る。
    starts0 = extract_jit(agent, batch, next_key())
    jax.block_until_ready(starts0["deter"])

    def pol_only():
        nonlocal agent, pol_state
        agent, pol_state, loss, _metrics = pol_step(
            agent, pol_state, starts0, next_key()
        )
        jax.block_until_ready(loss)
        return loss

    compile_ms, stats = _time_calls(pol_only, N_WARM, N_ITER)
    add("pol_step", stats, "JIT 済みの想像と方策更新。開始状態は固定", compile_ms)

    print("1回の更新", flush=True)

    def current_update():
        nonlocal agent, wm_state, pol_state
        raw1 = vec_buf.sample_paths(BATCH, SEQ_LEN, rng)
        b1 = batch_to_jax(raw1)
        agent, wm_state, _loss, metrics = wm_step(agent, wm_state, b1, next_key())
        _ = {k: float(v) for k, v in metrics.items()}
        raw2 = vec_buf.sample_paths(BATCH, SEQ_LEN, rng)
        b2 = batch_to_jax(raw2)
        starts = extract_start_states(agent, b2, next_key())
        jax.block_until_ready(starts["deter"])
        agent, pol_state, loss, pmetrics = pol_step(
            agent, pol_state, starts, next_key()
        )
        _ = {k: float(v) for k, v in pmetrics.items()}
        return loss

    compile_ms, stats = _time_calls(current_update, 1, 6)
    add("update_current", stats, "今の2段。バッチ2本、float、extract は JIT なし", compile_ms)

    @eqx.filter_jit
    def fused_two(agent_, wm_opt_state, pol_opt_state, batch_, key_):
        """observe は2回。起動と同期は1回。"""
        k_wm, k_ex, k_pol = jax.random.split(key_, 3)

        def wm_fn(agent_in, batch_in, key_in):
            return world_model_loss(
                agent_in,
                batch_in["obs"],
                batch_in["act"],
                batch_in["rew"],
                batch_in["reset"],
                config,
                key_in,
            )

        (wm_loss, _mets), wm_grads = eqx.filter_value_and_grad(wm_fn, has_aux=True)(
            agent_, batch_, k_wm
        )
        wm_updates, wm_opt_state = wm_opt.update(
            wm_grads, wm_opt_state, eqx.filter(agent_, eqx.is_inexact_array)
        )
        agent_ = eqx.apply_updates(agent_, wm_updates)
        starts = extract_start_states(agent_, batch_, k_ex)

        def pol_fn(agent_in, starts_in, key_in):
            return policy_loss(_stop_world(agent_in), starts_in, config, key_in)

        (pol_loss, _pmets), pol_grads = eqx.filter_value_and_grad(pol_fn, has_aux=True)(
            agent_, starts, k_pol
        )
        pol_updates, pol_opt_state = pol_opt.update(
            pol_grads, pol_opt_state, eqx.filter(agent_, eqx.is_inexact_array)
        )
        agent_ = eqx.apply_updates(agent_, pol_updates)
        return agent_, wm_opt_state, pol_opt_state, wm_loss + pol_loss

    @eqx.filter_jit
    def fused_one(agent_, wm_opt_state, pol_opt_state, batch_, key_):
        """observe は1回。末尾状態を想像の開始に使う。"""
        k_wm, k_pol = jax.random.split(key_)

        def wm_fn(agent_in, batch_in, key_in):
            return _observe_loss_and_starts(agent_in, batch_in, config, key_in)

        (wm_loss, starts), wm_grads = eqx.filter_value_and_grad(wm_fn, has_aux=True)(
            agent_, batch_, k_wm
        )
        wm_updates, wm_opt_state = wm_opt.update(
            wm_grads, wm_opt_state, eqx.filter(agent_, eqx.is_inexact_array)
        )
        agent_ = eqx.apply_updates(agent_, wm_updates)

        def pol_fn(agent_in, starts_in, key_in):
            return policy_loss(_stop_world(agent_in), starts_in, config, key_in)

        (pol_loss, _pmets), pol_grads = eqx.filter_value_and_grad(pol_fn, has_aux=True)(
            agent_, starts, k_pol
        )
        pol_updates, pol_opt_state = pol_opt.update(
            pol_grads, pol_opt_state, eqx.filter(agent_, eqx.is_inexact_array)
        )
        agent_ = eqx.apply_updates(agent_, pol_updates)
        return agent_, wm_opt_state, pol_opt_state, wm_loss + pol_loss

    def run_fused(fn):
        nonlocal agent, wm_state, pol_state
        agent, wm_state, pol_state, loss = fn(
            agent, wm_state, pol_state, batch, next_key()
        )
        jax.block_until_ready(loss)
        return loss

    compile_ms, stats = _time_calls(lambda: run_fused(fused_two), N_WARM, N_ITER)
    add("update_fused_two_observe", stats, "1 JIT・1バッチ。observe は世界モデルと開始状態で2回", compile_ms)

    compile_ms, stats = _time_calls(lambda: run_fused(fused_one), N_WARM, N_ITER)
    add("update_fused_one_observe", stats, "1 JIT・1バッチ。observe は1回で、末尾を想像に使う", compile_ms)

    report = {
        "device": str(jax.devices()[0]),
        "jax": jax.__version__,
        "batch": BATCH,
        "seq_len": SEQ_LEN,
        "horizon": config.horizon,
        "rows": rows,
    }
    text = json.dumps(report, indent=2)
    print(text)
    out = "/workspace/models/active_inference/experiments/bench_update_breakdown.json"
    with open(out, "w", encoding="utf-8") as f:
        f.write(text)


if __name__ == "__main__":
    main()
