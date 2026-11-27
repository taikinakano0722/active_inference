# リファクタリング計画書

レビューで決めた内容を、ファイルごとに並べる。各項目は指摘 ID と対応する。
済み印 `[x]` は、このチャットの中ですでに入れた修正。

---

## 1. config.py

### 1-1. `train_steps` を削除する（R3）
- L73 の `train_steps: int = 200` を削除する。
- コード上に参照箇所はない。

### 1-2. `gamma` にコメントを付ける（L5）
- L51 の `gamma: float = 0.99` は残す。
- 直後に「※ 現在は EFE の和に掛かっていない。将来割引を入れるときの予約。」と書く。

### 1-3. `replay_mode` と `stream_capacity` を追加する（L1）
- `replay_mode: str = "episode"` を追加する。値は `"episode"` か `"stream"`。
- `stream_capacity: int = 100_000` を追加する。`stream` のときだけ使う。

---

## 2. replay_buffer.py

### 2-1. `episode` の reset ロジックを修正する（L1）
- 切り出しの先頭は常に `reset=True` にする。開始位置に関係なく先頭1歩は開始扱い。
- 途中に done がある場合のコメントと未使用の `dones` 配列を削除する。
- 1件の途中に `done=True` が最後以外にあったら `ValueError` にする assert を入れる。

### 2-2. `StreamReplayBuffer` を追加する（L1）
- 1本の列にステップを足すバッファ。
- `add_step(obs, act, rew, done, is_first)` で1ステップずつ追加する。
  - `is_first=True` のとき、前のステップとの接続が途切れたことを記録する。
- `stream_capacity` を超えた分は、一番古いステップから1つずつ捨てる。
- `sample_paths(n_paths, n_steps, rng)` は既存と同じ戻り値の形。
  - 切り出しの先頭は常に `reset=True`。
  - 切り出しの途中で `is_first=True` のステップがあれば、その位置も `reset=True`。
- `can_sample(n_steps)` は、列の長さが `n_steps` 以上なら真。

---

## 3. agent.py

### 3-1. [x] デッドコードの削除（R1 — 済み）
- `_batch_encode` 内の `encode_bt` と `encode_one_time` は削除済み。

### 3-2. `_batch_encode` を転置 → vmap に書き換える（P1）
- 現在: `jax.vmap(at_t)(jnp.arange(T))` で動的インデックス。
- 変更後: obs を `{k: jnp.swapaxes(v, 0, 1)}` で `(T, B, ...)` にし、外側 vmap を時間、内側 vmap を B にする。
- 戻す: `jnp.swapaxes(result, 0, 1)` で `(B, T, E)`。
- 損失の値は変わらない。

### 3-3. 再構成損失の vmap を書き換える（P2）
- 現在: `decode_path` 内で `jax.vmap(body)(jnp.arange(T))` で動的インデックス。
- 変更後: `jax.vmap(decode_loss_t)` を直接、時間軸に当てる。
  - `decode_loss_t(deter, stoch, obs_t, rew_t)` の引数は1時刻分のまま。
  - 外側 vmap がバッチ、内側 vmap が時間。obs の辞書は各キーを `(T, ...)` にしてから渡す。
- `decode_path` と `body` は不要になる。
- 損失の値は変わらない。

### 3-4. [x] `policy_loss` の efe_mode チェック（L4 — 済み）
- `efe_mode == "kl"` のとき `ValueError` を出す検査を追加済み。

---

## 4. rssm.py

### 4-1. `observe` のゼロ状態を scan の外へ出す（P3）
- 現在: scan の step 関数内で毎回 `self.initial()` を呼ぶ。
- 変更後: `observe` メソッドの先頭で `zero_state = self.initial()` を作り、step の closure で参照する。
- 損失の値は変わらない。

### 4-2. `posterior` の fold_in にコメントを追加する（L7）
- L170 の `key2 = jax.random.fold_in(key, 1)` の前に、
  「prior は deter 更新のみで key 不使用。fold_in で事後専用キーを作る。」と書く。

---

## 5. efe.py

### 5-1. `_kl_from_imagined_obs` の key 引数を削除する（L6）
- 関数シグネチャから `key: jax.Array` を外す。
- 呼び出し元の `evaluate_action_sequence` scan 内で `k_use` を渡していた箇所を修正する。

---

## 6. train.py

### 6-1. `update_world_model` と `update_policy` を削除する（R2）
- L379〜L460 の2関数を削除する。
- `extract_start_states` はベンチスクリプト（`bench_update_breakdown.py`）が使っているので残す。
- import の整理は不要（外部から import している箇所はない）。

### 6-2. `replay_mode` に応じたバッファ選択を `train()` に入れる（L1）
- `config.replay_mode == "episode"` なら今の `ReplayBuffer`。
- `config.replay_mode == "stream"` なら新しい `StreamReplayBuffer`。
- `_validate_train_config` に `replay_mode` の検査を追加する。
- 収集の `buffer.add_episode(ep)` を、`stream` のときはステップごとの `add_step` に切り替える。
- 学習の `buffer.sample_paths(...)` は、両方のバッファで同じ名前・同じ戻り値の形にする。

---

## 7. テスト

### 7-1. 既存テストの修正
- `test_train_loop.py`: `update_world_model` / `update_policy` を import しているテストがあれば外す（現在は無い）。
- `test_agent.py`: 損失の値が変わらないことを、修正前と同じ seed で確認する。

### 7-2. 新規テスト
- `test_replay_buffer.py` に以下を追加する。
  - 切り出しの先頭が常に `reset=True` になること（episode）。
  - 途中に done があったときエラーになること（episode）。
  - `StreamReplayBuffer` で、容量を超えた分が先頭から消えること。
  - `StreamReplayBuffer` で、エピソード境界の次のステップが `reset=True` になること。
  - 切り出しの先頭が常に `reset=True` になること（stream）。
- `test_train_loop.py` に以下を追加する。
  - `replay_mode="stream"` で `train()` が動くこと。

---

## 実装の順番

依存が浅いものから入れる。各ステップのあとにテストを通す。

1. **config.py** — `train_steps` 削除、`gamma` コメント、`replay_mode` / `stream_capacity` 追加
2. **rssm.py** — ゼロ状態を外へ出す（P3）、`fold_in` コメント（L7）
3. **efe.py** — `_kl_from_imagined_obs` の key 引数削除（L6）
4. **agent.py** — `_batch_encode` 書き換え（P1）、再構成損失書き換え（P2）
5. **replay_buffer.py** — `episode` の reset 修正、`StreamReplayBuffer` 追加
6. **train.py** — 未使用関数削除（R2）、`replay_mode` に応じたバッファ選択
7. **テスト** — 既存テスト確認、新規テスト追加
