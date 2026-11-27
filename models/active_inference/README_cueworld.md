# cueworld subtasks で回すためのチートシート

`../cueworld/subtasks_for_cueworld` の `SanityEnv` と `CommunicationEnv` を、このモデルで学習・評価する手順。別のマシンで clone してコンテナを立てるところは [SETUP_cueworld.md](SETUP_cueworld.md)。

関係するファイル。

| ファイル | 役割 |
|---|---|
| `envs/cueworld_subtasks.py` | Gymnasium の環境を、このモデルの `Env` 契約（観測 `{"obs"}`、行動 `{"action": one-hot}`、4 値の `step`）につなぐ |
| `experiments/cueworld_subtasks.py` | 学習 → 評価 → `results/cueworld/<名前>.json` を書く入口 |
| `experiments/summarize_cueworld.py` | 結果 JSON を表にする |
| `tests/test_cueworld_subtasks.py` | アダプタと、短い学習・評価が通るかのテスト |

環境のコードは subtasks 側が正本。ここには写さず、ファイルを直接読む。

---

## 0. 初回だけ: コンテナの準備

`docker-compose.yml` に `../cueworld:/cueworld:ro` のマウントを足した。`requirements.txt` に `gymnasium==1.2.1` を足した。反映にはコンテナを作り直す。**コンテナ内で動いているものは止まる。**

```bash
cd ~/infant_model/proposed_model
docker compose up -d --build        # イメージを作り直し、マウントを反映
docker exec nakano_infant_project ls /cueworld/subtasks_for_cueworld/envs   # communication.py, sanity.py が見えれば OK
```

作り直したくないときの一時しのぎ。コンテナにコピーし、場所を環境変数で渡す。ホストで subtasks を直しても反映されないので注意。

```bash
docker exec nakano_infant_project pip install gymnasium==1.2.1
docker exec nakano_infant_project mkdir -p /tmp/cw
docker cp ~/infant_model/cueworld/subtasks_for_cueworld nakano_infant_project:/tmp/cw/
# 以降のコマンドに CUEWORLD_SUBTASKS_DIR=/tmp/cw/subtasks_for_cueworld を付ける
```

subtasks の探し方は、`CUEWORLD_SUBTASKS_DIR` → `/cueworld/subtasks_for_cueworld` → ホストの `~/infant_model/cueworld/subtasks_for_cueworld` の順。

## 共通の書き方

以下はすべてホストから打つ形。

```bash
AI='docker exec -it nakano_infant_project bash -lc'
# 例: $AI 'cd /workspace && PYTHONPATH=/workspace/models python -m active_inference.experiments.cueworld_subtasks --env communication'
```

コンテナに入ってから打つなら、`cd /workspace && export PYTHONPATH=/workspace/models` のあと `python -m ...` だけでよい。以下はコンテナ内の形で書く。

## 1. 動作確認

```bash
python -m active_inference.tests.test_cueworld_subtasks      # 40 秒ほど。最後に ok
# 環境側のテスト。このコンテナに pytest は無いので、初回は pip install pytest==8.3.4
# /cueworld は読み取り専用なので、キャッシュを書かないよう -p no:cacheprovider を付ける
cd /cueworld/subtasks_for_cueworld && python -m pytest -q -p no:cacheprovider && cd /workspace
```

## 2. 1 本回す

既定（`max_steps=50`, 3 周, 30+10+10 エピソード, 方策モード）で 1 本 30 秒ほど。

```bash
# Sanity: 次の記号が決まる行動が1つだけある
python -m active_inference.experiments.cueworld_subtasks --env sanity
python -m active_inference.experiments.cueworld_subtasks --env sanity --reward-mode          # 乱数行動の1つに報酬

# Communication
python -m active_inference.experiments.cueworld_subtasks --env communication --dependency short --n-meaningful 1
python -m active_inference.experiments.cueworld_subtasks --env communication --dependency long
python -m active_inference.experiments.cueworld_subtasks --env communication --dependency branch

# yoke（親の列を録画再生。行動と無関係）。学習も評価も yoke になる
python -m active_inference.experiments.cueworld_subtasks --env communication --dependency short --condition yoke
```

ログを残すとき。

```bash
mkdir -p /workspace/logs/cueworld
python -m active_inference.experiments.cueworld_subtasks --env communication 2>&1 | tee /workspace/logs/cueworld/comm_short.log
```

## 3. よく変える引数

| 引数 | 既定 | 意味 |
|---|---|---|
| `--env` | 必須 | `sanity` / `communication` |
| `--dependency` | `short` | communication の `short` / `long` / `branch` |
| `--n-meaningful` | 1 | short で返事が決まる記号の数 |
| `--reward-mode` | off | sanity で報酬を付ける |
| `--obs-noise` | 0 | 観測に足すガウスノイズの標準偏差。学習はノイズありが想定 |
| `--symbol-noise` | 0 | sanity で記号をランダムに置き換える確率 |
| `--max-steps` | 50 | 1 エピソードの長さ。`--seq-len` 以上にする |
| `--condition` | `dependent` | `yoke` で録画再生 |
| `--action-mode` | `policy` | `policy`（方策ネット）/ `candidate`（候補列の全列挙） |
| `--efe-mode` | `entropy` | `kl` は candidate のみ |
| `--epistemic-weight` | 1.0 | **符号に注意。** EFE = −pw·報酬 − ew·情報利得 で、小さい列を選ぶ。正なら不確かな未来（事前エントロピーが大きい）を好み、負なら予測しやすい未来を好む |
| `--pragmatic-weight` | 1.0 | 報酬の重み。communication は報酬 0 なので効かない |
| `--horizon` | 3 | 想像の長さ。candidate は `n_actions^horizon ≤ --candidate-count` なら全列挙 |
| `--rounds` | 3 | 収集と更新の周回数 |
| `--random-episodes` / `--collect-episodes` | 30 / 10 | 1 周目 / 2 周目以降に集める本数 |
| `--random-initial` | off | 1 周目を一様ランダムで集める |
| `--train-ratio` | 32 | 新しい 1 ステップあたり、リプレイから学習に使うステップ数 |
| `--deter-dim` / `--stoch-dim` / `--stoch-type` | 64 / 16 / gaussian | 世界モデルの大きさ |
| `--seed` | 0 | |
| `--tag` | なし | 出力ファイル名の末尾に付く |
| `--save PATH` / `--load PATH` | なし | 重みを `.eqx` で保存 / 読み込んで評価だけ |

全部の引数は `--help`。

## 4. 結果の見方

結果は `models/active_inference/results/cueworld/<名前>.json`。名前は `<環境>_<condition>_<action_mode>_ew<重み>_s<seed>[_tag]`。学習後のエージェントと一様ランダムを、同じ評価環境で `--eval-episodes`（既定 20）本ずつ回した指標が入る。

| キー | 中身 |
|---|---|
| `action_hist` | 各行動を選んだ割合 |
| `frac_determined` | 選んだ行動の次の観測が、環境の規則で 1 つに決まっていた割合。sanity は決まる行動、short は `n_meaningful` 内の記号、long は位相に合った行動、branch は残りの枝の返事が 1 つの行動 |
| `frac_reward` | sanity のみ。報酬行動の割合 |
| `final_remaining` | branch のみ。エピソード末に残った枝の数の平均 |
| `mean_return` | 報酬の合計の平均 |

`agent` の `frac_determined` が `random` より高ければ、予測しやすい行動に寄っている。yoke では観測が行動と無関係なので、差が出ないのが期待どおり。

```bash
python -m active_inference.experiments.summarize_cueworld              # 全部
python -m active_inference.experiments.summarize_cueworld comm_short   # 名前の部分一致
```

## 5. まとめて回す

```bash
# seed × 条件
for s in 0 1 2; do for c in dependent yoke; do
  python -m active_inference.experiments.cueworld_subtasks --env communication --dependency short --condition $c --seed $s
done; done

# 情報利得の重みの符号と大きさ
for ew in -1 -0.1 0 0.1 1; do
  python -m active_inference.experiments.cueworld_subtasks --env communication --dependency long --epistemic-weight $ew
done

# 3 種類の依存 × 方策 / 候補
for d in short long branch; do for m in policy candidate; do
  python -m active_inference.experiments.cueworld_subtasks --env communication --dependency $d --action-mode $m
done; done
```

ホストから流しっぱなしにするとき。

```bash
docker exec -d nakano_infant_project bash -lc 'cd /workspace && PYTHONPATH=/workspace/models \
  python -m active_inference.experiments.cueworld_subtasks --env communication --dependency long \
  > /workspace/logs/cueworld/long.log 2>&1'
tail -f ~/infant_model/proposed_model/logs/cueworld/long.log
```

## 6. つまずきやすいところ

- `subtasks_for_cueworld が見つかりません`: 0 のマウントができていない。`docker exec nakano_infant_project ls /cueworld` で確認する。
- `No module named 'gymnasium'`: イメージを作り直していない。`pip install gymnasium==1.2.1` で一時対応。
- `--seq-len は --max-steps 以下に`: エピソードが短いと系列を切れず、学習が一度も走らない。
- `policy モードは --efe-mode entropy のみ`: `kl` を試すなら `--action-mode candidate`。
- 初回の数エピソードは JIT のコンパイルで遅い。2 本目以降は速い。
