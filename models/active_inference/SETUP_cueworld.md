# cueworld 実験を別のマシンで始める

`SanityEnv` と `CommunicationEnv` を、この能動的推論モデルで学習・評価するまでの手順。引数の意味と条件の回し方は [README_cueworld.md](README_cueworld.md)。

この手順が指す版。

| リポジトリ | URL | ブランチ | 役割 |
|---|---|---|---|
| このリポジトリ | https://github.com/taikinakano0722/active_inference.git | `feature/select_action` | モデル、学習、評価 |
| CueWorld | https://github.com/taikinakano0722/CueWorld.git | `dev` | 環境の正本（`subtasks_for_cueworld`） |

`main` にはこの実験一式が無い。CueWorld は `dev` に `communication.py` の修正がある。

## 0. 必要なもの

- NVIDIA GPU と、そのドライバ（コンテナは CUDA 12.6 のイメージ、JAX は `jax[cuda12]==0.4.33`）
- Docker Engine と Docker Compose v2（`docker compose`）
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)（`docker-compose.yml` の `runtime: nvidia` に使う）
- ビルド中に外へ出るネットワーク（ベースイメージ、pip、Atari ROM）
- ホストの空きポート `6910` と `8000`。この実験自体は使わないが、compose が割り当てる

先に GPU がコンテナから見えるか確認する。

```bash
docker run --rm --gpus all nvidia/cuda:12.6.2-base-ubuntu22.04 nvidia-smi
```

## 1. 並べて clone する

`docker-compose.yml` は `../cueworld` を `/cueworld` にマウントする。フォルダ名は `cueworld`（小文字）で、このリポジトリと**同じ親**に置く。GitHub 上の名前は `CueWorld` なので、clone 時に名前を付ける。

```text
somewhere/
  active_inference/    # このリポジトリ。フォルダ名は任意
  cueworld/            # CueWorld。この名前にする
    subtasks_for_cueworld/
      envs/
        communication.py
        sanity.py
```

```bash
mkdir -p ~/work && cd ~/work
git clone -b feature/select_action https://github.com/taikinakano0722/active_inference.git
git clone -b dev https://github.com/taikinakano0722/CueWorld.git cueworld
```

## 2. コンテナ名を決める

`.env` は git に入っていない。`CONTAINER_NAME` が空だと compose がコンテナ名を決められない。自分のマシンでぶつからない名前にする。

```bash
cd ~/work/active_inference
printf 'CONTAINER_NAME=cueworld_active_inference\n' > .env
```

以降、この名前を `$NAME` として使う。

```bash
NAME=$(grep '^CONTAINER_NAME=' .env | cut -d= -f2)
```

共有マシンで全 GPU を見せたくないときは、`docker-compose.yml` の `count: all` を `count: 1` にし、`NVIDIA_VISIBLE_DEVICES` を `"0"` のように使う番号へ変える。

`6910` か `8000` が既に使われているときは、`ports` の左側だけ変える（例: `"6911:6006"`）。

## 3. イメージを作って起動する

初回は CUDA のベースイメージと Python パッケージの取得がある。数分から、回線によってはそれ以上かかる。

```bash
cd ~/work/active_inference
docker compose up -d --build
docker exec "$NAME" ls /cueworld/subtasks_for_cueworld/envs
docker exec "$NAME" bash -lc 'python -c "import gymnasium, jax; print(gymnasium.__version__); print(jax.devices())"'
```

`communication.py` と `sanity.py` が見え、`gymnasium 1.2.1` と `CudaDevice` が出れば準備できている。作業ディレクトリはコンテナ内の `/workspace` で、ホストのリポジトリがそのままマウントされている。`PYTHONPATH` は compose が付ける。

止めるとき。結果ファイルはマウント側に残る。

```bash
docker compose down
```

## 4. 動作確認

ホストから実行する。対話シェルが要るときは `docker exec -it` にする。

```bash
docker exec "$NAME" bash -lc 'cd /workspace && python -m active_inference.tests.test_cueworld_subtasks'
```

最後に `test_cueworld_subtasks: ok` と出る。初回は JIT で 1 分弱。

環境側のテスト。イメージに pytest は入っていないので、コンテナへ入れる。`/cueworld` は読み取り専用なので、キャッシュを書かない。

```bash
docker exec "$NAME" bash -lc 'pip install pytest==8.3.4 && cd /cueworld/subtasks_for_cueworld && python -m pytest -q -p no:cacheprovider'
```

イメージを作り直すと、この `pip install` は消える。そのときは同じコマンドを打ち直す。

## 5. 1 本回す

既定（`max_steps=50`、3 周、30+10+10 エピソード、方策モード）で 1 本 30 秒ほど。初回の数エピソードは JIT で遅い。

```bash
docker exec "$NAME" bash -lc 'cd /workspace && python -m active_inference.experiments.cueworld_subtasks --env sanity'
docker exec "$NAME" bash -lc 'cd /workspace && python -m active_inference.experiments.cueworld_subtasks --env communication --dependency short --n-meaningful 1'
```

結果はホストの `models/active_inference/results/cueworld/<名前>.json`。名前は `<環境>_<condition>_<action_mode>_ew<重み>_s<seed>`。

```bash
docker exec "$NAME" bash -lc 'cd /workspace && python -m active_inference.experiments.summarize_cueworld'
```

ログを残して流すとき。

```bash
mkdir -p logs/cueworld
docker exec -d "$NAME" bash -lc 'cd /workspace && python -m active_inference.experiments.cueworld_subtasks --env communication --dependency long > /workspace/logs/cueworld/long.log 2>&1'
tail -f logs/cueworld/long.log
```

条件の組み合わせ、引数の表、結果の読み方は [README_cueworld.md](README_cueworld.md)。あそこのコマンドはコンテナ内（`cd /workspace` のあと）の形なので、ホストからは `docker exec "$NAME" bash -lc '...'` で包む。

## 6. つまずきやすいところ

| 症状 | 対処 |
|---|---|
| `subtasks_for_cueworld が見つかりません` | 親ディレクトリに `cueworld` があるか。`CueWorld` のままだとマウントされない。`docker exec "$NAME" ls /cueworld` |
| `No module named 'gymnasium'` | イメージが古い。`docker compose up -d --build` |
| `CONTAINER_NAME` が空、または名前が衝突 | `.env` を書き、`docker compose up -d` |
| ポートが使用中 | `docker-compose.yml` の `6910` か `8000` のホスト側を空いている番号へ変える |
| `runtime: nvidia` で起動できない | NVIDIA Container Toolkit を入れ、手順 0 の `nvidia-smi` が通ることを確認する |
| `CudaDevice` が出ず CPU になる | 同じ。`NVIDIA_VISIBLE_DEVICES` が空になっていないかも見る |
| `--seq-len は --max-steps 以下に` | エピソードが系列より短いと学習が走らない |
| `policy モードは --efe-mode entropy のみ` | `kl` は `--action-mode candidate` |
| pytest が無い | 手順 4 の `pip install pytest==8.3.4`。作り直したあともう一度 |
