# Deep Active Inference

JAX（Equinox + Optax + Distrax）でスクラッチ実装した RSSM ベースの深層能動的推論。

進め方は [進め方.md](進め方.md) を参照。cueworld の subtasks を別のマシンで始める手順は [SETUP_cueworld.md](SETUP_cueworld.md)、回し方は [README_cueworld.md](README_cueworld.md)。

## アーキテクチャ

エージェントは、観測を潜在状態にまとめ、その状態から観測と報酬を予測する世界モデルと、想像した未来の期待自由エネルギーで行動を選ぶ部分からなる。単一サンプルの関数を書き、バッチは `vmap`、時系列は `scan` で広げる。

```
観測 (position, goal, 画像)
        │
        ▼
   Encoder ── キーごとに MLP / CNN ──► トークン
        │
        ▼
      RSSM
   deter（GRU の記憶） + stoch（対角ガウス、またはカテゴリカル）
   prior: 前状態と行動だけから次の stoch
   posterior: それにトークンを足して次の stoch
        │
        ├─► ObsDecoder   観測の再構成（変分自由エネルギー）
        ├─► RewardDecoder 報酬の予測（エンコーダには報酬を入れない）
        │
        ▼
  想像ロールアウト（観測なしで prior を進める）
        │
        ▼
  期待自由エネルギー
    実用項     −予測報酬の平均
    情報利得   デフォルトは stoch の事前エントロピー
               オプションは想像観測をエンコードし直した KL
        │
        ├─ 方策モード   PolicyNet が行動分布を出し、和が小さくなるよう学習
        └─ 候補モード   行動空間から列を作り、和が最小の列の先頭を実行
```

学習時の世界モデル損失は、観測の再構成、報酬の対数尤度、事前と事後の KL の和である。方策の更新では世界モデル側の勾配は止める。

主要なファイルは次のとおり。

| ファイル | 役割 |
|---|---|
| `spaces.py` | 環境が宣言する観測・行動空間。形状と離散・連続の正本 |
| `config.py` | 隠れ幅、分布の種類、EFE のモード、行動モード、学習の数値 |
| `networks.py` | MLP、GRU、画像用 CNN |
| `encoder.py` / `decoder.py` | キーごとのエンコードと、観測・報酬のデコーダ |
| `rssm.py` | `observe`（実データ）と `imagine`（想像） |
| `efe.py` | 実用項と情報利得、行動列の評価 |
| `policy.py` | `PolicyNet` と `CandidatePlanner` |
| `agent.py` | 上記を束ねた損失と更新 |
| `envs/point1d.py` | 第一段の 1 次元環境（速度の積分、離散 3 値と連続） |
| `envs/point1d_image.py` | 同じ力学に画像キーを足した第二段の環境 |
| `train.py` | ランダム収集 → 世界モデル → 候補モード → 方策モード |

## 実装確認のテスト

コンテナ `nakano_infant_project` 内で実行する。ホストからは次の形で入る。

```bash
docker exec -it nakano_infant_project bash -lc 'cd /workspace && PYTHONPATH=/workspace/models <コマンド>'
```

以下の `<コマンド>` はコンテナ内の作業ディレクトリ `/workspace` で、`PYTHONPATH=/workspace/models` を付けて実行する。

### 部品

空間の定義。離散行動は one-hot で、連続行動は上限と下限の中に収まること。行動辞書を 1 本のベクトルにできること。

```bash
python -m active_inference.tests.test_spaces
```

第一段の環境。離散の正方向で位置が進むこと。連続の速度が上限で切られ、1 ステップでは目標に届かないこと。`max_steps` でエピソードが終わること。

```bash
python -m active_inference.tests.test_point1d
```

RSSM。事前・事後の形と KL。`scan` による `observe` と `imagine` の時間軸。カテゴリカル潜在の形とエントロピー。

```bash
python -m active_inference.tests.test_rssm
```

期待自由エネルギー。報酬が高い未来のほうが値が小さいこと。離散の候補列（3 のホライズン乗）に対して有限な評価が返ること。

```bash
python -m active_inference.tests.test_efe
```

### 学習が動くこと

ランダム行動のデータで世界モデルを更新し、損失の後半平均が前半より小さいこと。再構成・報酬・KL をまとめた更新が壊れていないことの確認。

```bash
python -m active_inference.tests.test_agent
```

第二段。画像キー付き環境で Encoder と Decoder に CNN が付き、世界モデルの 1 ステップ更新が有限な損失を返すこと。

```bash
python -m active_inference.tests.test_image_env
```

### 閉じたループ

第一段の一連の手続き。ランダム収集のあと世界モデルを学習し、候補モード、続いて方策モードで行動する。標準出力に、ランダム・候補・方策の平均リターンが出る。候補と方策がランダムを上回ることを目視で確認する。

```bash
python -m active_inference.train --seed 0
```

連続行動にするときは `--continuous` を付ける。

```bash
python -m active_inference.train --seed 0 --continuous
```
