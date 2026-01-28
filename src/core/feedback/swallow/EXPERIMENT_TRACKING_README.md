# 実験管理ツール統合ガイド

このプロジェクトにTensorBoardとWeights & Biases (W&B) の実験管理機能が統合されました。

## 🚀 クイックスタート

### 1. セットアップ

```bash
# 実験管理ツールのセットアップを実行
python3 experiment_setup.py
```

### 2. 基本的な使用方法

```bash
# 実験の実行（TensorBoardとW&Bの両方を使用）
python3 emotion_reward_sft.py
```

## 📊 利用可能な実験管理ツール

### TensorBoard
- **ローカル実行**: プライベートな実験管理
- **リアルタイムモニタリング**: 損失、メトリクスの可視化
- **ハイパーパラメータ比較**: 複数実験の比較

### Weights & Biases (W&B)
- **クラウドベース**: チーム共有、リモートアクセス
- **高機能ダッシュボード**: 詳細な分析とレポート
- **モデル管理**: アーティファクトの自動保存

## ⚙️ 設定方法

### 環境変数での設定（推奨）

```bash
# .env ファイルを作成
cp .env.example .env

# .env ファイルを編集
export EXPERIMENT_TRACKING_TOOL=both  # "tensorboard", "wandb", "both", "none"
export WANDB_API_KEY=your_api_key_here
export WANDB_PROJECT=emotion-reward-sft
```

### コード内での設定

`emotion_reward_sft.py` の設定部分を編集:

```python
# 実験管理の設定
EXPERIMENT_TRACKING_TOOL = "both"  # "tensorboard", "wandb", "both", "none"
```

## 📈 記録される指標

### 自動記録される指標
- `train_loss`: 訓練損失
- `mse_loss`: MSE損失
- `learning_rate`: 学習率
- `step`: 訓練ステップ数
- `gradient_norm`: 勾配ノルム

### ハイパーパラメータ
- モデル名、バッチサイズ、学習率
- LoRAパラメータ（r, alpha, dropout）
- データセットサイズ、エポック数

## 🖥️ TensorBoardの使用方法

### 1. TensorBoardの起動

```bash
# 自動生成されたスクリプトを使用
./start_tensorboard.sh

# または手動で起動
tensorboard --logdir=./logs_tensorboard --port=6006
```

### 2. ブラウザでアクセス

```
http://localhost:6006
```

### 3. 表示される情報
- **SCALARS**: 損失とメトリクスの時系列グラフ
- **HPARAMS**: ハイパーパラメータの比較
- **GRAPHS**: モデルの計算グラフ（利用可能な場合）

## ☁️ W&Bの使用方法

### 1. アカウント設定

```bash
# W&Bアカウントの作成
# https://wandb.ai でサインアップ

# ログイン
wandb login
```

### 2. APIキーの設定

```bash
# 環境変数で設定
export WANDB_API_KEY=your_api_key_here

# または対話的に設定
wandb login
```

### 3. プロジェクトの確認

W&Bダッシュボード（https://wandb.ai）で以下を確認:
- 実験の進行状況
- メトリクスの比較
- ハイパーパラメータの影響
- モデルアーティファクト

## 🔧 トラブルシューティング

### TensorBoard関連

```bash
# ポートが使用中の場合
tensorboard --logdir=./logs_tensorboard --port=6007

# ログディレクトリが見つからない場合
mkdir -p ./logs_tensorboard
```

### W&B関連

```bash
# ログインの問題
wandb login --relogin

# オフラインモード（インターネット接続なし）
export WANDB_MODE=offline
```

### 依存関係の問題

```bash
# 必要なパッケージのインストール
pip install tensorboard wandb

# または
python3 experiment_setup.py
```

## 📝 実験管理のベストプラクティス

### 1. 実験の命名規則
- 日付とモデルを含む: `emotion_sft_20250101_swallow7b`
- 設定の変更を反映: `emotion_sft_lr0001_batch4`

### 2. タグの活用（W&B）
- モデルタイプ: `swallow-7b`, `gpt-3.5`
- 実験タイプ: `baseline`, `ablation`, `hyperopt`
- データセット: `kokorochat`, `full-dataset`

### 3. メトリクスの解釈
- `train_loss`: 下降傾向を確認
- `mse_loss`: 回帰タスクの性能指標
- `learning_rate`: 学習率スケジューラーの動作確認

## 🔄 実験の比較

### TensorBoardでの比較
1. 複数の実験ログを同じディレクトリに保存
2. TensorBoardで同時に表示
3. HPARAMSタブでハイパーパラメータ比較

### W&Bでの比較
1. プロジェクトページで複数実験を選択
2. 比較ビューで並列表示
3. スイープ機能でハイパーパラメータ最適化

## 📚 参考資料

- [TensorBoard公式ドキュメント](https://www.tensorflow.org/tensorboard)
- [W&B公式ドキュメント](https://docs.wandb.ai/)
- [PyTorch TensorBoard統合](https://pytorch.org/docs/stable/tensorboard.html)

## 🆘 サポート

問題が発生した場合:
1. ログファイルを確認
2. 環境変数の設定を確認
3. 依存関係のバージョンを確認
4. 実験管理ツールの公式ドキュメントを参照
