# 感情報酬SFT (Supervised Fine-Tuning) システム

大学サーバー（Python 3.10.12）対応の感情報酬モデル学習システムです。

## 📁 ファイル構成

### 🔥 コアスクリプト
- **`emotion_reward_sft.py`** - メインの学習スクリプト
- **`data_processing.py`** - データ処理・前処理
- **`llm_evaluation.py`** - LLM評価機能
- **`turn_segmentation.py`** - 対話のターン分割

### 🔬 実験管理
- **`experiment_config.py`** - 実験設定管理
- **`experiment_tracker.py`** - TensorBoard & W&B統合
- **`experiment_setup.py`** - 環境セットアップスクリプト

### 📚 ドキュメント
- **`EXPERIMENT_TRACKING_README.md`** - 実験管理機能の詳細ガイド
- **`README.md`** - このファイル

## 🚀 使用方法

### 1. 環境セットアップ（初回のみ）
```bash
# 実験管理ツールのセットアップ
python3 experiment_setup.py

# 必要なライブラリのインストール
python3 -m pip install --user torch transformers datasets peft tensorboard wandb
```

### 2. 学習の実行
```bash
# メインスクリプトの実行
python3 emotion_reward_sft.py
```

### 3. 実験結果の確認
```bash
# TensorBoardの起動（自動生成されるスクリプト使用）
./start_tensorboard.sh

# または手動起動
python3 -m tensorboard.main --logdir=./logs_tensorboard --port=6006
```

## ⚙️ 設定

### 実験管理ツール
`emotion_reward_sft.py` 内で設定可能：
```python
EXPERIMENT_TRACKING_TOOL = "both"  # "tensorboard", "wandb", "both", "none"
```

### 環境変数（オプション）
```bash
export WANDB_API_KEY=your_api_key
export WANDB_PROJECT=emotion-reward-sft
export EXPERIMENT_TRACKING_TOOL=both
```

## 📊 記録される情報

- **メトリクス**: 訓練損失、MSE損失、学習率、勾配ノルム
- **ハイパーパラメータ**: モデル設定、LoRAパラメータ、バッチサイズ
- **アーティファクト**: 学習済みモデル（W&B）

## 🔧 互換性

- **Python**: 3.6以降（3.10.12で動作確認済み）
- **プラットフォーム**: Linux, macOS, Windows
- **依存関係**: PyTorch, Transformers, Datasets, PEFT, TensorBoard, W&B

## 📝 特徴

### Python互換性
- f-string を `.format()` 形式に変更
- 型ヒントを削除（古いバージョン対応）
- `__future__` インポートを削除

### トークナイザー対応
- 複数の方法でトークナイザー読み込みを試行
- SentencePiece問題の自動回避
- 代替モデルへの自動フォールバック

### 実験管理
- TensorBoard と W&B の統合
- 自動的なメトリクス記録
- モデルアーティファクトの保存

## 🆘 トラブルシューティング

### よくある問題
1. **ModuleNotFoundError**: 必要なライブラリをインストール
2. **SentencePiece エラー**: 自動的にフォールバック処理
3. **権限エラー**: `--user` フラグでインストール

### 解決方法
```bash
# 依存関係の再インストール
python3 -m pip install --user --upgrade torch transformers datasets

# キャッシュのクリア
python3 -m pip cache purge

# 実験管理ツールの再セットアップ
python3 experiment_setup.py
```

## 📈 実行例

```bash
# 1. セットアップ
python3 experiment_setup.py

# 2. 学習実行
python3 emotion_reward_sft.py

# 3. 結果確認
python3 -m tensorboard.main --logdir=./logs_tensorboard
# ブラウザで http://localhost:6006 を開く
```

## 🔄 生成されるファイル・ディレクトリ

実行時に以下が自動生成されます：
- `logs_tensorboard/` - TensorBoardログ
- `swallow_emotion_reward_adapter/` - 学習済みモデル
- `regression_dataset_real_labels.jsonl` - データキャッシュ
- `start_tensorboard.sh` - TensorBoard起動スクリプト
- `.env.example` - 環境変数テンプレート
