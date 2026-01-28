# OpenAI GPT-4o mini Supervised Fine Tuning

このディレクトリには、OpenAI GPT-4o miniをKokoro Chatデータセットでファインチューニングするためのコードが含まれています。

## 機能

- ✅ KokoroChat データセットの自動読み込みとOpenAI形式への変換
- ✅ **8:1:1データ分割**（train:test:valid）による適切な学習・評価
- ✅ 10エポック、32バッチサイズでのファインチューニング
- ✅ チューニング済みモデルの自動保存
- ✅ 平均二乗誤差（MSE）による損失計算（OpenAI APIで自動処理）
- ✅ **包括的モデル評価機能**（テストデータ・検証データ両方で評価）
- ✅ CPU・メモリ使用量の監視機能
- ✅ 進捗ログとエラーハンドリング
- ✅ 再現性のためのシード設定

## 必要な環境設定

### 1. 依存関係のインストール

```bash
# ryeを使用する場合
rye sync

# pipを使用する場合  
pip install openai datasets tiktoken psutil tqdm
```

### 2. OpenAI APIキーの設定

プロジェクトルートに`.env`ファイルを作成（`.env.example`をコピーすると安全です）：

```bash
# プロジェクトルートに.envファイルを作成
echo "OPENAI_API_KEY=your-api-key-here" > .env
```

`.env`ファイルの内容例：
```
# OpenAI API設定
OPENAI_API_KEY=your-openai-api-key-here

# Google Cloud設定（既存のGCSサービスで使用）
GOOGLE_APPLICATION_CREDENTIALS=credentials/your-service-account.json
```

または、環境変数として設定：
```bash
export OPENAI_API_KEY="your-api-key-here"
```

## 使用方法

### 基本的な使用方法

```python
from openai_sft import OpenAISFT

# SFTインスタンス作成（.envファイルから自動的にAPIキーを読み込み）
sft = OpenAISFT()

# フルパイプライン実行（10エポック、32バッチサイズ）
sft.run_full_pipeline(epochs=10, batch_size=32)
```

### コマンドライン実行

```bash
# 基本実行（8:1:1データ分割）
python openai_sft_example.py

# カスタムパラメータ
python openai_sft_example.py --epochs 5 --batch-size 16 --max-samples 1000 --seed 123

# データ準備のみ（8:1:1分割）
python openai_sft_example.py --data-only --max-samples 300

# 評価のみ
python openai_sft_example.py --evaluate-only --model-id ft:gpt-4o-mini-2024-07-18:your-org:model:abc123

# 詳細設定
python openai_sft_example.py --epochs 10 --batch-size 32 --max-test-cases 20 --seed 42
```

### ステップバイステップ実行

```python
from openai_sft import OpenAISFT

# SFTインスタンス作成（.envファイルから自動的にAPIキーを読み込み）
sft = OpenAISFT()

# 1. データセット分割（8:1:1）
train_dataset, test_dataset, valid_dataset = sft.load_and_split_dataset(max_samples=1000, seed=42)

# 2. 各データセットを準備
train_data = sft.prepare_dataset(train_dataset, "train")
test_data = sft.prepare_dataset(test_dataset, "test")
valid_data = sft.prepare_dataset(valid_dataset, "valid")

# 3. 全データセット保存
filepaths = sft.save_all_datasets(train_data, test_data, valid_data)

# 4. ファイルアップロード
file_id = sft.upload_training_file(filepaths["train"])

# 5. ファインチューニング開始
job_id = sft.create_fine_tune_job(file_id, epochs=10, batch_size=32)

# 6. 進捗監視
job_result = sft.monitor_fine_tune_job(job_id)

# 7. 包括的モデル評価
if job_result.status == 'succeeded':
    model_id = job_result.fine_tuned_model
    results = sft.evaluate_model_comprehensive(model_id, test_data, valid_data, max_test_cases=10)
    sft.save_results(results)
```

## データ形式

### 入力データ（KokoroChat）

```json
{
  "dialogue": [
    {"role": "client", "utterance": "最近、仕事でストレスを感じています..."},
    {"role": "counselor", "utterance": "そのストレスについて、もう少し詳しく教えていただけますか？"}
  ]
}
```

### OpenAI形式への変換後

```json
{
  "messages": [
    {"role": "user", "content": "最近、仕事でストレスを感じています..."},
    {"role": "assistant", "content": "そのストレスについて、もう少し詳しく教えていただけますか？"}
  ]
}
```

## コスト計算

GPT-4o miniの料金（2024年現在）：

- **入力**: $0.40 / 1M tokens
- **出力**: $1.60 / 1M tokens
- **ファインチューニング**: 入力料金と同じ

KokoroChat全データ（11.62M tokens）での推定コスト：

- **学習コスト（10エポック）**: 約6,823円
- **推論コスト**: 約1,091円
- **総コスト**: 約7,914円

## システム監視

実行中は以下の情報が30秒間隔で表示されます：

```
[監視] CPU: 45.2% | メモリ: 62.1% (8.5/16.0GB) | ディスク: 73.4%
```

アラート条件：
- CPU使用率 > 90%
- メモリ使用率 > 90%  
- ディスク使用率 > 90%

## 出力ファイル

実行すると `openai_sft_outputs/` ディレクトリに以下のファイルが生成されます：

```
openai_sft_outputs/
├── train_data_20250101_120000.jsonl       # トレーニングデータ（80%）
├── test_data_20250101_120000.jsonl        # テストデータ（10%）
├── valid_data_20250101_120000.jsonl       # 検証データ（10%）
├── comprehensive_evaluation_20250101_130000.json # 包括的評価結果
└── ...
```

### データ分割の詳細

- **トレーニングデータ（80%）**: モデル学習に使用
- **テストデータ（10%）**: モデル性能の最終評価
- **検証データ（10%）**: ハイパーパラメータ調整やモデル選択

### 評価結果の構造

```json
{
  "model_id": "ft:gpt-4o-mini-2024-07-18:your-org:model:abc123",
  "evaluation_timestamp": "2025-01-01T13:00:00",
  "test_evaluation": {
    "dataset_type": "test",
    "total_test_cases": 659,
    "evaluated_cases": 10,
    "responses": [...]
  },
  "valid_evaluation": {
    "dataset_type": "valid",
    "total_test_cases": 659,
    "evaluated_cases": 10,
    "responses": [...]
  },
  "summary": {
    "test_cases_evaluated": 10,
    "valid_cases_evaluated": 10,
    "total_cases_evaluated": 20
  }
}
```

## トラブルシューティング

### よくある問題

1. **APIキーエラー**
   ```
   ValueError: OpenAI API keyが設定されていません
   ```
   → プロジェクトルートに`.env`ファイルを作成し、`OPENAI_API_KEY=your-api-key-here`を設定してください

2. **メモリ不足**
   ```
   ⚠️ メモリ使用率が高いです: 92.3%
   ```
   → `max_samples` パラメータでデータサイズを制限してください

3. **ファインチューニング失敗**
   ```
   ファインチューニング失敗: failed
   ```
   → OpenAIのダッシュボードでエラー詳細を確認してください

### デバッグ方法

```python
# デバッグ用に少数サンプルで実行
sft.run_full_pipeline(epochs=1, batch_size=4, max_samples=10)
```

## 注意事項

- OpenAI APIの利用料金が発生します
- ファインチューニングには時間がかかります（数時間〜数日）
- 大量のデータを扱う場合はメモリ使用量にご注意ください
- 実運用前に必ずテスト実行を行ってください

## ライセンス

このコードはMITライセンスの下で提供されています。
