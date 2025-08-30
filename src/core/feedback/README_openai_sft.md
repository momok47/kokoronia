# OpenAI GPT-4o mini Supervised Fine Tuning

このディレクトリには、OpenAI GPT-4o miniをKokoro Chatデータセットでファインチューニングするためのコードが含まれています。

## 機能

- ✅ KokoroChat データセットの自動読み込みとOpenAI形式への変換
- ✅ 10エポック、16バッチサイズでのファインチューニング
- ✅ チューニング済みモデルの自動保存
- ✅ 平均二乗誤差（MSE）による損失計算（OpenAI APIで自動処理）
- ✅ モデル評価機能
- ✅ CPU・メモリ使用量の監視機能
- ✅ 進捗ログとエラーハンドリング

## 必要な環境設定

### 1. 依存関係のインストール

```bash
# ryeを使用する場合
rye sync

# pipを使用する場合  
pip install openai datasets tiktoken psutil tqdm
```

### 2. OpenAI APIキーの設定

プロジェクトルートに`.env`ファイルを作成（推奨）：

```bash
# プロジェクトルートに.envファイルを作成
echo "OPENAI_API_KEY=your-api-key-here" > .env
```

`.env`ファイルの内容例：
```
# OpenAI API設定
OPENAI_API_KEY=your-openai-api-key-here

# Google Cloud設定（既存のGCSサービスで使用）
GOOGLE_APPLICATION_CREDENTIALS=credentials/my-project-shirakawa-452105-1d7701f94540.json
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

# フルパイプライン実行（10エポック、16バッチサイズ）
sft.run_full_pipeline(epochs=10, batch_size=16)
```

### コマンドライン実行

```bash
# 基本実行
python openai_sft_example.py

# カスタムパラメータ
python openai_sft_example.py --epochs 5 --batch-size 8 --max-samples 1000

# データ準備のみ
python openai_sft_example.py --data-only --max-samples 100

# 評価のみ
python openai_sft_example.py --evaluate-only --model-id ft:gpt-4o-mini-2024-07-18:your-org:model:abc123
```

### ステップバイステップ実行

```python
from openai_sft import OpenAISFT

# SFTインスタンス作成（.envファイルから自動的にAPIキーを読み込み）
sft = OpenAISFT()

# 1. データ準備
training_data = sft.prepare_dataset(max_samples=1000)  # デバッグ用

# 2. データ保存
training_file = sft.save_training_data(training_data)

# 3. ファイルアップロード
file_id = sft.upload_training_file(training_file)

# 4. ファインチューニング開始
job_id = sft.create_fine_tune_job(file_id, epochs=10, batch_size=16)

# 5. 進捗監視
job_result = sft.monitor_fine_tune_job(job_id)

# 6. モデル評価
if job_result.status == 'succeeded':
    model_id = job_result.fine_tuned_model
    test_messages = [item["messages"] for item in training_data[:10]]
    results = sft.evaluate_model(model_id, test_messages)
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
├── training_data_20250101_120000.jsonl    # トレーニングデータ
├── evaluation_results_20250101_130000.json # 評価結果
└── ...
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
