# SFTモデルの保存と使用方法

このドキュメントでは、OpenAI Supervised Fine Tuning（SFT）で作成したモデルを保存し、後で使用する方法について説明します。

## 概要

OpenAIのSFTモデルは直接ダウンロードできませんが、以下の方法でモデルを管理・使用できます：

1. **モデル情報の保存**: ファインチューニング完了後、モデルIDと学習パラメータを自動保存
2. **モデル一覧管理**: 保存されたモデルの一覧表示と検索
3. **簡単な使用**: 保存されたモデルを使った推論実行

## ファイル構成

```
src/core/feedback/
├── openai_sft.py              # メインライブラリ（モデル管理機能追加済み）
├── openai_sft_example.py      # 使用例スクリプト（--list-modelsオプション追加）
├── use_saved_model.py         # 保存されたモデルの使用専用スクリプト
├── test_openai_sft.py         # テストスクリプト
└── openai_sft_outputs/        # 出力ディレクトリ（自動作成）
    ├── model_info_*.json      # モデル情報ファイル
    ├── train_data_*.jsonl     # 学習データ
    ├── test_data_*.jsonl      # テストデータ
    ├── valid_data_*.jsonl     # 検証データ
    └── evaluation_results_*.json # 評価結果
```

## 使用方法

### 1. ファインチューニング実行（モデル情報自動保存）

```bash
# 基本実行（モデル情報が自動保存される）
python openai_sft_example.py

# カスタムパラメータで実行
python openai_sft_example.py --epochs 5 --batch-size 8 --max-samples 1000
```

ファインチューニング完了後、以下のファイルが自動生成されます：
- `model_info_ft_gpt-4o-mini-2024-07-18_your-org_your-model_abc123_20250824_150000.json`

### 2. 保存されたモデル一覧の表示

```bash
# 方法1: example スクリプトから
python openai_sft_example.py --list-models

# 方法2: 専用スクリプトから
python use_saved_model.py --list
```

出力例：
```
=== 保存されているモデル (2件) ===
1. ft:gpt-4o-mini-2024-07-18:your-org:kokoro-chat-v1:abc123
   タイムスタンプ: 20250824_150000
   学習パラメータ: {'epochs': 10, 'batch_size': 32, 'max_samples': None}
   ファイル: openai_sft_outputs/model_info_ft_gpt-4o-mini-2024-07-18_your-org_kokoro-chat-v1_abc123_20250824_150000.json

2. ft:gpt-4o-mini-2024-07-18:your-org:kokoro-chat-v2:def456
   タイムスタンプ: 20250823_140000
   学習パラメータ: {'epochs': 5, 'batch_size': 16, 'max_samples': 1000}
   ファイル: openai_sft_outputs/model_info_ft_gpt-4o-mini-2024-07-18_your-org_kokoro-chat-v2_def456_20250823_140000.json
```

### 3. 保存されたモデルの使用

#### A. 単発推論

```bash
# 特定のモデルIDで推論
python use_saved_model.py --model-id ft:gpt-4o-mini-2024-07-18:your-org:kokoro-chat-v1:abc123 --input "最近、仕事でストレスを感じています。"

# 最新のモデルを自動選択
python use_saved_model.py --use-latest --input "人間関係で悩んでいます。"

# モデル情報ファイルから読み込み
python use_saved_model.py --model-file openai_sft_outputs/model_info_ft_gpt-4o-mini-2024-07-18_your-org_kokoro-chat-v1_abc123_20250824_150000.json --input "相談内容"
```

#### B. インタラクティブモード（対話形式）

```bash
# 特定のモデルで対話
python use_saved_model.py --interactive --model-id ft:gpt-4o-mini-2024-07-18:your-org:kokoro-chat-v1:abc123

# 最新のモデルで対話
python use_saved_model.py --interactive --use-latest
```

対話例：
```
相談者: 最近、仕事でストレスを感じています。
カウンセラー: お仕事でストレスを感じていらっしゃるのですね。どのようなことが特にストレスの原因となっているでしょうか？

相談者: 上司との関係が上手くいかなくて...
カウンセラー: 上司との関係でお困りなのですね。具体的にはどのような場面で困難を感じることが多いでしょうか？

相談者: quit
対話を終了します
```

### 4. プログラムから使用

```python
from openai_sft import OpenAISFT

# SFTインスタンス作成
sft = OpenAISFT()

# 保存されたモデル一覧取得
models = sft.list_saved_models()
latest_model = models[0]['model_id'] if models else None

# モデル使用
if latest_model:
    response = sft.use_saved_model(
        model_id=latest_model,
        messages=[{"role": "user", "content": "相談内容"}],
        max_tokens=150,
        temperature=0.7
    )
    print(response)

# モデル情報ファイルから読み込み
model_info = sft.load_model_from_file("openai_sft_outputs/model_info_*.json")
print(f"モデルID: {model_info['model_id']}")
print(f"学習パラメータ: {model_info['training_params']}")
```

## モデル情報ファイルの構造

```json
{
  "model_id": "ft:gpt-4o-mini-2024-07-18:your-org:kokoro-chat-v1:abc123",
  "timestamp": "20250824_150000",
  "base_model": "gpt-4o-mini-2024-07-18",
  "training_params": {
    "epochs": 10,
    "batch_size": 32,
    "max_samples": null,
    "seed": 42,
    "use_evaluation_prompts": false,
    "data_splits": {
      "train": 5271,
      "test": 659,
      "valid": 659
    }
  },
  "evaluation_results": {
    "model_id": "ft:gpt-4o-mini-2024-07-18:your-org:kokoro-chat-v1:abc123",
    "evaluation_timestamp": "2025-08-24T15:00:00",
    "test_evaluation": { ... },
    "valid_evaluation": { ... }
  },
  "usage_instructions": {
    "description": "このモデルをOpenAI APIで使用する方法",
    "example_code": "..."
  }
}
```

## 重要な注意事項

### 1. モデルの保存場所
- OpenAIのSFTモデルは**OpenAIのサーバー上**に保存されます
- ローカルにダウンロードすることは**できません**
- モデルの使用には**常にOpenAI APIが必要**です

### 2. コスト管理
- モデルの使用には推論コストがかかります
- `gpt_cost.py`でコスト計算ができます
- 現在の推論単価: 入力59円/100万トークン、出力235円/100万トークン

### 3. モデルの管理
- モデルIDは一意で、削除されない限り永続的に使用可能
- 不要なモデルはOpenAIダッシュボードから削除できます
- モデル情報ファイルは手動で削除可能（モデル自体は残る）

### 4. APIキーの管理
- `.env`ファイルに`OPENAI_API_KEY=your-api-key-here`を設定
- APIキーは外部に漏らさないよう注意
- 本番環境では環境変数での管理を推奨

## トラブルシューティング

### Q: モデル一覧が表示されない
A: `openai_sft_outputs/`ディレクトリに`model_info_*.json`ファイルがあるか確認してください。

### Q: モデルが使用できない
A: 
1. APIキーが正しく設定されているか確認
2. モデルIDが正確か確認
3. OpenAIダッシュボードでモデルが削除されていないか確認

### Q: 推論が遅い・エラーが出る
A:
1. OpenAI APIの利用制限を確認
2. `max_tokens`や`temperature`パラメータを調整
3. ネットワーク接続を確認

## 参考リンク

- [OpenAI Fine-tuning Guide](https://platform.openai.com/docs/guides/fine-tuning)
- [OpenAI API Documentation](https://platform.openai.com/docs/api-reference)
- [KokoroChat Dataset](https://huggingface.co/datasets/UEC-InabaLab/KokoroChat)
