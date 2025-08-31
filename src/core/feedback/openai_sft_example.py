#!/usr/bin/env python3
"""
OpenAI GPT-4o mini Supervised Fine Tuning 使用例

このスクリプトは、KokoroChat データセットを使用してGPT-4o miniをファインチューニングする方法を示します。

必要な環境変数:
- OPENAI_API_KEY: OpenAI APIキー

使用方法:
1. 基本的な使用:
   python openai_sft_example.py

2. カスタムパラメータでの実行:
   python openai_sft_example.py --epochs 5 --batch-size 8 --max-samples 500

3. データ制限を調整した実行:
   python openai_sft_example.py --max-tokens-per-sample 3000 --max-messages-per-dialogue 30

4. 評価のみ実行:
   python openai_sft_example.py --evaluate-only --model-id ft:gpt-4o-mini-2024-07-18:your-org:your-model:abc123
"""

import os
import argparse
import logging
from pathlib import Path

from openai_sft import OpenAISFT
from dotenv import load_dotenv

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='OpenAI GPT-4o mini Supervised Fine Tuning')
    
    # パラメータ設定
    parser.add_argument('--epochs', type=int, default=10, help='エポック数 (default: 10)')
    parser.add_argument('--batch-size', type=int, default=32, help='バッチサイズ (default: 32)')
    parser.add_argument('--max-samples', type=int, default=None, help='最大サンプル数（デバッグ用）')
    parser.add_argument('--seed', type=int, default=42, help='データ分割の再現性のためのシード (default: 42)')
    parser.add_argument('--max-test-cases', type=int, default=10, help='評価時の最大テストケース数 (default: 10)')
    parser.add_argument('--evaluation-prompts', action='store_true', help='評価用プロンプト形式を使用')
    
    # データ制限パラメータ
    parser.add_argument('--max-tokens-per-sample', type=int, default=4000, help='サンプルあたりの最大トークン数 (default: 4000)')
    parser.add_argument('--max-messages-per-dialogue', type=int, default=50, help='対話あたりの最大メッセージ数 (default: 50)')
    
    parser.add_argument('--api-key', type=str, default=None, help='OpenAI APIキー（環境変数優先）')
    
    # 実行モード
    parser.add_argument('--data-only', action='store_true', help='データ準備のみ実行')
    parser.add_argument('--evaluate-only', action='store_true', help='評価のみ実行')
    parser.add_argument('--list-models', action='store_true', help='保存されたモデル一覧表示')
    parser.add_argument('--model-id', type=str, help='評価するモデルID（--evaluate-onlyと併用）')
    
    args = parser.parse_args()
    
    # プロジェクトルートの.envファイルを読み込み
    project_root = Path(__file__).parent.parent.parent.parent  # src/core/feedback から4つ上
    env_path = project_root / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        logger.info(f".envファイルを読み込みました: {env_path}")
    
    # APIキーの確認
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OpenAI APIキーが設定されていません。")
        logger.error(f"プロジェクトルートに.envファイルを作成し、OPENAI_API_KEY=your-api-key-here を設定するか、")
        logger.error("--api-key オプションを使用してください。")
        return 1
    
    try:
        # SFTインスタンス作成
        sft = OpenAISFT(api_key=api_key)
        
        if args.list_models:
            # モデル一覧表示
            logger.info("保存されているモデル一覧を表示中...")
            models = sft.list_saved_models()
            
            if not models:
                logger.info("保存されているモデルがありません")
                return 0
            
            logger.info(f"=== 保存されているモデル ({len(models)}件) ===")
            for i, model in enumerate(models, 1):
                logger.info(f"{i}. {model['model_id']}")
                logger.info(f"   タイムスタンプ: {model['timestamp']}")
                logger.info(f"   学習パラメータ: {model['training_params']}")
                logger.info(f"   ファイル: {model['file_path']}")
                logger.info("")
            
        elif args.evaluate_only:
            # 評価のみ実行
            if not args.model_id:
                logger.error("--evaluate-only を使用する場合は --model-id を指定してください")
                return 1
            
            logger.info(f"モデル評価を開始: {args.model_id}")
            
            # テスト用データを準備
            training_data = sft.prepare_dataset(max_samples=20)  # 評価用に少数のサンプル
            test_messages = [item["messages"] for item in training_data]
            
            # 評価実行
            evaluation_results = sft.evaluate_model(args.model_id, test_messages)
            
            # 結果保存
            results_file = sft.save_results(evaluation_results)
            logger.info(f"評価結果を保存しました: {results_file}")
            
        elif args.data_only:
            # データ準備のみ実行
            logger.info("データ準備のみ実行（8:1:1分割）")
            
            # データセット分割（制限チェック付き）
            train_dataset, test_dataset, valid_dataset = sft.load_and_split_dataset(
                max_samples=args.max_samples, seed=args.seed,
                max_tokens_per_sample=args.max_tokens_per_sample,
                max_messages_per_dialogue=args.max_messages_per_dialogue
            )
            
            # 各データセットを準備
            train_data = sft.prepare_dataset(train_dataset, "train")
            test_data = sft.prepare_dataset(test_dataset, "test") 
            valid_data = sft.prepare_dataset(valid_dataset, "valid")
            
            # 全データセットを保存
            filepaths = sft.save_all_datasets(train_data, test_data, valid_data)
            
            logger.info("=== データ準備完了 ===")
            for dataset_type, filepath in filepaths.items():
                logger.info(f"{dataset_type}: {filepath}")
            
        else:
            # フルパイプライン実行
            logger.info("=== フルパイプライン実行（8:1:1データ分割・制限チェック付き） ===")
            logger.info(f"エポック数: {args.epochs}")
            logger.info(f"バッチサイズ: {args.batch_size}")
            logger.info(f"最大サンプル数: {args.max_samples or '制限なし'}")
            logger.info(f"データ分割シード: {args.seed}")
            logger.info(f"最大テストケース数: {args.max_test_cases}")
            logger.info(f"最大トークン数/サンプル: {args.max_tokens_per_sample}")
            logger.info(f"最大メッセージ数/対話: {args.max_messages_per_dialogue}")
            
            result = sft.run_full_pipeline(
                epochs=args.epochs,
                batch_size=args.batch_size,
                max_samples=args.max_samples,
                seed=args.seed,
                max_test_cases=args.max_test_cases,
                max_tokens_per_sample=args.max_tokens_per_sample,
                max_messages_per_dialogue=args.max_messages_per_dialogue
            )
            
            if result and "model_id" in result:
                logger.info(f"✅ ファインチューニング成功: {result['model_id']}")
                logger.info(f"データ分割: {result['data_splits']}")
            else:
                logger.error("❌ ファインチューニングが失敗しました")
        
        logger.info("処理が完了しました！")
        return 0
        
    except KeyboardInterrupt:
        logger.info("ユーザーによって中断されました")
        return 1
    except Exception as e:
        logger.error(f"エラーが発生しました: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
