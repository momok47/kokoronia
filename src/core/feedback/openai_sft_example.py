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

3. 評価のみ実行:
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
    parser.add_argument('--batch-size', type=int, default=16, help='バッチサイズ (default: 16)')
    parser.add_argument('--max-samples', type=int, default=None, help='最大サンプル数（デバッグ用）')
    parser.add_argument('--api-key', type=str, default=None, help='OpenAI APIキー（環境変数優先）')
    
    # 実行モード
    parser.add_argument('--data-only', action='store_true', help='データ準備のみ実行')
    parser.add_argument('--evaluate-only', action='store_true', help='評価のみ実行')
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
        
        if args.evaluate_only:
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
            logger.info("データ準備のみ実行")
            training_data = sft.prepare_dataset(max_samples=args.max_samples)
            training_file = sft.save_training_data(training_data)
            logger.info(f"トレーニングデータを保存しました: {training_file}")
            
        else:
            # フルパイプライン実行
            logger.info("=== フルパイプライン実行 ===")
            logger.info(f"エポック数: {args.epochs}")
            logger.info(f"バッチサイズ: {args.batch_size}")
            logger.info(f"最大サンプル数: {args.max_samples or '制限なし'}")
            
            sft.run_full_pipeline(
                epochs=args.epochs,
                batch_size=args.batch_size,
                max_samples=args.max_samples
            )
        
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
