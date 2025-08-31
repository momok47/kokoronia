#!/usr/bin/env python3
"""
失敗したファインチューニングジョブの詳細を確認するスクリプト
"""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_failed_job(job_id: str):
    """
    失敗したジョブの詳細を確認
    
    Args:
        job_id: 失敗したジョブID
    """
    # プロジェクトルートの.envファイルを読み込み
    project_root = Path(__file__).parent.parent.parent.parent
    env_path = project_root / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        logger.info(f".envファイルを読み込みました: {env_path}")
    
    # APIキーの確認
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OpenAI APIキーが設定されていません")
        return
    
    try:
        client = OpenAI(api_key=api_key)
        
        # ジョブ情報を取得
        job = client.fine_tuning.jobs.retrieve(job_id)
        
        print(f"\n=== ジョブ詳細: {job_id} ===")
        print(f"ステータス: {job.status}")
        print(f"モデル: {job.model}")
        print(f"作成日時: {job.created_at}")
        print(f"完了日時: {job.finished_at}")
        print(f"トレーニングファイル: {job.training_file}")
        
        if hasattr(job, 'error') and job.error:
            print(f"\n❌ エラー詳細:")
            print(f"エラー: {job.error}")
        
        if hasattr(job, 'result_files') and job.result_files:
            print(f"\n📁 結果ファイル:")
            for file_id in job.result_files:
                print(f"  - {file_id}")
        
        if hasattr(job, 'hyperparameters') and job.hyperparameters:
            print(f"\n⚙️ ハイパーパラメータ:")
            for key, value in job.hyperparameters.items():
                print(f"  {key}: {value}")
        
        print(f"\n📊 統計情報:")
        print(f"  処理済みトークン数: {getattr(job, 'trained_tokens', 'N/A')}")
        print(f"  処理済みサンプル数: {getattr(job, 'trained_examples', 'N/A')}")
        
    except Exception as e:
        logger.error(f"エラーが発生しました: {e}")

def main():
    # 失敗したジョブID
    failed_job_id = "ftjob-N6KpXmRlivMYE0sMCKNJqxVz"
    
    print(f"失敗したジョブの詳細を確認中: {failed_job_id}")
    debug_failed_job(failed_job_id)

if __name__ == "__main__":
    main()
