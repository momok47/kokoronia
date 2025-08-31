#!/usr/bin/env python3
"""
既存のファインチューニングジョブからloss情報を取得してグラフを作成するスクリプト
"""

import os
import json
import logging
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import numpy as np

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_fine_tuning_job_info(job_id: str, api_key: str):
    """
    ファインチューニングジョブの詳細情報を取得
    
    Args:
        job_id: ファインチューニングジョブID
        api_key: OpenAI APIキー
        
    Returns:
        ジョブ情報
    """
    client = OpenAI(api_key=api_key)
    
    try:
        job = client.fine_tuning.jobs.retrieve(job_id)
        logger.info(f"ジョブ情報を取得: {job_id}")
        return job
    except Exception as e:
        logger.error(f"ジョブ情報の取得に失敗: {e}")
        return None

def get_fine_tuning_events(job_id: str, api_key: str):
    """
    ファインチューニングジョブのイベント（loss情報）を取得
    
    Args:
        job_id: ファインチューニングジョブID
        api_key: OpenAI APIキー
        
    Returns:
        イベントリスト
    """
    client = OpenAI(api_key=api_key)
    
    try:
        # 正しいメソッドを使用
        events = client.fine_tuning.jobs.list_events(fine_tuning_job_id=job_id)
        logger.info(f"イベント情報を取得: {job_id}")
        return events
    except Exception as e:
        logger.error(f"イベント情報の取得に失敗: {e}")
        return None

def parse_training_loss(events):
    """
    イベントから学習lossを抽出
    
    Args:
        events: ファインチューニングイベント
        
    Returns:
        (timestamps, losses, steps): タイムスタンプ、loss値、ステップ数のリスト
    """
    timestamps = []
    losses = []
    steps = []
    
    print(f"\n=== イベント情報の詳細 ===")
    print(f"イベント数: {len(events.data)}")
    
    for i, event in enumerate(events.data):
        print(f"\nイベント {i+1}:")
        print(f"  タイプ: {event.type}")
        print(f"  作成日時: {event.created_at}")
        print(f"  データ: {event.data}")
        
        if hasattr(event, 'data') and event.data:
            # metricsタイプのイベントからtrain_lossを抽出
            if event.type == 'metrics' and 'train_loss' in event.data:
                loss = event.data['train_loss']
                if loss is not None:
                    timestamps.append(event.created_at)
                    losses.append(loss)
                    
                    # ステップ数を抽出
                    step = event.data.get('step', len(steps))
                    steps.append(step)
                    print(f"  → Loss: {loss}, Step: {step}")
    
    print(f"\n=== 抽出結果 ===")
    print(f"タイムスタンプ数: {len(timestamps)}")
    print(f"Loss数: {len(losses)}")
    print(f"ステップ数: {len(steps)}")
    
    return timestamps, losses, steps

def create_loss_plot(timestamps, losses, steps, job_id, model_id):
    """
    lossのグラフを作成
    
    Args:
        timestamps: タイムスタンプのリスト
        losses: loss値のリスト
        steps: ステップ数のリスト
        job_id: ジョブID
        model_id: モデルID
    """
    if not timestamps or not losses:
        logger.warning("lossデータが不足しています")
        return
    
    # タイムスタンプをdatetimeオブジェクトに変換
    dt_timestamps = [datetime.fromtimestamp(ts) for ts in timestamps]
    
    # グラフの作成
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 1. 時間軸でのlossグラフ
    ax1.plot(dt_timestamps, losses, 'b-', linewidth=2, marker='o', markersize=4)
    ax1.set_title(f'Training Loss Over Time\nJob: {job_id}\nModel: {model_id}', fontsize=14)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Loss')
    ax1.grid(True, alpha=0.3)
    
    # x軸の日時フォーマット
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax1.xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # 2. ステップ数でのlossグラフ
    if steps and len(steps) == len(losses):
        ax2.plot(steps, losses, 'r-', linewidth=2, marker='s', markersize=4)
        ax2.set_title('Training Loss Over Steps', fontsize=14)
        ax2.set_xlabel('Training Step')
        ax2.set_ylabel('Loss')
        ax2.grid(True, alpha=0.3)
    else:
        # ステップ数が利用できない場合は、インデックスを使用
        ax2.plot(range(len(losses)), losses, 'r-', linewidth=2, marker='s', markersize=4)
        ax2.set_title('Training Loss Over Training Progress', fontsize=14)
        ax2.set_xlabel('Training Progress')
        ax2.set_ylabel('Loss')
        ax2.grid(True, alpha=0.3)
    
    # 統計情報を表示
    if losses:
        min_loss = min(losses)
        max_loss = max(losses)
        final_loss = losses[-1]
        
        stats_text = f'Min Loss: {min_loss:.4f}\nMax Loss: {max_loss:.4f}\nFinal Loss: {final_loss:.4f}'
        ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # グラフを保存
    output_dir = Path("openai_sft_outputs")
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"training_loss_{job_id}_{timestamp}.png"
    filepath = output_dir / filename
    
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    logger.info(f"グラフを保存: {filepath}")
    
    # グラフを表示
    plt.show()
    
    return str(filepath)

def get_latest_fine_tuning_jobs(api_key: str, limit: int = 10):
    """
    最新のファインチューニングジョブ一覧を取得
    
    Args:
        api_key: OpenAI APIキー
        limit: 取得するジョブ数
        
    Returns:
        ジョブ一覧
    """
    client = OpenAI(api_key=api_key)
    
    try:
        jobs = client.fine_tuning.jobs.list(limit=limit)
        logger.info(f"最新の{len(jobs.data)}件のジョブを取得しました")
        return jobs
    except Exception as e:
        logger.error(f"ジョブ一覧の取得に失敗: {e}")
        return None

def display_job_selection(jobs):
    """
    ジョブ選択画面を表示
    
    Args:
        jobs: ファインチューニングジョブ一覧
        
    Returns:
        選択されたジョブID
    """
    print("\n=== 利用可能なファインチューニングジョブ ===")
    
    for i, job in enumerate(jobs.data):
        # ステータスに応じたアイコン
        status_icon = "✅" if job.status == 'succeeded' else "❌" if job.status == 'failed' else "⏳"
        
        # 作成日時を読みやすい形式に変換
        created_time = datetime.fromtimestamp(job.created_at).strftime("%Y-%m-%d %H:%M")
        
        # モデル名を取得
        model_name = getattr(job, 'fine_tuned_model', 'N/A')
        
        print(f"{i+1}. {status_icon} {job.id}")
        print(f"   ステータス: {job.status}")
        print(f"   モデル: {model_name}")
        print(f"   作成日時: {created_time}")
        
        # 成功したジョブの場合は詳細情報も表示
        if job.status == 'succeeded':
            if hasattr(job, 'trained_tokens') and job.trained_tokens:
                print(f"   学習済みトークン: {job.trained_tokens:,}")
            if hasattr(job, 'trained_examples') and job.trained_examples:
                print(f"   学習済みサンプル: {job.trained_examples:,}")
        print()
    
    # ユーザーに選択を求める
    while True:
        try:
            choice = input("ジョブ番号を選択してください (1-{}): ".format(len(jobs.data)))
            job_index = int(choice) - 1
            
            if 0 <= job_index < len(jobs.data):
                selected_job = jobs.data[job_index]
                print(f"\n選択されたジョブ: {selected_job.id}")
                return selected_job.id
            else:
                print("無効な選択です。1-{}の範囲で選択してください。".format(len(jobs.data)))
        except ValueError:
            print("数字を入力してください。")
        except KeyboardInterrupt:
            print("\nキャンセルされました。")
            return None

def main():
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
    
    print("=== ファインチューニングジョブのlossグラフ作成 ===")
    
    try:
        # 最新のジョブ一覧を取得
        jobs = get_latest_fine_tuning_jobs(api_key, limit=20)
        if not jobs or not jobs.data:
            logger.error("ファインチューニングジョブが見つかりません")
            return
        
        # 成功したジョブのみをフィルタリング
        successful_jobs = [job for job in jobs.data if job.status == 'succeeded']
        
        if not successful_jobs:
            logger.warning("成功したジョブが見つかりません。失敗したジョブも表示します。")
            successful_jobs = jobs.data
        
        # ジョブ選択
        job_id = display_job_selection(type('Jobs', (), {'data': successful_jobs})())
        if not job_id:
            return
        
        # ジョブ情報を取得
        job_info = get_fine_tuning_job_info(job_id, api_key)
        if not job_info:
            return
        
        print(f"\n=== ジョブ情報 ===")
        print(f"ステータス: {job_info.status}")
        print(f"モデル: {getattr(job_info, 'fine_tuned_model', 'N/A')}")
        print(f"作成日時: {datetime.fromtimestamp(job_info.created_at)}")
        
        if job_info.status != 'succeeded':
            logger.warning(f"ジョブが完了していません: {job_info.status}")
            return
        
        # イベント情報を取得
        events = get_fine_tuning_events(job_id, api_key)
        if not events:
            return
        
        # loss情報を抽出
        timestamps, losses, steps = parse_training_loss(events)
        
        if not losses:
            logger.warning("lossデータが見つかりませんでした")
            return
        
        print(f"\n=== Loss情報 ===")
        print(f"データポイント数: {len(losses)}")
        print(f"最小loss: {min(losses):.4f}")
        print(f"最大loss: {max(losses):.4f}")
        print(f"最終loss: {losses[-1]:.4f}")
        
        # グラフを作成
        model_id = getattr(job_info, 'fine_tuned_model', 'Unknown')
        plot_file = create_loss_plot(timestamps, losses, steps, job_id, model_id)
        
        if plot_file:
            print(f"\n✅ グラフが作成されました: {plot_file}")
        
    except Exception as e:
        logger.error(f"エラーが発生しました: {e}")

if __name__ == "__main__":
    main()
