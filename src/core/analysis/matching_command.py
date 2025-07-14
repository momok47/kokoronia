#!/usr/bin/env python3
# Matching Command Line Tool
# データベースからユーザーマッチングを実行するコマンドラインツール

import sys
import os
import argparse
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from accounts.models import User, UserTopicScore  # type: ignore

# 相対インポートのためのパス設定
sys.path.append(os.path.dirname(__file__))

# Django設定の初期化
if not hasattr(sys, '_django_setup_done'):
    django_project_root = os.path.join(os.path.dirname(__file__), '..', '..', 'webapp')
    sys.path.insert(0, django_project_root)
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'project.settings')
    
    import django
    django.setup()
    sys._django_setup_done = True

from user_matching import UserMatcher, create_mock_data

# Django関連のインポート（setup後に実行）
User = None
UserTopicScore = None
DJANGO_AVAILABLE = False

try:
    from accounts.models import User, UserTopicScore  # type: ignore
    DJANGO_AVAILABLE = True
except ImportError:
    DJANGO_AVAILABLE = False
    print("エラー: Django環境が利用できません。")
    print("このスクリプトはDjango環境が必要です。")
    sys.exit(1)


def display_user_stats():
    """ユーザー統計を表示"""
    assert User is not None and UserTopicScore is not None, "Django環境が必要です"
    print("=== ユーザー統計 ===")
    
    total_users = User.objects.count()
    users_with_scores = User.objects.filter(topic_scores__score__gt=0).distinct().count()
    
    print(f"総ユーザー数: {total_users}")
    print(f"トピックスコアを持つユーザー数: {users_with_scores}")
    
    if users_with_scores > 0:
        # トピック別統計
        topic_stats = UserTopicScore.objects.values('topic_label').distinct()
        print(f"記録されているトピック数: {len(topic_stats)}")
        
        print("\n【トピック別ユーザー数】")
        for topic_data in topic_stats:
            topic = topic_data['topic_label']
            count = UserTopicScore.objects.filter(
                topic_label=topic, 
                score__gt=0
            ).count()
            scores = UserTopicScore.objects.filter(
                topic_label=topic,
                score__gt=0
            ).values_list('score', flat=True)
            avg_score = sum(scores) / len(scores) if scores else 0
            
            print(f"  {topic:12s}: {count:3d}人 (平均スコア: {avg_score:.3f})")


def display_detailed_user_info(min_score: float = 0.01):
    """詳細なユーザー情報を表示"""
    assert User is not None and UserTopicScore is not None, "Django環境が必要です"
    print(f"\n=== 詳細ユーザー情報（最小スコア: {min_score}）===")
    
    users_with_scores = User.objects.filter(
        topic_scores__score__gte=min_score
    ).distinct()
    
    if not users_with_scores:
        print("該当するユーザーがいません。")
        return
    
    for user in users_with_scores:
        print(f"\n【{user.account_id}】")
        user_scores = UserTopicScore.objects.filter(
            user=user,
            score__gte=min_score
        ).order_by('-score')
        
        if user_scores:
            for score_obj in user_scores:
                print(f"  {score_obj.topic_label:12s}: {score_obj.score:.4f} (更新{score_obj.count}回)")
        else:
            print("  該当するトピックスコアがありません。")


def run_interactive_matching():
    """対話型マッチング実行"""
    assert User is not None and UserTopicScore is not None, "Django環境が必要です"
    print("\n=== 対話型マッチング ===")
    
    # パラメータ設定
    try:
        min_score = float(input("最小スコア閾値を入力してください (デフォルト: 0.01): ") or "0.01")
        use_quantum = input("量子コンピュータを使用しますか？ (y/N): ").lower().startswith('y')
        save_results = input("結果を保存しますか？ (y/N): ").lower().startswith('y')
    except ValueError:
        print("無効な入力です。デフォルト値を使用します。")
        min_score = 0.01
        use_quantum = False
        save_results = False
    
    # マッチング実行
    matcher = UserMatcher()
    result = matcher.run_database_matching(
        min_score_threshold=min_score,
        use_quantum=use_quantum,
        save_results=save_results
    )
    
    # 詳細結果表示
    if result['matches']:
        print(f"\n=== 詳細マッチング結果 ===")
        for i, (user1_id, user2_id) in enumerate(result['matches'], 1):
            print(f"\nマッチング {i}: {user1_id} ↔ {user2_id}")
            
            # 各ユーザーのトップトピックを表示
            try:
                user1 = User.objects.get(account_id=user1_id)
                user2 = User.objects.get(account_id=user2_id)
                
                user1_top = UserTopicScore.objects.filter(
                    user=user1, score__gt=0
                ).order_by('-score').first()
                
                user2_top = UserTopicScore.objects.filter(
                    user=user2, score__gt=0
                ).order_by('-score').first()
                
                if user1_top:
                    print(f"  {user1_id}: {user1_top.topic_label} ({user1_top.score:.3f})")
                if user2_top:
                    print(f"  {user2_id}: {user2_top.topic_label} ({user2_top.score:.3f})")
                    
            except Exception as e:
                print(f"  詳細情報の取得でエラー: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="ユーザーマッチングシステム",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  python matching_command.py --stats                    # ユーザー統計表示
  python matching_command.py --users --min-score 0.05  # 詳細ユーザー情報表示
  python matching_command.py --match                    # クイックマッチング実行
  python matching_command.py --interactive              # 対話型マッチング
  python matching_command.py --mock                     # モックデータテスト
        """
    )
    
    # コマンドライン引数
    parser.add_argument('--stats', action='store_true', help='ユーザー統計を表示')
    parser.add_argument('--users', action='store_true', help='詳細ユーザー情報を表示')
    parser.add_argument('--match', action='store_true', help='クイックマッチング実行')
    parser.add_argument('--interactive', action='store_true', help='対話型マッチング')
    parser.add_argument('--mock', action='store_true', help='モックデータテスト')
    
    # オプション引数
    parser.add_argument('--min-score', type=float, default=0.01, 
                       help='最小スコア閾値 (デフォルト: 0.01)')
    parser.add_argument('--quantum', action='store_true', 
                       help='量子コンピュータを使用（現在はシミュレーション）')
    parser.add_argument('--save', action='store_true', 
                       help='結果を保存')
    
    args = parser.parse_args()
    
    # Django環境でない場合の処理
    if not DJANGO_AVAILABLE:
        print("エラー: Django環境が利用できません。")
        if args.mock:
            print("モックデータテストを実行します...")
            from user_matching import test_mock_matching
            test_mock_matching()
        return
    
    # 引数に応じた処理実行
    if args.stats:
        display_user_stats()
    
    if args.users:
        display_detailed_user_info(args.min_score)
    
    if args.match:
        print("=== クイックマッチング実行 ===")
        matcher = UserMatcher()
        result = matcher.run_database_matching(
            min_score_threshold=args.min_score,
            use_quantum=args.quantum,
            save_results=args.save
        )
        
    if args.interactive:
        run_interactive_matching()
        
    if args.mock:
        print("=== モックデータテスト ===")
        from user_matching import test_mock_matching
        test_mock_matching()
    
    # 引数が何も指定されていない場合
    if not any([args.stats, args.users, args.match, args.interactive, args.mock]):
        print("マッチングコマンドツールへようこそ！")
        print("使用可能なオプション:")
        print("  --help       : ヘルプを表示")
        print("  --stats      : ユーザー統計")
        print("  --users      : ユーザー詳細情報")
        print("  --match      : クイックマッチング")
        print("  --interactive: 対話型マッチング")
        print("  --mock       : モックデータテスト")
        print("\n詳細は --help をご確認ください。")


if __name__ == "__main__":
    # ImportError対策
    try:
        from django.db import models
    except ImportError:
        print("Django環境が利用できません。")
        
    main() 