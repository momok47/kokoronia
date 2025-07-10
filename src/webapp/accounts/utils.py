"""
アカウント関連のユーティリティ関数
"""
from .models import User, UserTopicScore
from django.core.exceptions import ObjectDoesNotExist


def save_user_insights(account_id, insights):
    """
    分析結果（insights）をデータベースに保存する
    
    Args:
        account_id (str): ユーザーのアカウントID
        insights (dict): zero_shot_learning.extract_insights()の戻り値
                        必須キー: 'best_topic', 'best_score'
    
    Returns:
        tuple: (success: bool, message: str, topic_score: UserTopicScore or None)
    
    Example:
        insights = {
            'best_topic': 'テクノロジー',
            'best_score': 0.85,
            'topic_scores': {...},
            'interest_scores': [...],
            'speaker_ratios': {...}
        }
        success, message, topic_score = save_user_insights('user123', insights)
    """
    try:
        # ユーザーを取得
        user = User.objects.get(account_id=account_id)
        
        # best_topicとbest_scoreを抽出
        best_topic = insights.get('best_topic')
        best_score = insights.get('best_score')
        
        if not best_topic or best_score is None:
            return False, "insights辞書にbest_topicまたはbest_scoreが含まれていません", None
        
        if best_topic == "不明":
            return False, "トピックが'不明'のため、スコアを保存しません", None
        
        # スコアを更新
        topic_score, created = UserTopicScore.update_user_topic_score(
            user=user,
            topic_label=best_topic,
            new_score=best_score
        )
        
        action = "作成" if created else "更新"
        message = f"ユーザー '{account_id}' のトピック '{best_topic}' のスコアを{action}しました (スコア: {topic_score.score:.4f}, 回数: {topic_score.count})"
        
        return True, message, topic_score
        
    except ObjectDoesNotExist:
        return False, f"アカウントID '{account_id}' のユーザーが見つかりません", None
    except Exception as e:
        return False, f"エラーが発生しました: {str(e)}", None


def get_user_topic_rankings(account_id, top_n=5):
    """
    指定されたユーザーのトピックスコアランキングを取得
    
    Args:
        account_id (str): ユーザーのアカウントID
        top_n (int): 上位何位まで取得するか
    
    Returns:
        tuple: (success: bool, rankings: list or error_message: str)
                rankings = [{'topic_label': str, 'score': float, 'count': int}, ...]
    """
    try:
        user = User.objects.get(account_id=account_id)
        
        topic_scores = UserTopicScore.objects.filter(
            user=user,
            score__gt=0  # スコアが0より大きいもののみ
        ).order_by('-score')[:top_n]
        
        rankings = []
        for topic_score in topic_scores:
            rankings.append({
                'topic_label': topic_score.topic_label,
                'score': topic_score.score,
                'count': topic_score.count,
                'updated_at': topic_score.updated_at
            })
        
        return True, rankings
        
    except ObjectDoesNotExist:
        return False, f"アカウントID '{account_id}' のユーザーが見つかりません"
    except Exception as e:
        return False, f"エラーが発生しました: {str(e)}"


def get_all_user_topic_matrix(account_id):
    """
    指定されたユーザーの全トピックスコアマトリックスを取得
    
    Args:
        account_id (str): ユーザーのアカウントID
    
    Returns:
        tuple: (success: bool, matrix: dict or error_message: str)
                matrix = {'社会': 0.0, 'まなび': 0.75, ...}
    """
    try:
        user = User.objects.get(account_id=account_id)
        matrix = UserTopicScore.get_user_topic_matrix(user)
        
        return True, matrix
        
    except ObjectDoesNotExist:
        return False, f"アカウントID '{account_id}' のユーザーが見つかりません"
    except Exception as e:
        return False, f"エラーが発生しました: {str(e)}"


def print_user_topic_summary(account_id):
    """
    指定されたユーザーのトピックスコア要約を表示
    
    Args:
        account_id (str): ユーザーのアカウントID
    """
    print(f"\n=== {account_id} のトピックスコア要約 ===")
    
    success, rankings = get_user_topic_rankings(account_id, top_n=10)
    if not success:
        print(f"エラー: {rankings}")
        return
    
    if not rankings:
        print("まだトピックスコアが記録されていません。")
        return
    
    print("【上位トピック】")
    for i, ranking in enumerate(rankings, 1):
        print(f"{i:2d}位: {ranking['topic_label']:12s} - {ranking['score']:.4f} (更新回数: {ranking['count']})")
    
    # 全体のマトリックス表示
    success, matrix = get_all_user_topic_matrix(account_id)
    if success:
        print("\n【全トピックスコア】")
        for topic, score in matrix.items():
            status = "★" if score > 0 else "　"
            print(f"{status} {topic:12s}: {score:.4f}") 