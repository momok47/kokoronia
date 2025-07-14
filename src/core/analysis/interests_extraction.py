import pandas as pd
import MeCab
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import json
import sys
import os
from dotenv import load_dotenv
from .zero_shot_learning import ZeroShotLearning
from google.cloud import storage

# Django設定の初期化（分析スクリプト単体実行時）
if not hasattr(sys, '_django_setup_done'):
    django_project_root = os.path.join(os.path.dirname(__file__), '..', '..', 'webapp')
    sys.path.insert(0, django_project_root)
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'project.settings')
    
    import django
    django.setup()
    sys._django_setup_done = True

# Django関連のインポート（setup後に実行）
try:
    from accounts.utils import save_user_insights, print_user_topic_summary # type: ignore
    DJANGO_AVAILABLE = True
except ImportError:
    DJANGO_AVAILABLE = False
    print("警告: Django環境が利用できません。分析結果はデータベースに保存されません。")

# .envファイルを読み込み
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', '..', '.env'))

GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "kokoronia")

def get_transcription_content(blob_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(GCS_BUCKET_NAME)
    blob = bucket.blob(blob_name)
    
    content = json.loads(blob.download_as_string().decode('utf-8'))
    return content

def analyze_transcription(transcription_blob_name, speaker_tag_override=None):
    """
    Args:
        transcription_blob_name (str): 分析対象の文字起こしファイルのGCSブロブ名
        speaker_tag_override (str, optional): 話者タグの上書き設定。Noneの場合は元の話者タグを使用
        
    Returns:
        dict: 分析結果を含む辞書。失敗時はNone
    """
    try:
        content = get_transcription_content(transcription_blob_name)

        print("\n=== 関心度分析を開始します ===")
        
        # 日本語モデルを試す場合（コメントアウト）
        # model_name = "cl-tohoku/bert-base-japanese-whole-word-masking"
        
        # 多言語モデルを使用（日本語テキストに最適化）
        model_name = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
        
        # 改善されたラベル（高スコア獲得のため6個に削減）
        japanese_labels = ["社会", "まなび", "テクノロジー", "カルチャー", "アウトドア", "フード", 
                          "旅行おでかけ", "ライフスタイル", "ビジネス", "読書", "キャリア", 
                          "デザイン", "IT", "経済投資", "ネットワーク"]
        
        english_labels = ["society", "learning", "technology", "culture", "outdoor", "food", 
                         "travel", "lifestyle", "business", "reading", "career", 
                         "design", "IT", "economics", "network"]
        
        # ラベルマッピング辞書
        label_mapping = dict(zip(english_labels, japanese_labels))
        
        # 使用するラベルを決定
        if "japanese" in model_name or "tohoku" in model_name:
            topic_labels = japanese_labels
        else:
            topic_labels = english_labels
        
        unidic_path = os.getenv("UNIDIC_PATH", '/Users/shirakawamomoko/Desktop/electronic_dictionary/unidic-csj-202302')
        
        # ZeroShotLearningクラスの初期化（新しいAPI）
        topic_analyzer = ZeroShotLearning(
            model_name=model_name,
            unidic_path=unidic_path
        )
        
        # 会話データの準備（full_textとconversationの両方に対応）
        conversation_data = content.get("conversation", content["full_text"])
        
        # 関心度分析の実行
        insights = topic_analyzer.extract_insights(
            conversation_data=conversation_data,
            topic_labels=topic_labels,
            display_speaker_label=speaker_tag_override
        )
        
        # 結果を日本語ラベルに変換（英語モデルを使用した場合）
        if not ("japanese" in model_name or "tohoku" in model_name):
            # 英語ラベルを日本語に変換
            if insights['best_topic'] in label_mapping:
                insights['best_topic'] = label_mapping[insights['best_topic']]
            
            # topic_scoresも変換
            converted_topic_scores = {}
            for eng_label, score in insights['topic_scores'].items():
                jp_label = label_mapping.get(eng_label, eng_label)
                converted_topic_scores[jp_label] = score
            insights['topic_scores'] = converted_topic_scores
        
        print("\n=== 分析結果 ===")
        print(f"検出されたトピック: {insights['best_topic']}")
        print(f"トピックの信頼度: {insights['best_score']:.4f}")
        
        # データベースに保存（Django環境が利用可能な場合）
        if DJANGO_AVAILABLE and speaker_tag_override:
            print("\n=== データベース保存 ===")
            success, message, topic_score = save_user_insights(speaker_tag_override, insights)
            print(message)
            
            if success:
                """
                # ユーザーのトピックスコア要約を表示
                print_user_topic_summary(speaker_tag_override)
                """
        
        return insights
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        print("関心度分析中にエラーが発生しました。モデルのロード、MeCabの設定、または外部ライブラリのインストールを確認してください。")
        return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("エラー: 文字起こしファイル名が指定されていません。")
        print("使用法: python interests_extraction.py <transcription_file_name> [speaker_tag_override]")
        exit()

    transcription_file_name = sys.argv[1]
    transcription_blob_name = f"media/transcriptions/{transcription_file_name}" 
    # このコード単体で回す時用の引数
    test_speaker_tag_override = sys.argv[2] if len(sys.argv) > 2 else None

    if "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
        print("エラー: 環境変数 GOOGLE_APPLICATION_CREDENTIALS が設定されていません。")
        print("サービスアカウントキーのJSONファイルパスを設定してください。")
        exit()

    try:
        analyze_transcription(transcription_blob_name, test_speaker_tag_override)
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        print("Google CloudのIAM設定、APIの有効化、GCSバケット名、環境変数などが正しいか確認してください。")