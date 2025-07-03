import pandas as pd
import MeCab
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import json
import sys
import os
from zero_shot_learning import InterestsExtraction
from google.cloud import storage

GCS_BUCKET_NAME = "kokoronia"

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
        None: 関数は結果を出力するのみで戻り値なし
    """
    try:
        content = get_transcription_content(transcription_blob_name)
        """
        print("\n=== 会話の時系列 ===")
        sorted_utterances = sorted(content["conversation"], key=lambda x: x["start_time"])
        for utterance in sorted_utterances:
            start_time = int(utterance["start_time"])
            end_time = int(utterance["end_time"])
            
            start_min = start_time // 60
            start_sec = start_time % 60
            end_min = end_time // 60
            end_sec = end_time % 60
            
            text = utterance["text"].replace("  ", " ").strip()
            # JSONから取得したspeakerタグをそのまま表示
            print(f"{utterance['speaker']}: [{start_min}分{start_sec}秒 - {end_min}分{end_sec}秒] {text}")
        """

        print("\n=== 関心度分析を開始します ===")
        model_name = "cl-tohoku/bert-base-japanese-whole-word-masking" 
        topic_labels = ["社会", "まなび", "テクノロジー", "カルチャー", "アウトドア", "フード", "旅行おでかけ",
                         "ライフスタイル", "ビジネス", "読書", "キャリア", "デザイン", "IT", "経済投資","ネットワーク"]
        mecab_dic_path = '/Users/shirakawamomoko/Desktop/electronic_dictionary/unidic-csj-202302'
        
        topic_analyzer = InterestsExtraction(
            conversation_data=content["full_text"], 
            model_name=model_name,
            topic_labels=topic_labels,
            mecab_dic_path=mecab_dic_path
        )
        
        # zero-shot-learning を実行
        topic_analyzer.main_function(display_speaker_label=speaker_tag_override) 
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        print("関心度分析中にエラーが発生しました。モデルのロード、MeCabの設定、または外部ライブラリのインストールを確認してください。")

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