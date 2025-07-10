import os
import json
import re
from google.cloud import speech_v1p1beta1 as speech

def clean_japanese_text(text):
    """
    音素記号形式のテキストをクリーンアップする
    例: "佐藤|サトー" -> "佐藤"
    """
    if not text:
        return ""
    
    # 音素記号（|記号で区切られた読み情報）を除去
    # パターン: "漢字|ヨミ" -> "漢字"
    cleaned = re.sub(r'([^\|]+)\|[^\|]*', r'\1', text)
    
    # 残った単独の|記号を除去
    cleaned = re.sub(r'\|+', '', cleaned)
    
    # 複数のスペースを1つに統一
    cleaned = re.sub(r'\s+', ' ', cleaned)
    
    return cleaned.strip()

def transcribe_gcs(gcs_uri, user_speaker_tag="unknown"):
    """
    Args:
        gcs_uri (str): 文字起こしする音声ファイルのGCS URI
        user_speaker_tag (str, optional): 発言者の名前を示すタグ

    Returns:
        dict: 最適化された文字起こし結果
              形式: {
                  'conversation': [{'speaker': '話者名', 'text': '発言', 'start_time': 0.0, 'end_time': 1.5}],
                  'full_text': '全体テキスト'
              }
    """
    client = speech.SpeechClient()
    audio = speech.RecognitionAudio(uri=gcs_uri)

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=44100,
        language_code="ja-JP",
        enable_automatic_punctuation=True,
        enable_word_time_offsets=True,
        enable_speaker_diarization=False
    )

    print("文字起こし処理を開始しています...")
    try:
        operation = client.long_running_recognize(config=config, audio=audio)
        response = operation.result(timeout=300)
    except Exception as e:
        print(f"文字起こしAPI呼び出し中にエラーが発生しました: {e}")
        return None

    # 文字起こし結果の収集
    full_transcript_buffer = []
    all_words = []

    for i, result in enumerate(response.results):
        if result.alternatives:
            transcript = result.alternatives[0].transcript
            full_transcript_buffer.append(transcript)
            
            if hasattr(result.alternatives[0], 'words') and result.alternatives[0].words:
                all_words.extend(result.alternatives[0].words)

    # テキストのクリーンアップ
    raw_full_text = " ".join(full_transcript_buffer).strip()
    cleaned_full_text = clean_japanese_text(raw_full_text)
    
    print(f"認識されたテキスト: '{cleaned_full_text}'")

    # 最適化されたデータ構造
    transcription_data = {
        "conversation": [],
        "full_text": cleaned_full_text
    }

    if all_words:
        try:
            # 単語レベルのテキストをクリーンアップ
            raw_combined_text = "".join([word_info.word for word_info in all_words])
            cleaned_combined_text = clean_japanese_text(raw_combined_text)
            
            first_word_time = all_words[0].start_time.total_seconds()
            last_word_time = all_words[-1].end_time.total_seconds()

            transcription_data["conversation"].append({
                "speaker": user_speaker_tag,
                "text": cleaned_combined_text,
                "start_time": first_word_time,
                "end_time": last_word_time
            })
            
        except Exception as e:
            print(f"単語データの処理中にエラーが発生しました: {e}")
            # フォールバック: full_textを使用
            if cleaned_full_text:
                transcription_data["conversation"].append({
                    "speaker": user_speaker_tag,
                    "text": cleaned_full_text,
                    "start_time": 0.0,
                    "end_time": 0.0
                })
    else:
        # 単語情報がない場合のフォールバック
        print("警告: 単語情報が取得できませんでした。")
        if cleaned_full_text:
            transcription_data["conversation"].append({
                "speaker": user_speaker_tag,
                "text": cleaned_full_text,
                "start_time": 0.0,
                "end_time": 0.0
            })

    return transcription_data