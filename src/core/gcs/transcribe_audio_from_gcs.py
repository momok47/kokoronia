import os
import json
from google.cloud import speech_v1p1beta1 as speech

def transcribe_gcs(gcs_uri, user_speaker_tag="unknown"):
    """
    Args:
        gcs_uri (str): 文字起こしする音声ファイルのGCS URI (例: "gs://your-bucket/audio.wav")。
        user_speaker_tag (str, optional): 発言者の名前を示すタグ。デフォルトは"unknown"。

    Returns:
        dict: 文字起こしされたテキスト全体と、話者ごとの発言を含む辞書。
              形式: {'full_text': '文字起こし', 'speakers': {'話者名': [{'text': '発言', 'start_time': 0.0, 'end_time': 1.5}, ...]}, 'conversation': [...]}
              文字起こしに失敗した場合は None。
    """
    client = speech.SpeechClient()
    audio = speech.RecognitionAudio(uri=gcs_uri)

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=44100,
        language_code="ja-JP",
        enable_automatic_punctuation=True,
        enable_word_time_offsets=True,  # 単語レベルの時間情報を取得するために必要
        enable_speaker_diarization=False # 話者ダイアライゼーションを無効にする
    )

    print("文字起こし処理を開始しています...")
    try:
        operation = client.long_running_recognize(config=config, audio=audio)
        response = operation.result(timeout=300)
    except Exception as e:
        print(f"文字起こしAPI呼び出し中にエラーが発生しました: {e}")
        return None

    transcription_data = {
        "full_text": "",
        "speakers": {},
        "conversation": []
    }

    full_transcript_buffer = []
    all_words = [] # すべての単語情報を保持するリスト

    # すべてのResultからtranscriptとwordsを収集
    for i, result in enumerate(response.results):
        if result.alternatives:
            transcript = result.alternatives[0].transcript
            full_transcript_buffer.append(transcript)
            
            if hasattr(result.alternatives[0], 'words') and result.alternatives[0].words:
                words_in_this_result = len(result.alternatives[0].words)
                print(f"Result {i}: {words_in_this_result} words found")
                all_words.extend(result.alternatives[0].words)
            else:
                print(f"Result {i}: No words information available")
    
    transcription_data["full_text"] = " ".join(full_transcript_buffer).strip()
    print(f"Full text: '{transcription_data['full_text']}'")

    if not all_words:
        print("警告: 単語情報が取得できませんでした。full_textから基本情報を作成します。")
        # 単語情報がない場合でもfull_textから基本的な会話データを作成
        if transcription_data["full_text"]:
            if user_speaker_tag not in transcription_data["speakers"]:
                transcription_data["speakers"][user_speaker_tag] = []
            
            transcription_data["speakers"][user_speaker_tag].append({
                "text": transcription_data["full_text"],
                "start_time": 0.0,
                "end_time": 0.0  # 時間情報が取得できない場合は0
            })

            transcription_data["conversation"].append({
                "speaker": user_speaker_tag,
                "text": transcription_data["full_text"],
                "start_time": 0.0,
                "end_time": 0.0
            })
        return transcription_data

    # 単一話者のため、全ての単語をuser_speaker_tagに関連付ける
    if user_speaker_tag not in transcription_data["speakers"]:
        transcription_data["speakers"][user_speaker_tag] = []
    
    try:
        # すべての単語を結合して1つの発言として処理
        # 日本語の場合、単語をそのまま結合
        combined_text = "".join([word_info.word for word_info in all_words])
        first_word_time = all_words[0].start_time.total_seconds()
        last_word_time = all_words[-1].end_time.total_seconds()


        transcription_data["speakers"][user_speaker_tag].append({
            "text": combined_text.strip(),
            "start_time": first_word_time,
            "end_time": last_word_time
        })

        transcription_data["conversation"].append({
            "speaker": user_speaker_tag,
            "text": combined_text.strip(),
            "start_time": first_word_time,
            "end_time": last_word_time
        })
        
        print(f"Successfully created conversation data for {user_speaker_tag}")
        
    except Exception as e:
        print(f"単語データの処理中にエラーが発生しました: {e}")
        # フォールバックとしてfull_textを使用
        if transcription_data["full_text"]:
            transcription_data["speakers"][user_speaker_tag].append({
                "text": transcription_data["full_text"],
                "start_time": 0.0,
                "end_time": 0.0
            })

            transcription_data["conversation"].append({
                "speaker": user_speaker_tag,
                "text": transcription_data["full_text"],
                "start_time": 0.0,
                "end_time": 0.0
            })

    return transcription_data