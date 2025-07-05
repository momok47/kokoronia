import os
import sys
import pyaudio
import json

# プロジェクトルートをPYTHONPATHに追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.audio.device2_audio_recorder import record_dual_audio
from src.core.gcs.gcs_uploader import upload_to_gcs
from src.core.gcs.transcribe_audio_from_gcs import transcribe_gcs
from src.core.analysis.interests_extraction import analyze_transcription

def main():
    # GCSの設定
    bucket_name = "kokoronia"
    
    if "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
        print("エラー: 環境変数 GOOGLE_APPLICATION_CREDENTIALS が設定されていません。")
        print("サービスアカウントキーのJSONファイルパスを設定してください。")
        exit()

    try:
        print("\n=== Welcome to KOKORONIA ===")
        
        p = pyaudio.PyAudio()
        info = p.get_host_api_info_by_index(0)
        num_devices = info.get('deviceCount') 
        
        print("\n--- 利用可能な録音デバイス ---")
        input_devices = []
        for i in range(0, num_devices):
            if p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels') > 0:
                device_name = p.get_device_info_by_host_api_device_index(0, i).get('name')
                print(f"デバイスID: {i} - {device_name}")
                input_devices.append(i)
        
        p.terminate()

        if len(input_devices) < 2:
            print("\nエラー: 録音可能なデバイスが2つ以上ありません。デュアル録音は実行できません。")
            exit()

        try:
            device_a_index = int(input("\nデバイス1台目のデバイスIDは？: "))
            speaker_tag_a = input("このデバイスを使う人のアカウントIDを入力してください: ")

            device_b_index = int(input("\nデバイス2台目のデバイスIDは？: "))
            speaker_tag_b = input("このデバイスを使う人のアカウントIDを入力してください: ")

            if device_a_index not in input_devices or device_b_index not in input_devices:
                print("エラー: 入力されたデバイスIDが無効です。リストから存在するIDを選択してください。")
                exit()
            elif device_a_index == device_b_index:
                print("エラー: 同じデバイスを2回選択することはできません。")
                exit()
        except ValueError:
            print("エラー: 無効な入力です。数値でデバイスIDを入力してください。")
            exit()

        # 会話の録音
        wav_data_a, filename_a, wav_data_b, filename_b = record_dual_audio(device_a_index, device_b_index)
        
        if wav_data_a and wav_data_b:
            # GCSにWAVファイルをアップロード (デバイスA)
            print(f"\n--- {speaker_tag_a} の分析を開始 ---")
            gcs_uri_a = upload_to_gcs(bucket_name, wav_data_a, f"media/audio/{filename_a}", content_type="audio/wav")
            
            if gcs_uri_a:
                transcription_data_a = transcribe_gcs(gcs_uri_a, speaker_tag_a)
                
                if transcription_data_a:
                    # 文字起こし結果JSONをGCSにアップロード
                    transcription_json_a = json.dumps(transcription_data_a, ensure_ascii=False, indent=2)
                    transcription_base_name_a = os.path.splitext(filename_a)[0]
                    transcription_blob_name_a = f"media/transcriptions/{transcription_base_name_a}.json"
                    
                    gcs_json_uri_a = upload_to_gcs(bucket_name, transcription_json_a, transcription_blob_name_a, content_type="application/json")
                    
                    if gcs_json_uri_a:
                        print(f"文字起こしデータのアップロード完了： {gcs_json_uri_a}")
                        # jsonとspeaker_tag_overrideを渡す
                        analyze_transcription(transcription_blob_name_a, speaker_tag_override=speaker_tag_a) 
                    else:
                        print("文字起こしデータのアップロードに失敗しました。")
                else:
                    print(f"文字起こしに失敗しました: {gcs_uri_a}")
            else:
                print(f"音声データ({filename_a})のアップロードに失敗しました。")

            # GCSにWAVファイルをアップロード (デバイスB)
            print(f"\n--- {speaker_tag_b} の分析を開始 ---")
            gcs_uri_b = upload_to_gcs(bucket_name, wav_data_b, f"media/audio/{filename_b}", content_type="audio/wav")
            
            if gcs_uri_b:
                transcription_data_b = transcribe_gcs(gcs_uri_b, speaker_tag_b)
                
                if transcription_data_b:
                    # 文字起こし結果JSONをGCSにアップロード
                    transcription_json_b = json.dumps(transcription_data_b, ensure_ascii=False, indent=2)
                    transcription_base_name_b = os.path.splitext(filename_b)[0]
                    transcription_blob_name_b = f"media/transcriptions/{transcription_base_name_b}.json"
                    
                    gcs_json_uri_b = upload_to_gcs(bucket_name, transcription_json_b, transcription_blob_name_b, content_type="application/json")
                    
                    if gcs_json_uri_b:
                        print(f"文字起こしデータのアップロード完了： {gcs_json_uri_b}")
                        analyze_transcription(transcription_blob_name_b, speaker_tag_override=speaker_tag_b)
                    else:
                        print("文字起こしデータのアップロードに失敗しました。")
                else:
                    print(f"文字起こしに失敗しました: {gcs_uri_b}")
            else:
                print(f"音声データ({filename_b})のアップロードに失敗しました。")
        else:
            print("録音に失敗しました。")
        
    except Exception as e:
        print(f"予期せぬエラーが発生しました: {e}")
        print("Google CloudのIAM設定、APIの有効化、GCSバケット名、環境変数などが正しいか確認してください。")

if __name__ == "__main__":
    main()