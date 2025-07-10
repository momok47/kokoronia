import os
import sys
import pyaudio
import json
from dotenv import load_dotenv

# プロジェクトルートをPYTHONPATHに追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Django設定の初期化
django_project_root = os.path.join(os.path.dirname(__file__), '..', 'src', 'webapp')
sys.path.insert(0, django_project_root)
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'project.settings')

import django
django.setup()

# Django初期化後にモデルをインポート
from accounts.models import User

# .envファイルを読み込み
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
from src.core.audio.device2_audio_recorder import record_dual_audio
from src.core.gcs.gcs_uploader import upload_to_gcs
from src.core.gcs.transcribe_audio_from_gcs import transcribe_gcs
from src.core.analysis.interests_extraction import analyze_transcription


def validate_user_account(account_id):
    """
    アカウントIDが存在するかチェック
    
    Args:
        account_id (str): チェックするアカウントID
    
    Returns:
        tuple: (exists: bool, user: User or None, message: str)
    """
    try:
        user = User.objects.get(account_id=account_id)
        return True, user, f"ユーザー '{account_id}' を確認しました"
    except User.DoesNotExist:
        return False, None, f"アカウントID '{account_id}' のユーザーが見つかりません"
    except Exception as e:
        return False, None, f"エラーが発生しました: {str(e)}"


def get_valid_account_id(device_name):
    """
    有効なアカウントIDを取得（再入力機能付き）
    
    Args:
        device_name (str): デバイス名（表示用）
    
    Returns:
        str: 有効なアカウントID
    """
    while True:
        account_id = input(f"{device_name}を使う人のアカウントIDを入力してください: ").strip()
        
        if not account_id:
            print("有効なアカウントIDを入力してください。")
            continue
        
        exists, user, message = validate_user_account(account_id)
        print(message)
        
        if exists:
            print(f"{user.last_name}{user.first_name}さん　ようこそ！")
            return account_id
        else:
            print("\n登録済みユーザーか確認してください")
            
            retry = input("\n再入力しますか？ (y/n): ").strip().lower()
            if retry in ['n', 'no', 'いいえ']:
                print("アプリケーションを終了します。")
                exit()
            print("─" * 50)


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
            if device_a_index not in input_devices:
                print("エラー: 入力されたデバイスIDが無効です。")
                exit()
            
            speaker_tag_a = get_valid_account_id("デバイス1台目")

            device_b_index = int(input("\nデバイス2台目のデバイスIDは？: "))
            if device_b_index not in input_devices:
                print("エラー: 入力されたデバイスIDが無効です。")
                exit()
            elif device_a_index == device_b_index:
                print("エラー: 同じデバイスを2回選択することはできません。")
                exit()
            
            speaker_tag_b = get_valid_account_id("デバイス2台目")
            
        except ValueError:
            print("エラー: 無効な入力です。数値でデバイスIDを入力してください。")
            exit()

        print(f"\n^-^ 録音準備完了 ^-^")
        print(f"   デバイス1: {speaker_tag_a}")
        print(f"   デバイス2: {speaker_tag_b}")

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