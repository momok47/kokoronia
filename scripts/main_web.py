#!/usr/bin/env python3
"""
Webアプリ用非対話型main.py
コマンドライン引数でデバイスとユーザーを指定
"""

import os
import sys
import argparse
import json
from dotenv import load_dotenv
from typing import Optional, Tuple, List, Dict

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
from src.core.audio.device2_audio_recorder_noninteractive import record_dual_audio_noninteractive
from src.core.gcs.gcs_uploader import upload_to_gcs
from src.core.gcs.transcribe_audio_from_gcs import transcribe_gcs
from src.core.analysis.interests_extraction import analyze_transcription


class WebAudioRecordingSession:
    """Web用音声録音セッション（非対話型）"""
    
    def __init__(self, bucket_name: str = "kokoronia"):
        self.bucket_name = bucket_name
        self._validate_environment()
    
    def _validate_environment(self):
        """環境設定の検証"""
        if "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
            raise EnvironmentError(
                "環境変数 GOOGLE_APPLICATION_CREDENTIALS が設定されていません。\n"
                "サービスアカウントキーのJSONファイルパスを設定してください。"
            )
    
    def validate_user(self, account_id: str) -> bool:
        """ユーザー存在確認"""
        try:
            user = User.objects.get(account_id=account_id)
            print(f"✅ ユーザー確認: {user.last_name} {user.first_name}さん")
            return True
        except User.DoesNotExist:
            print(f"❌ ユーザーが見つかりません: {account_id}")
            return False
    
    def process_audio_data(self, wav_data: bytes, filename: str, speaker_tag: str) -> bool:
        """音声データの処理（アップロード→文字起こし→分析）"""
        print(f"\n--- {speaker_tag} の分析を開始 ---")
        
        # GCSにWAVファイルをアップロード
        print("音声データをGCSにアップロード中...")
        gcs_uri = upload_to_gcs(
            self.bucket_name, 
            wav_data, 
            f"media/audio/{filename}", 
            content_type="audio/wav"
        )
        
        if not gcs_uri:
            print(f"❌ 音声データ({filename})のアップロードに失敗しました。")
            return False
        
        print(f"✅ 音声アップロード完了: {gcs_uri}")
        
        # 文字起こし実行
        print("文字こしを実行中...")
        transcription_data = transcribe_gcs(gcs_uri, speaker_tag)
        
        if not transcription_data:
            print(f"❌ 文字起こしに失敗しました: {gcs_uri}")
            return False
        
        print("✅ 文字起こし完了")
        
        # 文字起こし結果JSONをGCSにアップロード
        transcription_json = json.dumps(transcription_data, ensure_ascii=False, indent=2)
        transcription_base_name = os.path.splitext(filename)[0]
        transcription_blob_name = f"media/transcriptions/{transcription_base_name}.json"
        
        gcs_json_uri = upload_to_gcs(
            self.bucket_name, 
            transcription_json, 
            transcription_blob_name, 
            content_type="application/json"
        )
        
        if not gcs_json_uri:
            print("❌ 文字起こしデータのアップロードに失敗しました。")
            return False
        
        print(f"✅ 文字起こし結果アップロード完了: {gcs_json_uri}")
        
        # 関心度分析を実行
        print("関心度分析を実行中...")
        try:
            analyze_transcription(transcription_blob_name, speaker_tag_override=speaker_tag)
            print("✅ 関心度分析完了")
        except Exception as e:
            print(f"❌ 関心度分析エラー: {e}")
            return False
        
        print("✅ データベース保存完了")
        return True
    
    def run_session(self, device_a_index: int, speaker_tag_a: str, device_b_index: int, speaker_tag_b: str):
        """録音セッションの実行（非対話型）"""
        try:
            print(f"\n=== KOKORONIA Web Session ===")
            print(f"デバイス1 (ID: {device_a_index}) → {speaker_tag_a}")
            print(f"デバイス2 (ID: {device_b_index}) → {speaker_tag_b}")
            
            # ユーザー存在確認
            if not self.validate_user(speaker_tag_a):
                raise ValueError(f"ユーザーが見つかりません: {speaker_tag_a}")
            if not self.validate_user(speaker_tag_b):
                raise ValueError(f"ユーザーが見つかりません: {speaker_tag_b}")
            
            print(f"\n🎙️ デュアル録音を開始...")
            
            # 会話の録音（非対話型）
            wav_data_a, filename_a, wav_data_b, filename_b = record_dual_audio_noninteractive(
                device_a_index, device_b_index, duration_seconds=60  # 60秒録音（実用的な時間）
            )
            
            if not (wav_data_a and wav_data_b):
                print("❌ 録音に失敗しました。")
                return False
            
            print(f"✅ 録音完了: {len(wav_data_a)} bytes, {len(wav_data_b)} bytes")
            
            # 音声データの処理（並行処理可能だが、シンプルにするために順次実行）
            print("\n📊 分析処理を開始...")
            success_a = self.process_audio_data(wav_data_a, filename_a, speaker_tag_a)
            success_b = self.process_audio_data(wav_data_b, filename_b, speaker_tag_b)
            
            if success_a and success_b:
                print("\n🎉 全ての処理が完了しました")
                return True
            else:
                print("\n⚠️ 一部の処理でエラーが発生しました")
                return False
                
        except (EnvironmentError, ValueError) as e:
            print(f"❌ エラー: {e}")
            return False
        except Exception as e:
            print(f"❌ 予期せぬエラーが発生しました: {e}")
            print("Google CloudのIAM設定、APIの有効化、GCSバケット名、環境変数などが正しいか確認してください。")
            return False


def main():
    """メイン実行関数"""
    parser = argparse.ArgumentParser(description='KOKORONIA Web録音セッション')
    parser.add_argument('device_a', type=int, help='デバイス1のID')
    parser.add_argument('speaker_a', type=str, help='デバイス1のユーザー')
    parser.add_argument('device_b', type=int, help='デバイス2のID')
    parser.add_argument('speaker_b', type=str, help='デバイス2のユーザー')
    parser.add_argument('--bucket', default='kokoronia', help='GCSバケット名')
    
    args = parser.parse_args()
    
    # セッション実行
    session = WebAudioRecordingSession(bucket_name=args.bucket)
    success = session.run_session(
        device_a_index=args.device_a,
        speaker_tag_a=args.speaker_a,
        device_b_index=args.device_b,
        speaker_tag_b=args.speaker_b
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main() 