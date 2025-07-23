import os
import sys
import pyaudio
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
from src.core.audio.device2_audio_recorder import record_dual_audio
from src.core.gcs.gcs_uploader import upload_to_gcs
from src.core.gcs.transcribe_audio_from_gcs import transcribe_gcs
from src.core.analysis.interests_extraction import analyze_transcription


class UserValidator:
    """ユーザー認証・検証を担当するクラス"""
    
    @staticmethod
    def validate_user_account(account_id: str) -> Tuple[bool, Optional[User], str]:
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

    @classmethod
    def get_valid_account_id(cls, device_name: str) -> str:
        """
        有効なアカウントIDを取得（再入力機能付き）
        
        Args:
            device_name (str): デバイス名（表示用）
        
        Returns:
            str: 有効なアカウントID
        """
        # 自動モードの場合はスキップ
        auto_mode = os.getenv('KOKORONIA_AUTO_MODE', '').lower() == 'true'
        if auto_mode:
            return ""  # 自動モードでは呼び出されない想定
            
        while True:
            account_id = input(f"{device_name}を使う人のアカウントIDを入力してください: ").strip()
            
            if not account_id:
                print("有効なアカウントIDを入力してください。")
                continue
            
            exists, user, message = cls.validate_user_account(account_id)
            print(message)
            
            if exists:
                print(f"{user.last_name}{user.first_name}さん　ようこそ！")
                return account_id
            else:
                print("\n登録済みユーザーか確認してください")
                
                retry = input("\n再入力しますか？ (y/n): ").strip().lower()
                if retry in ['n', 'no', 'いいえ']:
                    print("アプリケーションを終了します。")
                    sys.exit()
                print("─" * 50)


class DeviceManager:
    """音声デバイス管理を担当するクラス"""
    
    def __init__(self):
        self.input_devices: List[int] = []
        self.auto_mode = os.getenv('KOKORONIA_AUTO_MODE', '').lower() == 'true'
        
    def discover_audio_devices(self) -> List[int]:
        """利用可能な音声入力デバイスを検出"""
        p = pyaudio.PyAudio()
        try:
            info = p.get_host_api_info_by_index(0)
            num_devices = info.get('deviceCount')
            
            if not self.auto_mode:
                print("\n--- 利用可能な録音デバイス ---")
            input_devices = []
            for i in range(0, num_devices):
                if p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels') > 0:
                    device_name = p.get_device_info_by_host_api_device_index(0, i).get('name')
                    if not self.auto_mode:
                        print(f"デバイスID: {i} - {device_name}")
                    input_devices.append(i)
            
            self.input_devices = input_devices
            return input_devices
        finally:
            p.terminate()
    
    def validate_device_selection(self, device_index: int, already_selected: List[int] = None) -> bool:
        """デバイス選択の妥当性をチェック"""
        if already_selected is None:
            already_selected = []
            
        if device_index not in self.input_devices:
            if not self.auto_mode:
                print("エラー: 入力されたデバイスIDが無効です。")
            return False
        
        if device_index in already_selected:
            if not self.auto_mode:
                print("エラー: 同じデバイスを2回選択することはできません。")
            return False
        
        return True
    
    def get_device_selection(self) -> Tuple[int, str, int, str]:
        """デバイス選択とユーザー認証を実行"""
        
        # 自動モード（Web経由）の場合
        if self.auto_mode:
            device_a_index = int(os.getenv('KOKORONIA_AUTO_DEVICE_A'))
            speaker_tag_a = os.getenv('KOKORONIA_AUTO_SPEAKER_A')
            device_b_index = int(os.getenv('KOKORONIA_AUTO_DEVICE_B'))
            speaker_tag_b = os.getenv('KOKORONIA_AUTO_SPEAKER_B')
            
            print(f"自動モード: デバイス{device_a_index}({speaker_tag_a}) & デバイス{device_b_index}({speaker_tag_b})")
            
            # 有効性チェック
            if not self.validate_device_selection(device_a_index):
                raise ValueError(f"無効なデバイスID: {device_a_index}")
            if not self.validate_device_selection(device_b_index, [device_a_index]):
                raise ValueError(f"無効なデバイスID: {device_b_index}")
            
            # ユーザー存在チェック
            exists_a, _, msg_a = UserValidator.validate_user_account(speaker_tag_a)
            exists_b, _, msg_b = UserValidator.validate_user_account(speaker_tag_b)
            
            if not exists_a:
                raise ValueError(f"ユーザーが見つかりません: {speaker_tag_a}")
            if not exists_b:
                raise ValueError(f"ユーザーが見つかりません: {speaker_tag_b}")
                
            print(f"ユーザー確認完了: {speaker_tag_a}, {speaker_tag_b}")
            return device_a_index, speaker_tag_a, device_b_index, speaker_tag_b
        
        # 対話モード（通常実行）の場合
        # デバイス1台目
        while True:
            try:
                device_a_index = int(input("\nデバイス1台目のデバイスIDは？: "))
                if self.validate_device_selection(device_a_index):
                    break
            except ValueError:
                print("エラー: 無効な入力です。数値でデバイスIDを入力してください。")
        
        speaker_tag_a = UserValidator.get_valid_account_id("デバイス1台目")
        
        # デバイス2台目
        while True:
            try:
                device_b_index = int(input("\nデバイス2台目のデバイスIDは？: "))
                if self.validate_device_selection(device_b_index, [device_a_index]):
                    break
            except ValueError:
                print("エラー: 無効な入力です。数値でデバイスIDを入力してください。")
        
        speaker_tag_b = UserValidator.get_valid_account_id("デバイス2台目")
        
        return device_a_index, speaker_tag_a, device_b_index, speaker_tag_b


class AudioRecordingSession:
    """音声録音セッション全体を管理するクラス"""
    
    def __init__(self, bucket_name: str = "kokoronia"):
        self.bucket_name = bucket_name
        self.device_manager = DeviceManager()
        self._validate_environment()
    
    def _validate_environment(self):
        """環境設定の検証"""
        if "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
            raise EnvironmentError(
                "環境変数 GOOGLE_APPLICATION_CREDENTIALS が設定されていません。\n"
                "サービスアカウントキーのJSONファイルパスを設定してください。"
            )
    
    def setup_devices(self) -> Tuple[int, str, int, str]:
        """デバイスのセットアップ"""
        input_devices = self.device_manager.discover_audio_devices()
        
        if len(input_devices) < 2:
            raise RuntimeError(
                f"録音可能なデバイスが2つ以上ありません。デュアル録音は実行できません。\n"
                f"現在利用可能なデバイス数: {len(input_devices)}"
            )
        
        return self.device_manager.get_device_selection()
    
    def process_audio_data(self, wav_data: bytes, filename: str, speaker_tag: str) -> bool:
        """音声データの処理（アップロード→文字起こし→分析）"""
        print(f"\n--- {speaker_tag} の分析を開始 ---")
        
        # GCSにWAVファイルをアップロード
        gcs_uri = upload_to_gcs(
            self.bucket_name, 
            wav_data, 
            f"media/audio/{filename}", 
            content_type="audio/wav"
        )
        
        if not gcs_uri:
            print(f"音声データ({filename})のアップロードに失敗しました。")
            return False
        
        # 文字起こし実行
        transcription_data = transcribe_gcs(gcs_uri, speaker_tag)
        
        if not transcription_data:
            print(f"文字起こしに失敗しました: {gcs_uri}")
            return False
        
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
            print("文字起こしデータのアップロードに失敗しました。")
            return False
        
        print(f"文字起こしデータのアップロード完了： {gcs_json_uri}")
        
        # 関心度分析を実行
        analyze_transcription(transcription_blob_name, speaker_tag_override=speaker_tag)
        return True
    
    def run_session(self):
        """録音セッションの実行"""
        try:
            print("\n=== Welcome to KOKORONIA ===")
            
            # デバイスセットアップ
            device_a_index, speaker_tag_a, device_b_index, speaker_tag_b = self.setup_devices()
            
            print(f"\n^-^ 録音準備完了 ^-^")
            print(f"   デバイス1: {speaker_tag_a}")
            print(f"   デバイス2: {speaker_tag_b}")
            
            # 会話の録音
            wav_data_a, filename_a, wav_data_b, filename_b = record_dual_audio(
                device_a_index, device_b_index
            )
            
            if not (wav_data_a and wav_data_b):
                print("録音に失敗しました。")
                return
            
            # 音声データの処理（並行処理可能だが、シンプルにするために順次実行）
            success_a = self.process_audio_data(wav_data_a, filename_a, speaker_tag_a)
            success_b = self.process_audio_data(wav_data_b, filename_b, speaker_tag_b)
            
            if success_a and success_b:
                print("\n=== 全ての処理が完了しました ===")
            else:
                print("\n=== 一部の処理でエラーが発生しました ===")
                
        except (EnvironmentError, RuntimeError) as e:
            print(f"エラー: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"予期せぬエラーが発生しました: {e}")
            print("Google CloudのIAM設定、APIの有効化、GCSバケット名、環境変数などが正しいか確認してください。")


def main():
    """メイン関数"""
    session = AudioRecordingSession()
    session.run_session()


if __name__ == "__main__":
    main()