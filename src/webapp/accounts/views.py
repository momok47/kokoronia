from django.contrib.auth import login, authenticate
from django.views.generic import TemplateView, CreateView
from django.contrib.auth.views import LoginView as BaseLoginView,  LogoutView as BaseLogoutView
from django.urls import reverse_lazy
from django.contrib.auth.mixins import LoginRequiredMixin
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.views import View
from django.shortcuts import render, redirect
from django.contrib import messages
import json
import os
import sys
import time
import tempfile
import subprocess
import threading
import pyaudio
import re
from django.core.cache import cache
from django.conf import settings
from .forms import SignUpForm, LoginFrom
from .topic_suggestion_service import generate_next_topic_sentence

# デバッグモードの設定
DEBUG_RECORDING = getattr(settings, 'DEBUG_RECORDING', settings.DEBUG)

def debug_print(*args, **kwargs):
    """デバッグ出力のヘルパー関数"""
    if DEBUG_RECORDING:
        print(*args, **kwargs)

# プロジェクトルートをPATHに追加してcore機能をインポート
project_root = os.path.join(os.path.dirname(__file__), '..', '..', '..')
sys.path.insert(0, project_root)

try:
    from src.core.gcs.gcs_uploader import upload_to_gcs
    from src.core.gcs.transcribe_audio_from_gcs import transcribe_gcs
    from src.core.analysis.interests_extraction import analyze_transcription
    from .utils import save_user_insights
    CORE_FUNCTIONS_AVAILABLE = True
except ImportError as e:
    print(f"警告: コア機能のインポートに失敗しました: {e}")
    CORE_FUNCTIONS_AVAILABLE = False

# ユーザー登録とホームーページを実装するためのviewを定義
class IndexView(TemplateView):
    """ ホームビュー """
    template_name = "index.html"


class SignupView(CreateView):
    """ ユーザー登録用ビュー """
    form_class = SignUpForm # 作成した登録用フォームを設定
    template_name = "accounts/signup.html" 
    success_url = reverse_lazy("accounts:index") # ユーザー作成後のリダイレクト先ページ

    def form_valid(self, form):
        # ユーザー作成後にそのままログイン状態にする設定
        response = super().form_valid(form)
        account_id = form.cleaned_data.get("account_id")
        password = form.cleaned_data.get("password1")
        user = authenticate(account_id=account_id, password=password)
        login(self.request, user)
        return response

# ログインビューを作成
class LoginView(BaseLoginView):
    form_class = LoginFrom
    template_name = "accounts/login.html"


class LogoutView(BaseLogoutView):
    next_page = reverse_lazy("accounts:index")


class RecordingSessionView(LoginRequiredMixin, TemplateView):
    """ 用ビュー """
    template_name = "accounts/recording_session.html"
    login_url = "accounts:login"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['user'] = self.request.user
        return context


@method_decorator(csrf_exempt, name='dispatch')
class ProcessRecordingView(LoginRequiredMixin, View):
    """ 録音データ処理用ビュー """
    login_url = "accounts:login"
    
    def post(self, request, *args, **kwargs):
        try:
            if not CORE_FUNCTIONS_AVAILABLE:
                return JsonResponse({
                    'status': 'error',
                    'message': 'コア機能が利用できません。システム管理者に連絡してください。'
                }, status=500)
            
            # リクエストデータを取得
            participant1 = request.POST.get('participant1')  # 現在のユーザー
            participant2 = request.POST.get('participant2')  # 相手のユーザー
            session_name = request.POST.get('session_name', '未設定')
            audio_file = request.FILES.get('audio')
            
            if not audio_file:
                return JsonResponse({
                    'status': 'error',
                    'message': '音声ファイルがアップロードされていません。'
                }, status=400)
            
            # 相手ユーザーの存在確認
            if participant2:
                try:
                    from .models import User
                    partner_user = User.objects.get(account_id=participant2)
                except User.DoesNotExist:
                    return JsonResponse({
                        'status': 'error',
                        'message': f'参加者2のアカウントID "{participant2}" が見つかりません。'
                    }, status=400)
            
            debug_print(f"=== 音声処理開始 ===")
            debug_print(f"参加者1: {participant1}")
            debug_print(f"参加者2: {participant2}")
            debug_print(f"セッション名: {session_name}")
            debug_print(f"音声ファイルサイズ: {audio_file.size} バイト")
            
            # 音声ファイルを処理
            analysis_result1 = self._process_user_audio(
                audio_file, participant1, session_name
            )
            
            analysis_result2 = None
            if participant2:
                # 同じ音声ファイルを参加者2としても処理
                audio_file.seek(0)  # ファイルポジションをリセット
                analysis_result2 = self._process_user_audio(
                    audio_file, participant2, session_name
                )
            
            # レスポンスデータを作成
            response_data = {
                'status': 'success',
                'message': 'セッションが正常に完了しました',
                'session_name': session_name,
                'participant1': participant1,
                'participant2': participant2,
                'analysis': analysis_result1  # 参加者1の分析結果をフロントエンドに返す
            }
            
            return JsonResponse(response_data)
            
        except Exception as e:
            debug_print(f"処理エラー: {e}")
            return JsonResponse({
                'status': 'error',
                'message': f'処理中にエラーが発生しました: {str(e)}'
            }, status=500)
    
    def _process_user_audio(self, audio_file, account_id, session_name):
        """
        個別ユーザーの音声を処理する
        
        Args:
            audio_file: アップロードされた音声ファイル
            account_id: ユーザーのアカウントID
            session_name: セッション名
        
        Returns:
            dict: 分析結果
        """
        try:
            # GCS設定
            bucket_name = "kokoronia"
            timestamp = int(time.time())
            
            # ファイル名を生成
            original_extension = audio_file.name.split('.')[-1] if '.' in audio_file.name else 'webm'
            audio_filename = f"web_recording_{account_id}_{timestamp}.{original_extension}"
            audio_blob_name = f"media/audio/{audio_filename}"
            
            # 音声データを読み込み
            audio_file.seek(0)
            audio_data = audio_file.read()
            
            debug_print(f"--- {account_id} の音声処理開始 ---")
            
            # 1. GCSに音声ファイルをアップロード
            debug_print("音声ファイルをGCSにアップロード中...")
            gcs_uri = upload_to_gcs(
                bucket_name=bucket_name,
                data_content=audio_data,
                destination_blob_name=audio_blob_name,
                content_type=f"audio/{original_extension}"
            )
            
            if not gcs_uri:
                raise Exception("音声ファイルのGCSアップロードに失敗しました")
            
            debug_print(f"音声アップロード完了: {gcs_uri}")
            
            # 2. 文字起こしを実行
            debug_print("文字起こしを実行中...")
            transcription_data = transcribe_gcs(gcs_uri, account_id)
            
            if not transcription_data:
                raise Exception("文字起こしに失敗しました")
            
            debug_print("文字起こし完了")
            
            # 3. 文字起こし結果をGCSにアップロード
            transcription_json = json.dumps(transcription_data, ensure_ascii=False, indent=2)
            transcription_filename = f"web_transcription_{account_id}_{timestamp}.json"
            transcription_blob_name = f"media/transcriptions/{transcription_filename}"
            
            gcs_json_uri = upload_to_gcs(
                bucket_name=bucket_name,
                data_content=transcription_json,
                destination_blob_name=transcription_blob_name,
                content_type="application/json"
            )
            
            if not gcs_json_uri:
                raise Exception("結果のGCSアップロードに失敗しました")
            
            debug_print(f"文字起こし結果アップロード完了: {gcs_json_uri}")
            
            # 4. 興味分析を実行
            debug_print("興味分析を実行中...")
            analysis_result = analyze_transcription(
                transcription_blob_name, 
                speaker_tag_override=account_id
            )
            
            if not analysis_result:
                raise Exception("興味分析に失敗しました")
            
            debug_print(f"分析完了 - 検出トピック: {analysis_result.get('best_topic')}")
            
            # 5. データベースに保存
            debug_print("データベースに保存中...")
            success, message, topic_score = save_user_insights(account_id, analysis_result)
            
            if success:
                debug_print(f"データベース保存成功: {message}")
            else:
                debug_print(f"データベース保存警告: {message}")
            
            debug_print(f"--- {account_id} の処理完了 ---")
            
            return analysis_result
            
        except Exception as e:
            debug_print(f"{account_id} の音声処理中にエラーが発生しました: {e}")
            raise e


class ProgressAPIView(LoginRequiredMixin, View):
    """進行状況を取得するAPI"""
    login_url = "accounts:login"
    
    def get(self, request, *args, **kwargs):
        """現在の処理進行状況を取得"""
        session_id = request.GET.get('session_id')
        
        if not session_id:
            return JsonResponse({
                'status': 'idle',
                'current_step': '',
                'message': 'セッションが開始されていません',
                'progress': 0,
                'steps_completed': [],
                'analysis_results': None
            })
        
        progress_key = f"recording_progress_{session_id}"
        
        progress_data = cache.get(progress_key, {
            'status': 'idle',
            'current_step': '',
            'message': '処理待機中',
            'progress': 0,
            'steps_completed': [],
            'analysis_results': None
        })
        
        return JsonResponse(progress_data)


class RunMainScriptView(LoginRequiredMixin, View):
    """main.py スクリプト実行ビュー"""
    login_url = "accounts:login"
    
    def get(self, request, *args, **kwargs):
        # GETの場合はrecording画面にリダイレクト
        return redirect('accounts:recording')
    
    def post(self, request, *args, **kwargs):
        try:
            # フォームデータを取得
            device_a_index = request.POST.get('device_a_index')
            speaker_tag_a = request.POST.get('speaker_tag_a')
            device_b_index = request.POST.get('device_b_index')
            speaker_tag_b = request.POST.get('speaker_tag_b')
            
            # 入力検証
            if not all([device_a_index, speaker_tag_a, device_b_index, speaker_tag_b]):
                messages.error(request, '全ての項目を入力してください。')
                return redirect('accounts:recording')
            
            try:
                device_a_index = int(device_a_index)
                device_b_index = int(device_b_index)
            except ValueError:
                messages.error(request, 'デバイスIDは数値で入力してください。')
                return redirect('accounts:recording')
            
            if device_a_index == device_b_index:
                messages.error(request, '同じデバイスIDを選択することはできません。')
                return redirect('accounts:recording')
            
            # ユーザーの存在確認
            from .models import User
            try:
                User.objects.get(account_id=speaker_tag_a)
                User.objects.get(account_id=speaker_tag_b)
            except User.DoesNotExist:
                messages.error(request, '指定されたアカウントIDが見つかりません。')
                return redirect('accounts:recording')
            
            # main.pyを実行
            result = self.run_main_script(
                device_a_index, speaker_tag_a, 
                device_b_index, speaker_tag_b
            )
            
            context = {
                'script_output': result['output'],
                'script_error': result['error'],
                'return_code': result['return_code'],
                'execution_success': result['return_code'] == 0,
                'device_a_index': device_a_index,
                'speaker_tag_a': speaker_tag_a,
                'device_b_index': device_b_index,
                'speaker_tag_b': speaker_tag_b,
                'session_id': result.get('session_id'),
                'analysis_results': result.get('analysis_results'),
            }
            context['next_topic_suggestion'] = self.build_next_topic_suggestion(
                speaker_tag_a=speaker_tag_a,
                speaker_tag_b=speaker_tag_b,
            )
            
            return render(request, 'accounts/script_results.html', context)
            
        except Exception as e:
            messages.error(request, f'実行中にエラーが発生しました: {str(e)}')
            return redirect('accounts:recording')
    
    def run_main_script(self, device_a_index, speaker_tag_a, device_b_index, speaker_tag_b):
        """
        main.pyスクリプトを実行し、結果を返す
        
        Args:
            device_a_index: デバイス1のID
            speaker_tag_a: デバイス1のユーザー
            device_b_index: デバイス2のID  
            speaker_tag_b: デバイス2のユーザー
            
        Returns:
            dict: 実行結果 {'output': str, 'error': str, 'return_code': int, 'analysis_results': dict}
        """
        session_id = f"{device_a_index}_{device_b_index}_{int(time.time())}"
        progress_key = f"recording_progress_{session_id}"
        
        debug_print(f"=== テスト: セッション開始 ===")
        debug_print(f"Session ID: {session_id}")
        debug_print(f"Progress Key: {progress_key}")
        debug_print(f"入力パラメーター:")
        debug_print(f"  - デバイス1: {device_a_index} → {speaker_tag_a}")
        debug_print(f"  - デバイス2: {device_b_index} → {speaker_tag_b}")
        
        try:
            # main.py のパス（元の対話型版を非対話的に実行）
            project_root = os.path.join(os.path.dirname(__file__), '..', '..', '..')
            main_py_path = os.path.join(project_root, 'scripts', 'main.py')
            
            debug_print(f"=== 元のmain.pyを実行 ===")
            debug_print(f"Project Root: {project_root}")
            debug_print(f"Main.py Path: {main_py_path}")
            debug_print(f"Main.py Exists: {os.path.exists(main_py_path)}")
            
            if not os.path.exists(main_py_path):
                debug_print(f"❌ エラー: main.pyが見つかりません")
                self.update_progress(progress_key, "error", "main.pyが見つかりません", 0, [])
                return {
                    'output': '',
                    'error': f'main.pyが見つかりません: {main_py_path}',
                    'return_code': -1,
                    'session_id': session_id
                }
            
            # 環境変数の設定
            env = os.environ.copy()
            env['PYTHONPATH'] = project_root
            
            # 自動入力用の環境変数を設定（対話型入力をスキップ）
            env['KOKORONIA_AUTO_DEVICE_A'] = str(device_a_index)
            env['KOKORONIA_AUTO_SPEAKER_A'] = str(speaker_tag_a)
            env['KOKORONIA_AUTO_DEVICE_B'] = str(device_b_index)
            env['KOKORONIA_AUTO_SPEAKER_B'] = str(speaker_tag_b)
            env['KOKORONIA_AUTO_MODE'] = 'true'
            
            # GCS認証情報を確実に設定
            gcs_credentials = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
            if gcs_credentials:
                env['GOOGLE_APPLICATION_CREDENTIALS'] = gcs_credentials
            else:
                # フォールバック: .envから読み込み
                env_path = os.path.join(project_root, '.env')
                if os.path.exists(env_path):
                    with open(env_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            if line.startswith('GOOGLE_APPLICATION_CREDENTIALS='):
                                credential_path = line.split('=', 1)[1].strip()
                                env['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path
                                break
            
            # Pythonのパス
            python_executable = sys.executable
            
            debug_print(f"=== テスト: 実行環境 ===")
            debug_print(f"Python Executable: {python_executable}")
            debug_print(f"PYTHONPATH: {env.get('PYTHONPATH')}")
            debug_print(f"GOOGLE_APPLICATION_CREDENTIALS: {'設定済み' if env.get('GOOGLE_APPLICATION_CREDENTIALS') else '未設定'}")
            
            self.update_progress(progress_key, "starting", "音声データ処理を開始しています...", 10, [])
            debug_print(f"✅ 進行状況更新: 10% - 処理開始")
            
            debug_print(f"=== テスト: プロセス開始 ===")
            
            # main.pyを実行（環境変数で自動入力）
            cmd_args = [
                python_executable, main_py_path
            ]
            
            debug_print(f"実行コマンド: {' '.join(cmd_args)}")
            debug_print(f"環境変数:")
            debug_print(f"  KOKORONIA_AUTO_MODE: {env.get('KOKORONIA_AUTO_MODE')}")
            debug_print(f"  KOKORONIA_AUTO_DEVICE_A: {env.get('KOKORONIA_AUTO_DEVICE_A')}")
            debug_print(f"  KOKORONIA_AUTO_SPEAKER_A: {env.get('KOKORONIA_AUTO_SPEAKER_A')}")
            debug_print(f"  KOKORONIA_AUTO_DEVICE_B: {env.get('KOKORONIA_AUTO_DEVICE_B')}")
            debug_print(f"  KOKORONIA_AUTO_SPEAKER_B: {env.get('KOKORONIA_AUTO_SPEAKER_B')}")
            
            process = subprocess.Popen(
                cmd_args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                cwd=project_root
            )
            
            debug_print(f"プロセスPID: {process.pid}")
            
            # プロセス実行（タイムアウト付き）+ 進行状況監視
            try:
                # バックグラウンドで進行状況を監視
                monitor_thread = threading.Thread(
                    target=self.monitor_progress, 
                    args=(process, progress_key)
                )
                monitor_thread.daemon = True
                monitor_thread.start()
                debug_print(f"✅ 進行状況監視スレッド開始")
                
                debug_print(f"=== テスト: プロセス実行開始 ===")
                start_time = time.time()
                
                stdout, stderr = process.communicate(timeout=900)  # 15分タイムアウト（非対話型）
                return_code = process.returncode
                
                execution_time = time.time() - start_time
                debug_print(f"=== テスト: プロセス実行完了 ===")
                debug_print(f"実行時間: {execution_time:.2f}秒")
                debug_print(f"リターンコード: {return_code}")
                debug_print(f"標準出力サイズ: {len(stdout)}文字")
                debug_print(f"標準エラーサイズ: {len(stderr)}文字")
                
            except subprocess.TimeoutExpired:
                debug_print(f"❌ タイムアウトエラー: 15分経過")
                process.kill()
                stdout, stderr = process.communicate()
                return_code = -1
                stderr = "実行がタイムアウトしました (15分)" + "\n" + stderr
                self.update_progress(progress_key, "timeout", "処理がタイムアウトしました", 0, [])
            
            # 分析結果を抽出
            debug_print(f"=== テスト: 分析結果抽出 ===")
            analysis_results = self.extract_analysis_results(stdout)
            debug_print(f"分析結果: {analysis_results}")
            
            # 完了状況を更新
            if return_code == 0:
                debug_print(f"✅ 処理成功: 進行状況を100%に更新")
                self.update_progress(progress_key, "completed", "処理完了", 100, 
                    ["processing", "upload", "transcription", "analysis", "database"], 
                    analysis_results)
            else:
                debug_print(f"❌ 処理失敗: エラー状態に更新")
                self.update_progress(progress_key, "error", "処理エラー", 0, [], None)
            
            debug_print(f"=== テスト: 結果サマリー ===")
            debug_print(f"成功: {return_code == 0}")
            debug_print(f"セッションID: {session_id}")
            debug_print(f"分析結果あり: {analysis_results is not None}")
            
            return {
                'output': stdout,
                'error': stderr,
                'return_code': return_code,
                'session_id': session_id,
                'analysis_results': analysis_results,
                'debug_info': {
                    'execution_time': execution_time if 'execution_time' in locals() else 0,
                    'project_root': project_root,
                    'main_py_exists': os.path.exists(main_py_path),
                    'env_set': bool(env.get('GOOGLE_APPLICATION_CREDENTIALS'))
                }
            }
            
        except Exception as e:
            error_message = f"main.py実行中にエラーが発生: {e}"
            debug_print(f"❌ {error_message}")
            debug_print(f"エラータイプ: {type(e).__name__}")
            debug_print(f"エラー詳細: {str(e)}")
            
            self.update_progress(progress_key, "error", f"実行エラー: {str(e)}", 0, [])
            return {
                'output': '',
                'error': f'スクリプト実行エラー: {str(e)}',
                'return_code': -1,
                'session_id': session_id
            }
    
    def update_progress(self, progress_key, status, message, progress, steps_completed, analysis_results=None):
        """進行状況を更新"""
        progress_data = {
            'status': status,
            'current_step': message,
            'message': message,
            'progress': progress,
            'steps_completed': steps_completed,
            'analysis_results': analysis_results,
            'updated_at': time.time()
        }
        cache.set(progress_key, progress_data, 600)  # 10分間有効
        
    def monitor_progress(self, process, progress_key):
        """プロセスの出力を監視して進行状況を更新"""
        step_patterns = {
            r'音声データ(\(|の)': ('processing', '音声データ処理中...', 15),
            r'GCSにアップロード': ('upload', 'GCSアップロード中...', 35),
            r'文字起こし|transcribe|文字起こし　': ('transcription', '文字起こし中...', 65),
            r'関心度分析|興味分析|analyze': ('analysis', '関心度分析中...', 85),
            r'データベース|保存|save': ('database', 'データベース保存中...', 95),
        }
        
        steps_completed = []
        
        # 初期進行状況を設定
        self.update_progress(progress_key, "processing", "音声データ処理中...", 10, [])
        
        # プロセスが動いている間、出力を監視
        start_time = time.time()
        last_update_time = start_time
        
        while process.poll() is None:
            time.sleep(2)  # 2秒ごとにチェック
            
            current_time = time.time()
            elapsed = current_time - start_time
            
            # 最低でも5秒ごとに進行状況を更新
            if current_time - last_update_time >= 5:
                # 経過時間に基づく進行状況推測
                if elapsed < 30:  # 最初の30秒 - 音声データ処理
                    progress = min(25, 10 + int((elapsed / 30) * 15))
                    self.update_progress(progress_key, "processing", "音声データ処理中...", progress, ["processing"])
                elif elapsed < 60:  # 30-60秒 - アップロード
                    progress = min(40, 25 + int(((elapsed - 30) / 30) * 15))
                    self.update_progress(progress_key, "upload", "GCSアップロード中...", progress, ["processing", "upload"])
                elif elapsed < 120:  # 1-2分 - 文字起こし
                    progress = min(70, 40 + int(((elapsed - 60) / 60) * 30))
                    self.update_progress(progress_key, "transcription", "文字起こし中...", progress, ["processing", "upload", "transcription"])
                elif elapsed < 180:  # 2-3分 - 関心度分析
                    progress = min(90, 70 + int(((elapsed - 120) / 60) * 20))
                    self.update_progress(progress_key, "analysis", "関心度分析中...", progress, ["processing", "upload", "transcription", "analysis"])
                else:  # 3分以上 - データベース保存
                    progress = min(95, 90 + int(((elapsed - 180) / 60) * 5))
                    self.update_progress(progress_key, "database", "データベース保存中...", progress, ["processing", "upload", "transcription", "analysis", "database"])
                
                last_update_time = current_time
                debug_print(f"進行状況更新: {progress}% - {elapsed:.1f}秒経過")
        
        # プロセス終了時の最終更新
        if process.returncode == 0:
            self.update_progress(progress_key, "completed", "処理完了", 100, 
                ["processing", "upload", "transcription", "analysis", "database"])
            debug_print("進行状況更新: 100% - 処理完了")
        else:
            self.update_progress(progress_key, "error", "処理エラー", 0, [])
            debug_print("進行状況更新: 0% - 処理エラー")
    
    def extract_analysis_results(self, stdout):
        """標準出力から分析結果を抽出"""
        try:
            analysis_results = {
                'participants': [],
                'topics_detected': [],
                'summary': ''
            }
            
            # 検出されたトピックを抽出（main.pyの実際の出力パターンに合わせて修正）
            topic_patterns = [
                r'検出されたトピック:\s*([^\n]+)',
                r'トピックの信頼度:\s*([\d.]+)',
                r'--- (.+?) の分析を開始 ---',
                r'分析完了.*?検出トピック:\s*([^\n]+)',
                r'best_topic.*?:\s*["\']?([^"\']+)["\']?',
                r'ユーザー確認完了:\s*([^,\n]+),\s*([^,\n]+)'
            ]
            
            participants = []
            topics = []
            
            for pattern in topic_patterns:
                matches = re.findall(pattern, stdout, re.IGNORECASE)
                if 'の分析を開始' in pattern:
                    participants.extend(matches)
                elif 'ユーザー確認完了' in pattern:
                    # タプルの場合は展開
                    for match in matches:
                        if isinstance(match, tuple):
                            participants.extend([p.strip() for p in match])
                        else:
                            participants.append(match.strip())
                elif ('検出' in pattern or 'best_topic' in pattern):
                    topics.extend([t.strip() for t in matches if t.strip()])
            
            # 重複を除去
            analysis_results['participants'] = list(set([p for p in participants if p.strip()]))
            analysis_results['topics_detected'] = list(set([t for t in topics if t.strip()]))
            
            # サマリーを生成
            participant_count = len(analysis_results['participants'])
            topic_count = len(analysis_results['topics_detected'])
            analysis_results['summary'] = f"参加者{participant_count}名の音声を分析し、{topic_count}個のトピックを検出しました。"
            
            # デバッグ出力
            debug_print(f"抽出された参加者: {analysis_results['participants']}")
            debug_print(f"抽出されたトピック: {analysis_results['topics_detected']}")
            
            return analysis_results
            
        except Exception as e:
            debug_print(f"分析結果抽出エラー: {e}")
            return None

    def build_next_topic_suggestion(self, speaker_tag_a, speaker_tag_b):
        """DB上の興味関心スコアを使って、次回話題を1文だけ提案する。"""
        try:
            return generate_next_topic_sentence(speaker_tag_a, speaker_tag_b)
        except Exception as e:
            debug_print(f"話題提案生成エラー: {e}")
            return "次回は、二人が最近気になっているテーマを1つ選んで深掘りしてみましょう。"


class AudioDevicesAPIView(LoginRequiredMixin, View):
    """音声デバイス検出API"""
    login_url = "accounts:login"
    
    def get(self, request, *args, **kwargs):
        """利用可能な音声デバイス一覧を取得"""
        try:
            devices = self.get_audio_devices()
            return JsonResponse({
                'status': 'success',
                'devices': devices
            })
        except Exception as e:
            return JsonResponse({
                'status': 'error',
                'message': str(e),
                'devices': []
            }, status=500)
    
    def get_audio_devices(self):
        """
        pyaudioを使用して利用可能な音声入力デバイス一覧を取得
        
        Returns:
            list: デバイス情報のリスト
        """
        devices = []
        p = pyaudio.PyAudio()
        
        try:
            info = p.get_host_api_info_by_index(0)
            num_devices = info.get('deviceCount')
            
            for i in range(0, num_devices):
                device_info = p.get_device_info_by_host_api_device_index(0, i)
                
                # 入力チャンネルがあるデバイスのみを取得
                if device_info.get('maxInputChannels') > 0:
                    devices.append({
                        'id': i,
                        'name': device_info.get('name'),
                        'max_input_channels': device_info.get('maxInputChannels'),
                        'default_sample_rate': device_info.get('defaultSampleRate'),
                        'host_api': device_info.get('hostApi')
                    })
                    
        except Exception as e:
            debug_print(f"デバイス検出中にエラー: {e}")
        finally:
            p.terminate()
        
        return devices


@method_decorator(csrf_exempt, name='dispatch')
class ProcessDualRecordingView(LoginRequiredMixin, View):
    """ 2台同時録音データ処理用ビュー """
    login_url = "accounts:login"
    
    def post(self, request, *args, **kwargs):
        try:
            if not CORE_FUNCTIONS_AVAILABLE:
                return JsonResponse({
                    'status': 'error',
                    'message': 'コア機能が利用できません。システム管理者に連絡してください。'
                }, status=500)
            
            # リクエストデータを取得
            participant_a = request.POST.get('participant_a')  # 録音者（マイク1）
            participant_b = request.POST.get('participant_b')  # 会話相手（マイク2）
            session_name = request.POST.get('session_name', '未設定')
            microphone_a_name = request.POST.get('microphone_a_name', 'マイク1')
            microphone_b_name = request.POST.get('microphone_b_name', 'マイク2')
            
            audio_file_a = request.FILES.get('audio_a')  # マイク1の音声ファイル
            audio_file_b = request.FILES.get('audio_b')  # マイク2の音声ファイル
            
            if not audio_file_a or not audio_file_b:
                return JsonResponse({
                    'status': 'error',
                    'message': '2つの音声ファイルがアップロードされていません。'
                }, status=400)
            
            # ユーザーの存在確認
            from .models import User
            try:
                user_a = User.objects.get(account_id=participant_a)
                user_b = User.objects.get(account_id=participant_b)
            except User.DoesNotExist as e:
                return JsonResponse({
                    'status': 'error',
                    'message': f'指定されたアカウントIDが見つかりません。'
                }, status=400)
            
            debug_print(f"=== 2台同時音声処理開始 ===")
            debug_print(f"参加者A: {participant_a} (マイク: {microphone_a_name})")
            debug_print(f"参加者B: {participant_b} (マイク: {microphone_b_name})")
            debug_print(f"セッション名: {session_name}")
            debug_print(f"音声ファイルA サイズ: {audio_file_a.size} バイト")
            debug_print(f"音声ファイルB サイズ: {audio_file_b.size} バイト")
            
            # 2つの音声ファイルを並行処理
            analysis_result_a = self._process_user_audio(
                audio_file_a, participant_a, f"{session_name}_device_a"
            )
            
            # 2番目の音声ファイルを処理
            audio_file_b.seek(0)  # ファイルポジションをリセット
            analysis_result_b = self._process_user_audio(
                audio_file_b, participant_b, f"{session_name}_device_b"
            )

            def get_display_name(user):
                full_name = f"{(user.last_name or '').strip()} {(user.first_name or '').strip()}".strip()
                return full_name or user.account_id
            
            # レスポンスデータを作成
            response_data = {
                'status': 'success',
                'message': '＼分析成功／',
                'session_name': session_name,
                'participant_a': participant_a,
                'participant_b': participant_b,
                'microphone_a_name': microphone_a_name,
                'microphone_b_name': microphone_b_name,
                'analysis_a': analysis_result_a,
                'analysis_b': analysis_result_b,
                'summary': {
                    'participants': [participant_a, participant_b],
                    'participant_names': [
                        get_display_name(user_a),
                        get_display_name(user_b),
                    ],
                    'topics_detected': list(set([
                        analysis_result_a.get('best_topic', '不明'),
                        analysis_result_b.get('best_topic', '不明')
                    ])),
                    'devices_used': [microphone_a_name, microphone_b_name],
                    'total_files_processed': 2
                }
            }
            response_data['next_topic_suggestion'] = generate_next_topic_sentence(
                participant_a,
                participant_b,
            )
            
            return JsonResponse(response_data)
            
        except Exception as e:
            debug_print(f"2台同時処理エラー: {e}")
            return JsonResponse({
                'status': 'error',
                'message': f'処理中にエラーが発生しました: {str(e)}'
            }, status=500)
    
    def _process_user_audio(self, audio_file, account_id, session_name):
        """
        個別ユーザーの音声を処理する（ProcessRecordingViewと同じ処理）
        
        Args:
            audio_file: アップロードされた音声ファイル
            account_id: ユーザーのアカウントID
            session_name: セッション名
        
        Returns:
            dict: 分析結果
        """
        try:
            # GCS設定
            bucket_name = "kokoronia"
            timestamp = int(time.time())
            
            # ファイル名を生成
            original_extension = audio_file.name.split('.')[-1] if '.' in audio_file.name else 'webm'
            audio_filename = f"dual_recording_{account_id}_{timestamp}.{original_extension}"
            audio_blob_name = f"media/audio/{audio_filename}"
            
            # 音声データを読み込み
            audio_file.seek(0)
            audio_data = audio_file.read()
            
            debug_print(f"--- {account_id} の音声処理開始 ---")
            
            # 1. GCSに音声ファイルをアップロード
            debug_print("音声ファイルをGCSにアップロード中...")
            gcs_uri = upload_to_gcs(
                bucket_name=bucket_name,
                data_content=audio_data,
                destination_blob_name=audio_blob_name,
                content_type=f"audio/{original_extension}"
            )
            
            if not gcs_uri:
                raise Exception("音声ファイルのGCSアップロードに失敗しました")
            
            debug_print(f"音声アップロード完了: {gcs_uri}")
            
            # 2. 音文字起こしを実行
            debug_print("音声文字起こしを実行中...")
            transcription_data = transcribe_gcs(gcs_uri, account_id)
            
            if not transcription_data:
                raise Exception("文字こしに失敗しました")
            
            debug_print("文字起こし完了")
            
            # 3. 文字起こし結果をGCSにアップロード
            transcription_json = json.dumps(transcription_data, ensure_ascii=False, indent=2)
            transcription_filename = f"dual_transcription_{account_id}_{timestamp}.json"
            transcription_blob_name = f"media/transcriptions/{transcription_filename}"
            
            gcs_json_uri = upload_to_gcs(
                bucket_name=bucket_name,
                data_content=transcription_json,
                destination_blob_name=transcription_blob_name,
                content_type="application/json"
            )
            
            if not gcs_json_uri:
                raise Exception("結果のGCSアップロードに失敗しました")
            
            debug_print(f"文字起こし結果アップロード完了: {gcs_json_uri}")
            
            # 4. 興味分析を実行
            debug_print("興味分析を実行中...")
            analysis_result = analyze_transcription(
                transcription_blob_name, 
                speaker_tag_override=account_id
            )
            
            if not analysis_result:
                raise Exception("興味分析に失敗しました")
            
            debug_print(f"分析完了 - 検出トピック: {analysis_result.get('best_topic')}")
            
            # 5. データベースに保存
            debug_print("データベースに保存中...")
            success, message, topic_score = save_user_insights(account_id, analysis_result)
            
            if success:
                debug_print(f"データベース保存成功: {message}")
            else:
                debug_print(f"データベース保存警告: {message}")
            
            debug_print(f"--- {account_id} の処理完了 ---")
            
            return analysis_result
            
        except Exception as e:
            debug_print(f"{account_id} の音声処理中にエラーが発生しました: {e}")
            raise e