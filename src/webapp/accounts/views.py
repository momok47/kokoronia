from django.contrib.auth import login, authenticate
from django.views.generic import TemplateView, CreateView
from django.contrib.auth.views import LoginView as BaseLoginView,  LogoutView as BaseLogoutView
from django.urls import reverse_lazy
from django.contrib.auth.mixins import LoginRequiredMixin
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.views import View
import json
import os
import sys
import time
import tempfile
from .forms import SignUpForm, LoginFrom

# プロジェクトルートをPATHに追加してcore機能をインポート
project_root = os.path.join(os.path.dirname(__file__), '..', '..')
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
            
            print(f"=== 音声処理開始 ===")
            print(f"参加者1: {participant1}")
            print(f"参加者2: {participant2}")
            print(f"セッション名: {session_name}")
            print(f"音声ファイルサイズ: {audio_file.size} バイト")
            
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
            print(f"処理エラー: {e}")
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
            
            print(f"--- {account_id} の音声処理開始 ---")
            
            # 1. GCSに音声ファイルをアップロード
            print("音声ファイルをGCSにアップロード中...")
            gcs_uri = upload_to_gcs(
                bucket_name=bucket_name,
                data_content=audio_data,
                destination_blob_name=audio_blob_name,
                content_type=f"audio/{original_extension}"
            )
            
            if not gcs_uri:
                raise Exception("音声ファイルのGCSアップロードに失敗しました")
            
            print(f"音声アップロード完了: {gcs_uri}")
            
            # 2. 音声転写を実行
            print("音声転写を実行中...")
            transcription_data = transcribe_gcs(gcs_uri, account_id)
            
            if not transcription_data:
                raise Exception("音声転写に失敗しました")
            
            print("音声転写完了")
            
            # 3. 転写結果をGCSにアップロード
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
                raise Exception("転写結果のGCSアップロードに失敗しました")
            
            print(f"転写結果アップロード完了: {gcs_json_uri}")
            
            # 4. 興味分析を実行
            print("興味分析を実行中...")
            analysis_result = analyze_transcription(
                transcription_blob_name, 
                speaker_tag_override=account_id
            )
            
            if not analysis_result:
                raise Exception("興味分析に失敗しました")
            
            print(f"分析完了 - 検出トピック: {analysis_result.get('best_topic')}")
            
            # 5. データベースに保存
            print("データベースに保存中...")
            success, message, topic_score = save_user_insights(account_id, analysis_result)
            
            if success:
                print(f"データベース保存成功: {message}")
            else:
                print(f"データベース保存警告: {message}")
            
            print(f"--- {account_id} の処理完了 ---")
            
            return analysis_result
            
        except Exception as e:
            print(f"{account_id} の音声処理中にエラーが発生しました: {e}")
            raise e