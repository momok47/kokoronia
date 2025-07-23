from django.urls import path
from . import views

app_name = "accounts"

urlpatterns = [
    path("", views.IndexView.as_view(), name="index"),
    path('signup/', views.SignupView.as_view(), name="signup"),
    path('login/', views.LoginView.as_view(), name="login"),
    path('logout/', views.LogoutView.as_view(), name="logout"),
    path('recording/', views.RecordingSessionView.as_view(), name="recording"),
    path('process-recording/', views.ProcessRecordingView.as_view(), name="process_recording"),
    path('process-dual-recording/', views.ProcessDualRecordingView.as_view(), name="process_dual_recording"),
    path('run-main-script/', views.RunMainScriptView.as_view(), name="run_main_script"),
    path('api/audio-devices/', views.AudioDevicesAPIView.as_view(), name="audio_devices_api"),
    path('api/progress/', views.ProgressAPIView.as_view(), name="progress_api"),
]