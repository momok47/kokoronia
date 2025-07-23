#!/usr/bin/env python3
"""
環境テストスクリプト
録音→分析→保存フローに必要な環境をテストします
"""

import os
import sys
import pyaudio
from pathlib import Path

# プロジェクトルートをPYTHONPATHに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src' / 'webapp'))

def test_python_environment():
    """Python環境のテスト"""
    print("=== Python環境テスト ===")
    print(f"✅ Python バージョン: {sys.version}")
    print(f"✅ 実行ファイル: {sys.executable}")
    return True

def test_audio_devices():
    """音声デバイスのテスト"""
    print("\n=== 音声デバイステスト ===")
    try:
        p = pyaudio.PyAudio()
        
        info = p.get_host_api_info_by_index(0)
        num_devices = info.get('deviceCount')
        
        print(f"総デバイス数: {num_devices}")
        
        input_devices = []
        for i in range(0, num_devices):
            device_info = p.get_device_info_by_host_api_device_index(0, i)
            if device_info.get('maxInputChannels') > 0:
                input_devices.append({
                    'id': i,
                    'name': device_info.get('name'),
                    'channels': device_info.get('maxInputChannels')
                })
                print(f"✅ デバイス {i}: {device_info.get('name')} ({device_info.get('maxInputChannels')} ch)")
        
        p.terminate()
        
        if len(input_devices) >= 2:
            print(f"✅ 録音可能デバイス: {len(input_devices)}個 (要求: 2個以上)")
            return True
        else:
            print(f"❌ 録音可能デバイス: {len(input_devices)}個 (要求: 2個以上)")
            return False
            
    except Exception as e:
        print(f"❌ エラー: {e}")
        return False

def test_file_paths():
    """必要なファイルパスのテスト"""
    print("\n=== ファイルパステスト ===")
    
    # main.pyの存在確認
    main_py = project_root / 'scripts' / 'main.py'
    if main_py.exists():
        print(f"✅ main.py: {main_py}")
        main_py_ok = True
    else:
        print(f"❌ main.py: {main_py} (見つかりません)")
        main_py_ok = False
    
    # Django設定の存在確認
    django_settings = project_root / 'src' / 'webapp' / 'project' / 'settings.py'
    if django_settings.exists():
        print(f"✅ Django設定: {django_settings}")
        django_ok = True
    else:
        print(f"❌ Django設定: {django_settings} (見つかりません)")
        django_ok = False
    
    return main_py_ok and django_ok

def test_environment_variables():
    """環境変数のテスト"""
    print("\n=== 環境変数テスト ===")
    
    # GCS認証情報
    gcs_creds = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
    if gcs_creds:
        if os.path.exists(gcs_creds):
            print(f"✅ GOOGLE_APPLICATION_CREDENTIALS: {gcs_creds}")
            gcs_ok = True
        else:
            print(f"❌ GOOGLE_APPLICATION_CREDENTIALS: {gcs_creds} (ファイルが見つかりません)")
            gcs_ok = False
    else:
        print("❌ GOOGLE_APPLICATION_CREDENTIALS: 未設定")
        gcs_ok = False
    
    return gcs_ok

def test_django_environment():
    """Django環境のテスト"""
    print("\n=== Django環境テスト ===")
    
    try:
        # Django設定初期化
        os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'project.settings')
        
        import django
        django.setup()
        
        print("✅ Django設定: OK")
        
        # データベース接続テスト
        from django.db import connection
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1")
        print("✅ データベース接続: OK")
        
        # テストユーザー確認
        from accounts.models import User
        test_users = User.objects.filter(account_id__in=['test_user1', 'test_user2'])
        user_count = test_users.count()
        
        if user_count >= 2:
            print(f"✅ テストユーザー: {user_count}名存在")
            print("  利用可能:", [u.account_id for u in test_users])
        else:
            print(f"⚠️  テストユーザー: {user_count}名のみ (推奨: 2名以上)")
            if user_count > 0:
                print("  存在するユーザー:", [u.account_id for u in test_users])
            
        return True
        
    except Exception as e:
        print(f"❌ Django環境エラー: {e}")
        return False

def test_imports():
    """必要なライブラリのインポートテスト"""
    print("\n=== ライブラリインポートテスト ===")
    
    required_modules = [
        'pyaudio',
        'google.cloud.storage',
        'google.cloud.speech',
        'transformers',
        'MeCab',
        'django',
        'pandas',
        'numpy'
    ]
    
    success_count = 0
    for module in required_modules:
        try:
            __import__(module)
            print(f"✅ {module}")
            success_count += 1
        except ImportError as e:
            print(f"❌ {module}: {e}")
    
    success_rate = success_count / len(required_modules) * 100
    print(f"インポート成功率: {success_rate:.1f}% ({success_count}/{len(required_modules)})")
    
    return success_rate >= 80  # 80%以上で成功とみなす

def main():
    """メインテスト実行"""
    print("🧪 KOKORONIA 環境テスト開始")
    print("=" * 50)
    
    tests = [
        ("Python環境", test_python_environment),
        ("音声デバイス", test_audio_devices),
        ("ファイルパス", test_file_paths),
        ("環境変数", test_environment_variables),
        ("ライブラリインポート", test_imports),
        ("Django環境", test_django_environment),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name}テスト中にエラー: {e}")
            results.append((test_name, False))
    
    # 結果サマリー
    print("\n" + "=" * 50)
    print("🎯 テスト結果サマリー")
    print("=" * 50)
    
    success_count = 0
    for test_name, result in results:
        status = "✅ 成功" if result else "❌ 失敗"
        print(f"{test_name:15}: {status}")
        if result:
            success_count += 1
    
    success_rate = success_count / len(results) * 100
    print(f"\n総合成功率: {success_rate:.1f}% ({success_count}/{len(results)})")
    
    if success_rate >= 80:
        print("\n🎉 環境テスト合格！Web録音フローを実行できます。")
        print("次のステップ:")
        print("1. cd src/webapp")
        print("2. python manage.py runserver")
        print("3. ブラウザで http://127.0.0.1:8000/ にアクセス")
        return 0
    else:
        print("\n⚠️  環境に問題があります。上記の失敗項目を修正してください。")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 