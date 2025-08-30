import os
from openai import OpenAI

# 🔐 APIキーの安全な管理
print("🔐 OpenAI APIキーを安全に設定します...")

# 方法1: 環境変数から取得（推奨）
api_key = os.getenv('OPENAI_API_KEY')

# 方法2: .envファイルから取得（オプション）
if not api_key:
    try:
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.getenv('OPENAI_API_KEY')
        print("📁 .envファイルから設定を読み込みました")
    except ImportError:
        print("💡 python-dotenvをインストールすると.envファイルが使用できます:")
        print("   pip install python-dotenv")


print("✅ APIキー設定完了")
# OpenAIクライアント初期化
client = OpenAI(api_key=api_key)

try:
    # GPT-4oを使用してレスポンス生成
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "user", "content": "武蔵野大学の位置を教えてください。"}
        ]
    )
    
    print("✅ GPT-4o レスポンス:")
    print(response.choices[0].message.content)
    
except Exception as e:
    print(f"❌ エラーが発生しました: {e}")
    print("🔧 APIキーまたは接続を確認してください。")
