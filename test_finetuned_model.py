#!/usr/bin/env python3
"""
ファインチューニングされたGPT-4o miniモデルのテストスクリプト
モデルID: ft:gpt-4o-mini-2024-07-18:personal::CAJ6PxFB
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
import json

def load_environment():
    """環境変数を読み込み"""
    project_root = Path(__file__).parent
    env_path = project_root / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print(f"✅ .envファイルを読み込みました: {env_path}")
    else:
        print(f"❌ .envファイルが見つかりません: {env_path}")
        return False
    
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEYが設定されていません")
        return False
    
    return True

def test_finetuned_model():
    """ファインチューニングされたモデルをテスト"""
    
    if not load_environment():
        return
    
    # OpenAI クライアント初期化
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # ファインチューニングされたモデルID
    model_id = "ft:gpt-4o-mini-2024-07-18:personal::CAJ6PxFB"
    
    print(f"\n🤖 ファインチューニングモデルをテスト中...")
    print(f"モデルID: {model_id}")
    print("=" * 60)
    
    # テスト用の相談シナリオ
    test_scenarios = [
        {
            "name": "仕事のストレス相談",
            "messages": [
                {"role": "user", "content": "こんにちは。最近仕事でとてもストレスを感じていて、相談したいことがあります。"},
                {"role": "assistant", "content": "こんにちは。お忙しい中、相談にいらしていただきありがとうございます。仕事でストレスを感じていらっしゃるのですね。どのような状況でストレスを感じているか、お話しいただけますか？"},
                {"role": "user", "content": "上司との関係がうまくいかなくて、毎日会社に行くのが辛いです。"}
            ]
        },
        {
            "name": "人間関係の悩み",
            "messages": [
                {"role": "user", "content": "友達との関係で悩んでいます。どうしたらいいでしょうか？"}
            ]
        },
        {
            "name": "自信喪失の相談",
            "messages": [
                {"role": "user", "content": "最近何をやってもうまくいかなくて、自分に自信が持てません。"}
            ]
        }
    ]
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n📝 テストシナリオ {i}: {scenario['name']}")
        print("-" * 40)
        
        try:
            # ファインチューニングされたモデルで応答生成
            response = client.chat.completions.create(
                model=model_id,
                messages=scenario["messages"],
                max_tokens=500,
                temperature=0.7,
                top_p=0.9
            )
            
            # 応答を表示
            assistant_response = response.choices[0].message.content
            print(f"👤 相談者: {scenario['messages'][-1]['content']}")
            print(f"🤖 カウンセラー: {assistant_response}")
            
            # トークン使用量を表示
            usage = response.usage
            print(f"📊 トークン使用量: 入力={usage.prompt_tokens}, 出力={usage.completion_tokens}, 合計={usage.total_tokens}")
            
        except Exception as e:
            print(f"❌ エラーが発生しました: {e}")
        
        print()
    
    print("=" * 60)
    print("🎉 テスト完了！")

def compare_with_base_model():
    """ベースモデルとの比較テスト"""
    
    if not load_environment():
        return
    
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # モデル設定
    finetuned_model = "ft:gpt-4o-mini-2024-07-18:personal::CAJ6PxFB"
    base_model = "gpt-4o-mini"
    
    # テスト用メッセージ
    test_message = [
        {"role": "user", "content": "最近うつ気分で、何もやる気が起きません。どうしたらいいでしょうか？"}
    ]
    
    print(f"\n🔄 ベースモデルとファインチューニングモデルの比較")
    print("=" * 60)
    
    for model_name, model_id in [("ベースモデル", base_model), ("ファインチューニング", finetuned_model)]:
        print(f"\n🤖 {model_name} ({model_id}):")
        print("-" * 40)
        
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=test_message,
                max_tokens=300,
                temperature=0.7
            )
            
            print(f"応答: {response.choices[0].message.content}")
            print(f"トークン: {response.usage.total_tokens}")
            
        except Exception as e:
            print(f"❌ エラー: {e}")
    
    print("\n=" * 60)
    print("🎯 比較完了！カウンセリングスタイルの違いを確認してください。")

def interactive_chat():
    """インタラクティブなチャット"""
    
    if not load_environment():
        return
    
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    model_id = "ft:gpt-4o-mini-2024-07-18:personal::CAJ6PxFB"
    
    print(f"\n💬 ファインチューニングモデルとの対話開始")
    print(f"モデル: {model_id}")
    print("終了するには 'quit' または 'exit' と入力してください")
    print("=" * 60)
    
    messages = []
    
    while True:
        try:
            # ユーザー入力
            user_input = input("\n👤 あなた: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 対話を終了します。")
                break
            
            if not user_input:
                continue
            
            # メッセージ履歴に追加
            messages.append({"role": "user", "content": user_input})
            
            # モデルに送信
            response = client.chat.completions.create(
                model=model_id,
                messages=messages,
                max_tokens=400,
                temperature=0.7
            )
            
            # 応答を取得・表示
            assistant_response = response.choices[0].message.content
            print(f"🤖 カウンセラー: {assistant_response}")
            
            # メッセージ履歴に追加
            messages.append({"role": "assistant", "content": assistant_response})
            
            # 長くなりすぎたら古いメッセージを削除
            if len(messages) > 10:
                messages = messages[-8:]  # 最新の8メッセージを保持
            
        except KeyboardInterrupt:
            print("\n\n👋 対話を終了します。")
            break
        except Exception as e:
            print(f"\n❌ エラーが発生しました: {e}")

def main():
    """メイン関数"""
    print("🚀 ファインチューニングモデル テストスクリプト")
    print("=" * 60)
    
    while True:
        print("\n選択してください:")
        print("1. 基本テスト（複数シナリオ）")
        print("2. ベースモデルとの比較")
        print("3. インタラクティブチャット")
        print("4. 終了")
        
        choice = input("\n番号を入力してください (1-4): ").strip()
        
        if choice == "1":
            test_finetuned_model()
        elif choice == "2":
            compare_with_base_model()
        elif choice == "3":
            interactive_chat()
        elif choice == "4":
            print("👋 プログラムを終了します。")
            break
        else:
            print("❌ 無効な選択です。1-4の番号を入力してください。")

if __name__ == "__main__":
    main()
