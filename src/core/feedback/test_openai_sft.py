#!/usr/bin/env python3
"""
OpenAI SFT テスト実行スクリプト

このスクリプトは、OpenAI SFTの機能を小規模でテストするためのものです。
実際のAPIを使用せずに、データ準備とフォーマット変換をテストできます。
"""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_data_preparation():
    """データ準備機能のテスト"""
    try:
        # プロジェクトルートの.envファイルを読み込み
        project_root = Path(__file__).parent.parent.parent.parent  # src/core/feedback から4つ上
        env_path = project_root / ".env"
        if env_path.exists():
            load_dotenv(env_path)
            logger.info(f".envファイルを読み込みました: {env_path}")
        
        from openai_sft import OpenAISFT
        
        logger.info("=== データ準備テスト開始 ===")
        
        # ダミーAPIキーでインスタンス作成（データ準備のみなのでAPIは使用しない）
        os.environ["OPENAI_API_KEY"] = "test-key-for-data-prep-only"
        sft = OpenAISFT()
        
        # 小規模データでテスト
        logger.info("小規模データセット（10サンプル）を準備中...")
        training_data = sft.prepare_dataset(max_samples=10)
        
        logger.info(f"準備されたサンプル数: {len(training_data)}")
        
        # 最初のサンプルを表示
        if training_data:
            logger.info("=== 最初のサンプル ===")
            first_sample = training_data[0]
            for i, message in enumerate(first_sample["messages"][:3]):  # 最初の3メッセージのみ表示
                logger.info(f"  {i+1}. {message['role']}: {message['content'][:100]}...")
        
        # データ保存テスト
        logger.info("データ保存テスト中...")
        training_file = sft.save_training_data(training_data, "test_training_data.jsonl")
        logger.info(f"テストデータを保存: {training_file}")
        
        # ファイル内容の確認
        with open(training_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            logger.info(f"保存されたJSONL行数: {len(lines)}")
        
        logger.info("=== データ準備テスト完了 ✅ ===")
        return True
        
    except Exception as e:
        logger.error(f"データ準備テストでエラー: {e}")
        return False

def test_system_monitor():
    """システム監視機能のテスト"""
    try:
        from openai_sft import SystemMonitor
        import time
        
        logger.info("=== システム監視テスト開始 ===")
        
        monitor = SystemMonitor(interval=5)  # 5秒間隔
        
        # 10秒間監視
        monitor.start_monitoring()
        logger.info("10秒間監視中...")
        time.sleep(10)
        monitor.stop_monitoring()
        
        logger.info("=== システム監視テスト完了 ✅ ===")
        return True
        
    except Exception as e:
        logger.error(f"システム監視テストでエラー: {e}")
        return False

def test_format_conversion():
    """データフォーマット変換のテスト"""
    try:
        from openai_sft import OpenAISFT
        
        logger.info("=== フォーマット変換テスト開始 ===")
        
        os.environ["OPENAI_API_KEY"] = "test-key-for-format-test-only"
        sft = OpenAISFT()
        
        # テスト用の対話データ
        test_dialogue = [
            {"role": "client", "utterance": "こんにちは、相談があります。"},
            {"role": "counselor", "utterance": "こんにちは。どのようなことでお悩みですか？"},
            {"role": "client", "utterance": "最近、仕事でストレスを感じています。"},
            {"role": "counselor", "utterance": "そのストレスについて、もう少し詳しく教えていただけますか？"}
        ]
        
        # フォーマット変換
        messages = sft._format_dialogue_for_openai(test_dialogue)
        
        logger.info("変換結果:")
        for i, message in enumerate(messages):
            logger.info(f"  {i+1}. {message['role']}: {message['content']}")
        
        # 期待される結果と比較
        expected_roles = ["user", "assistant", "user", "assistant"]
        actual_roles = [msg["role"] for msg in messages]
        
        if actual_roles == expected_roles:
            logger.info("=== フォーマット変換テスト完了 ✅ ===")
            return True
        else:
            logger.error(f"期待される役割: {expected_roles}, 実際の役割: {actual_roles}")
            return False
            
    except Exception as e:
        logger.error(f"フォーマット変換テストでエラー: {e}")
        return False

def main():
    """メインテスト関数"""
    logger.info("OpenAI SFT テスト実行開始")
    
    tests = [
        ("データ準備", test_data_preparation),
        ("システム監視", test_system_monitor),
        ("フォーマット変換", test_format_conversion)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"テスト実行: {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"{test_name}テストで予期しないエラー: {e}")
            results.append((test_name, False))
    
    # 結果サマリー
    logger.info(f"\n{'='*50}")
    logger.info("テスト結果サマリー")
    logger.info(f"{'='*50}")
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\n合計: {passed}/{len(results)} テスト通過")
    
    if passed == len(results):
        logger.info("🎉 全てのテストが成功しました！")
        return 0
    else:
        logger.error("⚠️ 一部のテストが失敗しました。")
        return 1

if __name__ == "__main__":
    exit(main())
