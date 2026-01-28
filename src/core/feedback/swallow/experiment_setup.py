# -*- coding: utf-8 -*-
# experiment_setup.py - 実験管理ツールのセットアップスクリプト

import os
import subprocess
import sys
import logging

logger = logging.getLogger(__name__)
FEEDBACK_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def install_experiment_tools():
    """実験管理ツールのインストール"""
    tools = [
        ("tensorboard", "TensorBoard"),
        ("wandb", "Weights & Biases")
    ]
    
    print("=== 実験管理ツールのセットアップ ===")
    
    for package, name in tools:
        try:
            __import__(package)
            print("✅ {} は既にインストールされています".format(name))
        except ImportError:
            print("📦 {} をインストールしています...".format(name))
            try:
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", package
                ])
                print("✅ {} のインストールが完了しました".format(name))
            except subprocess.CalledProcessError as e:
                print("❌ {} のインストールに失敗しました: {}".format(name, e))

def setup_wandb():
    """W&Bの初期設定"""
    print("\n=== W&B (Weights & Biases) の設定 ===")
    print("1. https://wandb.ai でアカウントを作成してください")
    print("2. APIキーを取得してください")
    print("3. 以下のコマンドを実行してログインしてください:")
    print("   wandb login")
    print("\n環境変数での設定（推奨）:")
    print("   export WANDB_API_KEY=your_api_key_here")
    print("   export WANDB_PROJECT=emotion-reward-sft")
    print("   export WANDB_ENTITY=your_username_or_team")

def create_tensorboard_script():
    """TensorBoard起動用スクリプトの作成"""
    script_content = '''#!/bin/bash
# TensorBoard起動スクリプト

LOG_DIR="./logs_tensorboard"

echo "TensorBoardを起動しています..."
echo "ログディレクトリ: $LOG_DIR"
echo "ブラウザで http://localhost:6006 を開いてください"

tensorboard --logdir=$LOG_DIR --port=6006 --host=0.0.0.0
'''
    
    script_path = os.path.join(FEEDBACK_DIR, "start_tensorboard.sh")
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    # 実行権限を付与
    os.chmod(script_path, 0o755)
    print("TensorBoard起動スクリプトを作成しました: {}".format(script_path))

def create_env_template():
    """環境変数設定テンプレートの作成"""
    env_content = '''# 実験管理ツール用環境変数設定テンプレート
# このファイルをコピーして .env として使用してください

# 使用する実験管理ツール ("tensorboard", "wandb", "both", "none")
EXPERIMENT_TRACKING_TOOL=both

# W&B設定
WANDB_API_KEY=your_wandb_api_key_here
WANDB_PROJECT=emotion-reward-sft
WANDB_ENTITY=your_username_or_team_name

# TensorBoard設定
TENSORBOARD_LOG_DIR=./logs_tensorboard

# その他の設定
EXPERIMENT_NAME=emotion_sft_experiment
'''
    
    env_path = os.path.join(FEEDBACK_DIR, ".env.example")
    if not os.path.exists(env_path):
        with open(env_path, 'w', encoding='utf-8') as f:
            f.write(env_content)
        print("環境変数設定テンプレートを作成しました: {}".format(env_path))
    else:
        print("環境変数設定テンプレートは既に存在します: {}".format(env_path))

def main():
    """メイン処理"""
    print("感情報酬SFT実験管理ツールのセットアップ")
    print("=" * 50)
    
    # パッケージのインストール
    install_experiment_tools()
    
    # TensorBoard起動スクリプトの作成
    create_tensorboard_script()
    
    # 環境変数テンプレートの作成
    create_env_template()
    
    # W&Bの設定案内
    setup_wandb()
    
    print("\n=== セットアップ完了 ===")
    print("実験を開始する前に:")
    print("1. .env.example をコピーして .env を作成し、設定を編集してください")
    print("2. W&Bを使用する場合は 'wandb login' を実行してください")
    print("3. TensorBoardを使用する場合は './start_tensorboard.sh' を実行してください")

if __name__ == "__main__":
    main()
