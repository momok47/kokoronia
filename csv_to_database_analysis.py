#!/usr/bin/env python3
"""
CSVファイルからzero-shot learningでの分析を実行し、SQLiteデータベースに保存するスクリプト

使用方法:
    python csv_to_database_analysis.py user001 sample_conversation_user001.csv

CSVファイルの形式（1つのCSVに1人のユーザーの会話がまとめられている）:
    text
    最近、機械学習について勉強しています。
    新しいプログラミング言語も学んでいます。
    パフォーマンスが良いのが魅力です。
"""

import sys
import os
import pandas as pd
import argparse
from pathlib import Path

# プロジェクトルートをPYTHONPATHに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'src'))

# Django設定の初期化
django_project_root = project_root / 'src' / 'webapp'
sys.path.insert(0, str(django_project_root))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'project.settings')

import django
django.setup()

# Django初期化後にインポート
from core.analysis.zero_shot_learning import ZeroShotLearning
from accounts.models import User, UserTopicScore
from accounts.utils import save_user_insights, print_user_topic_summary

class CSVToDBAnalysis:
    def __init__(self, unidic_path=None):
        """
        CSVからデータベース分析の初期化
        
        Args:
            unidic_path (str, optional): UniDic辞書のパス
                                      - None: デフォルト辞書（UniDic CSJ版）を使用
                                      - "lite": UniDic-liteを使用（軽量・高速）
                                      - パス指定: 指定されたUniDic辞書を使用
        """
        # 辞書の選択肢：
        # 1. UniDic CSJ版: 高品質、音声認識に最適（デフォルト）
        # 2. UniDic-lite: 軽量で高速
        # 3. IPA辞書: MeCabデフォルト
        
        if unidic_path is None:
            # デフォルト: UniDic CSJ版（音声認識に最適）
            self.unidic_path = "/Users/shirakawamomoko/Desktop/electronic_dictionary/unidic-csj-202302"
        elif unidic_path == "lite":
            # UniDic-liteを使用（軽量・高速）
            self.unidic_path = None  # MeCabがデフォルト辞書を使用
        else:
            # カスタムパスを使用
            self.unidic_path = unidic_path
        
        # 改善されたzero-shot learningの設定
        self.model_name = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
        # 元のラベル（15個）に戻す
        self.topic_labels = ["社会", "まなび", "テクノロジー", "カルチャー", "アウトドア", "フード", 
                           "旅行おでかけ", "ライフスタイル", "ビジネス", "読書", "キャリア", 
                           "デザイン", "IT", "経済投資", "ネットワーク"]
        
        # Zero-shot learning分析器の初期化
        print("🔧 Zero-shot learning分析器を初期化しています...")
        self.analyzer = ZeroShotLearning(
            model_name=self.model_name,
            unidic_path=self.unidic_path
        )
        print("✅ 分析器の初期化が完了しました")
    
    def load_user_conversation(self, csv_file_path):
        """
        CSVファイルから1人のユーザーの会話データを読み込む
        
        Args:
            csv_file_path (str): CSVファイルのパス
            
        Returns:
            list: 会話テキストのリスト
        """
        try:
            df = pd.read_csv(csv_file_path)
            
            # 'text'カラムの確認
            if 'text' not in df.columns:
                raise ValueError("CSVファイルに'text'カラムが見つかりません")
            
            # 空のテキストを除去
            df = df.dropna(subset=['text'])
            df = df[df['text'].str.strip() != '']
            
            # テキストのリストとして返す
            conversation_texts = df['text'].tolist()
            
            print(f"📊 CSVファイルから {len(conversation_texts)} 個の発言を読み込みました")
            return conversation_texts
            
        except Exception as e:
            print(f"❌ CSVファイルの読み込みに失敗しました: {e}")
            raise
    
    def analyze_user_conversation(self, account_id, conversation_texts):
        """
        ユーザーの会話全体を分析する
        
        Args:
            account_id (str): アカウントID
            conversation_texts (list): 発言テキストのリスト
            
        Returns:
            dict: 分析結果
        """
        try:
            # 会話データの準備（全ての発言をまとめて分析）
            conversation_data = []
            for text in conversation_texts:
                conversation_data.append({"speaker": account_id, "text": text})
            
            print(f"🔍 {len(conversation_texts)} 個の発言を分析しています...")
            
            # 分析実行
            insights = self.analyzer.extract_insights(
                conversation_data=conversation_data,
                topic_labels=self.topic_labels,
                display_speaker_label=account_id
            )
            
            return insights
            
        except Exception as e:
            print(f"❌ 会話分析に失敗しました ({account_id}): {e}")
            return None
    
    def ensure_user_exists(self, account_id):
        """
        ユーザーが存在することを確認し、存在しない場合は作成する
        
        Args:
            account_id (str): アカウントID
            
        Returns:
            tuple: (User, created: bool)
        """
        try:
            user = User.objects.get(account_id=account_id)
            return user, False
        except User.DoesNotExist:
            # ユーザーが存在しない場合は作成
            user = User.objects.create_user(
                account_id=account_id,
                email=f"{account_id}@example.com",  # 仮のメールアドレス
                first_name=account_id,
                last_name="User"
            )
            return user, True
    
    def process_user_csv(self, account_id, csv_file_path):
        """
        指定されたユーザーのCSVファイルを処理する
        
        Args:
            account_id (str): アカウントID
            csv_file_path (str): CSVファイルのパス
            
        Returns:
            dict: 処理結果
        """
        print(f"\n🚀 ユーザー '{account_id}' の会話分析を開始します...")
        print(f"📁 ファイル: {csv_file_path}")
        
        # CSVファイルから会話データを読み込み
        conversation_texts = self.load_user_conversation(csv_file_path)
        
        if not conversation_texts:
            return {
                'success': False,
                'error': '有効な会話データが見つかりませんでした',
                'account_id': account_id
            }
        
        # 全発言を結合して表示
        combined_text = " ".join(conversation_texts)
        print(f"💬 合計文字数: {len(combined_text)} 文字")
        print(f"📝 サンプル: {combined_text[:100]}...")
        
        # ユーザーの存在確認・作成
        user, created = self.ensure_user_exists(account_id)
        if created:
            print(f"✅ 新規ユーザー '{account_id}' を作成しました")
        else:
            print(f"👤 既存ユーザー '{account_id}' を確認しました")
        
        # 会話全体を分析
        insights = self.analyze_user_conversation(account_id, conversation_texts)
        
        if not insights:
            return {
                'success': False,
                'error': '会話分析に失敗しました',
                'account_id': account_id
            }
        
        # データベースに保存
        success, message, topic_score = save_user_insights(account_id, insights)
        
        if success:
            print(f"✅ {message}")
            return {
                'success': True,
                'account_id': account_id,
                'best_topic': insights['best_topic'],
                'best_score': insights['best_score'],
                'topic_scores': insights['topic_scores'],
                'conversation_count': len(conversation_texts),
                'message': message
            }
        else:
            print(f"❌ データベース保存に失敗: {message}")
            return {
                'success': False,
                'error': message,
                'account_id': account_id
            }
    
    def print_result_summary(self, result):
        """
        処理結果の要約を表示する
        
        Args:
            result (dict): 処理結果
        """
        print("\n" + "="*60)
        print("📈 分析結果の要約")
        print("="*60)
        
        if result['success']:
            print(f"👤 ユーザーID: {result['account_id']}")
            print(f"💬 発言数: {result['conversation_count']}")
            print(f"🎯 検出トピック: {result['best_topic']}")
            print(f"📊 信頼度: {result['best_score']:.3f}")
            
            print(f"\n📋 全トピックスコア:")
            sorted_scores = sorted(result['topic_scores'].items(), key=lambda x: x[1], reverse=True)
            for i, (topic, score) in enumerate(sorted_scores, 1):
                star = "★" if i == 1 else "　"
                print(f"  {star} {i}位: {topic:8s} → {score:.3f}")
        else:
            print(f"❌ 処理失敗")
            print(f"👤 ユーザーID: {result['account_id']}")
            print(f"🚫 エラー: {result['error']}")

def main():
    parser = argparse.ArgumentParser(
        description="CSVファイル（1人のユーザーの会話）からzero-shot learningで分析し、SQLiteデータベースに保存",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  python csv_to_database_analysis.py user001 conversation_user001.csv
  python csv_to_database_analysis.py user001 conversation_user001.csv --show-summary
  
CSVファイルの形式（1つのCSVに1人のユーザーの会話）:
  text
  最近、機械学習について勉強しています。
  新しいプログラミング言語も学んでいます。
  パフォーマンスが良いのが魅力です。
        """
    )
    
    parser.add_argument('account_id', help='分析対象のユーザーのアカウントID')
    parser.add_argument('csv_file', help='分析対象のCSVファイルのパス')
    parser.add_argument('--show-summary', action='store_true',
                       help='処理後にユーザーのトピックスコア要約を表示する')
    
    args = parser.parse_args()
    
    # CSVファイルの存在確認
    if not os.path.exists(args.csv_file):
        print(f"❌ CSVファイルが見つかりません: {args.csv_file}")
        return
    
    try:
        # 分析処理の実行
        analyzer = CSVToDBAnalysis()
        result = analyzer.process_user_csv(args.account_id, args.csv_file)
        
        # 結果の表示
        analyzer.print_result_summary(result)
        
        # オプション: ユーザーサマリーの表示
        if args.show_summary and result['success']:
            print_user_topic_summary(args.account_id)
        
        if result['success']:
            print("\n🎉 処理が完了しました！")
        else:
            print(f"\n💥 処理に失敗しました: {result['error']}")
        
    except Exception as e:
        print(f"❌ 処理中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 