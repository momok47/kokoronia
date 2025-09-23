#!/usr/bin/env python3
"""
テストデータの構造を詳しく調査し、正解スコア抽出の問題を特定するスクリプト
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

class TestDataStructureAnalyzer:
    """テストデータの構造を分析するクラス"""
    
    def __init__(self, output_dir: str = "openai_sft_outputs"):
        self.output_dir = Path(output_dir)
    
    def validate_data_structure(self):
        """データ構造の妥当性を検証（kokorochatの正しい構造に対応）"""
        logger.info("データ構造の妥当性を検証します")
        
        # 元のkokorochatデータファイルの存在確認
        logger.info("\n=== 元のkokorochatデータファイルの確認 ===")
        original_data_files = list(self.output_dir.glob("*kokorochat*.jsonl"))
        if original_data_files:
            logger.info(f"元のkokorochatデータファイル: {len(original_data_files)}個")
            for file in original_data_files:
                logger.info(f"  {file.name}")
                # 最初のサンプルの構造を確認
                try:
                    with open(file, 'r', encoding='utf-8') as f:
                        first_line = f.readline().strip()
                        if first_line:
                            sample = json.loads(first_line)
                            logger.info(f"    構造: {list(sample.keys())}")
                            if 'review_by_client_jp' in sample:
                                review_data = sample['review_by_client_jp']
                                if isinstance(review_data, dict):
                                    logger.info(f"    評価項目数: {len(review_data)}")
                                    # 最初の数項目のスコアを表示
                                    for i, (key, value) in enumerate(review_data.items()):
                                        if i < 3:  # 最初の3項目のみ
                                            logger.info(f"      {key}: {value}")
                                        else:
                                            break
                except Exception as e:
                    logger.error(f"    ファイル読み込みエラー: {e}")
        else:
            logger.warning("元のkokorochatデータファイルが見つかりません")
            logger.warning("データ変換の前後で比較できません")
        
        # 必要なファイルの存在確認
        required_files = [
            "train_data_*.jsonl",
            "valid_data_*.jsonl", 
            "test_data_*.jsonl"
        ]
        
        missing_files = []
        for pattern in required_files:
            files = list(self.output_dir.glob(pattern))
            if not files:
                missing_files.append(pattern)
            else:
                logger.info(f"{pattern}: {len(files)}個のファイルが見つかりました")
        
        if missing_files:
            logger.warning(f"以下のファイルが見つかりません: {missing_files}")
        
        # 各ファイルの構造を確認
        for pattern in required_files:
            files = list(self.output_dir.glob(pattern))
            if files:
                latest_file = max(files, key=lambda x: x.stat().st_mtime)
                self._analyze_file_structure(latest_file, pattern)
        
        # データ変換の問題を特定
        self._identify_data_conversion_issues()
    
    def _analyze_file_structure(self, file_path: Path, file_type: str):
        """ファイルの構造を分析"""
        logger.info(f"\n=== {file_type} の構造分析 ===")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # 最初の数行を読み込んで構造を確認
                sample_count = 0
                structure_summary = {}
                
                for i, line in enumerate(f):
                    if i >= 10:  # 最初の10サンプルのみ
                        break
                    
                    sample = json.loads(line.strip())
                    sample_count += 1
                    
                    # キーの構造を記録
                    keys = tuple(sorted(sample.keys()))
                    if keys not in structure_summary:
                        structure_summary[keys] = 0
                    structure_summary[keys] += 1
                    
                    # 最初のサンプルの詳細表示
                    if i == 0:
                        logger.info(f"最初のサンプルの構造:")
                        for key, value in sample.items():
                            if isinstance(value, dict):
                                logger.info(f"  {key}: dict with keys {list(value.keys())}")
                            elif isinstance(value, list):
                                logger.info(f"  {key}: list with {len(value)} items")
                            else:
                                logger.info(f"  {key}: {type(value).__name__}")
                
                logger.info(f"総サンプル数: {sample_count}")
                logger.info(f"構造パターン:")
                for keys, count in structure_summary.items():
                    logger.info(f"  {keys}: {count}サンプル")
                
                # 正解スコアが含まれているかチェック
                self._check_ground_truth_availability(file_path, file_type)
                
        except Exception as e:
            logger.error(f"{file_path} の分析中にエラー: {e}")
    
    def _check_ground_truth_availability(self, file_path: Path, file_type: str):
        """正解スコアの可用性をチェック（kokorochatの正しい構造に対応）"""
        logger.info(f"\n--- {file_type} の正解スコア可用性チェック ---")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # 最初の100サンプルをチェック
                ground_truth_found = 0
                score_keys = []
                structure_analysis = {
                    'has_dialogue': 0,
                    'has_topic': 0,
                    'has_review_by_client_jp': 0,
                    'has_review_by_client_en': 0,
                    'has_messages_only': 0,
                    'other_keys': set()
                }
                
                for i, line in enumerate(f):
                    if i >= 100:
                        break
                    
                    sample = json.loads(line.strip())
                    
                    # データ構造を分析
                    if 'dialogue' in sample:
                        structure_analysis['has_dialogue'] += 1
                    if 'topic' in sample:
                        structure_analysis['has_topic'] += 1
                    if 'review_by_client_jp' in sample:
                        structure_analysis['has_review_by_client_jp'] += 1
                    if 'review_by_client_en' in sample:
                        structure_analysis['has_review_by_client_en'] += 1
                    if 'messages' in sample and len(sample.keys()) == 1:
                        structure_analysis['has_messages_only'] += 1
                    
                    # その他のキーを記録
                    for key in sample.keys():
                        if key not in ['dialogue', 'topic', 'review_by_client_jp', 'review_by_client_en', 'messages']:
                            structure_analysis['other_keys'].add(key)
                    
                    # 正解スコアの可能性があるキーを探す
                    for key, value in sample.items():
                        if isinstance(value, (int, float)) and 0 <= value <= 5:
                            if key not in score_keys:
                                score_keys.append(key)
                            ground_truth_found += 1
                            break
                        
                        elif isinstance(value, dict):
                            for sub_key, sub_value in value.items():
                                if isinstance(sub_value, (int, float)) and 0 <= sub_value <= 5:
                                    full_key = f"{key}.{sub_key}"
                                    if full_key not in score_keys:
                                        score_keys.append(full_key)
                                    ground_truth_found += 1
                                    break
                            if ground_truth_found > 0:
                                break
                
                # 結果を表示
                logger.info(f"正解スコアを含むサンプル: {ground_truth_found}/100")
                if score_keys:
                    logger.info(f"発見されたスコアキー: {score_keys}")
                else:
                    logger.warning("正解スコアが見つかりません！")
                
                # データ構造の分析結果を表示
                logger.info(f"\n--- データ構造分析 ---")
                logger.info(f"dialogueを含む: {structure_analysis['has_dialogue']}/100")
                logger.info(f"topicを含む: {structure_analysis['has_topic']}/100")
                logger.info(f"review_by_client_jpを含む: {structure_analysis['has_review_by_client_jp']}/100")
                logger.info(f"review_by_client_enを含む: {structure_analysis['has_review_by_client_en']}/100")
                logger.info(f"messagesのみ（単純化）: {structure_analysis['has_messages_only']}/100")
                
                if structure_analysis['other_keys']:
                    logger.info(f"その他のキー: {structure_analysis['other_keys']}")
                
                # 問題の特定
                if structure_analysis['has_messages_only'] > 0:
                    logger.warning(f"⚠️  {structure_analysis['has_messages_only']}サンプルが単純化されています（messagesキーのみ）")
                    logger.warning("  これが正解スコアが抽出できない原因です")
                
                if structure_analysis['has_review_by_client_jp'] == 0:
                    logger.error("❌ review_by_client_jpが含まれていません")
                    logger.error("  正解スコアの抽出が不可能です")
                
                if structure_analysis['has_dialogue'] == 0:
                    logger.warning("⚠️  dialogueキーが含まれていません")
                    logger.warning("  元のkokorochatデータ構造が失われています")
                    
        except Exception as e:
            logger.error(f"正解スコアチェック中にエラー: {e}")
    
    def generate_score_comparison_csv(self):
        """スコア比較用のCSVを生成"""
        logger.info("スコア比較用のCSVを生成します")
        
        # テストデータファイルを探す
        test_files = list(self.output_dir.glob("test_data_*.jsonl"))
        if not test_files:
            logger.error("テストデータファイルが見つかりません")
            return
        
        latest_test_file = max(test_files, key=lambda x: x.stat().st_mtime)
        
        # 正解スコアを抽出
        results = []
        try:
            with open(latest_test_file, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    sample = json.loads(line.strip())
                    
                    # 正解スコアを抽出
                    ground_truth_score = self._extract_ground_truth_score(sample)
                    
                    # 予測スコアを抽出（現在のロジック）
                    predicted_score = self._extract_score_attempt(sample)
                    
                    results.append({
                        'sample_index': i,
                        'ground_truth_score': ground_truth_score,
                        'predicted_score': predicted_score,
                        'error': None if ground_truth_score is not None else "正解スコアが見つかりません",
                        'sample_keys': list(sample.keys()),
                        'sample_structure': self._get_sample_structure_summary(sample)
                    })
                    
                    if i >= 100:  # 最初の100サンプルのみ
                        break
        
        except Exception as e:
            logger.error(f"CSV生成中にエラー: {e}")
            return
        
        # DataFrameに変換
        df = pd.DataFrame(results)
        
        # CSVとして保存
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"score_comparison_{timestamp}.csv"
        csv_path = self.output_dir / csv_filename
        
        df.to_csv(csv_path, index=False, encoding='utf-8')
        logger.info(f"スコア比較CSVを保存しました: {csv_path}")
        
        # 統計情報を表示
        logger.info(f"\n=== スコア比較統計 ===")
        logger.info(f"総サンプル数: {len(df)}")
        logger.info(f"正解スコアあり: {df['ground_truth_score'].notna().sum()}")
        logger.info(f"正解スコアなし: {df['ground_truth_score'].isna().sum()}")
        logger.info(f"予測スコアあり: {df['predicted_score'].notna().sum()}")
        logger.info(f"予測スコアなし: {df['predicted_score'].isna().sum()}")
        
        # エラーの詳細
        error_counts = df['error'].value_counts()
        if not error_counts.empty:
            logger.info(f"\nエラーの詳細:")
            for error, count in error_counts.items():
                logger.info(f"  {error}: {count}サンプル")
        
        return csv_path
    
    def _extract_ground_truth_score(self, sample: Dict[str, Any]) -> float:
        """正解スコアを抽出（kokorochatの正しい構造に対応）"""
        # 1. kokorochatの正しい構造から正解スコアを抽出
        if 'review_by_client_jp' in sample:
            review_data = sample['review_by_client_jp']
            if isinstance(review_data, dict):
                # 各評価項目のスコアを取得
                scores = []
                for key, value in review_data.items():
                    if isinstance(value, (int, float)) and 0 <= value <= 5:
                        scores.append(float(value))
                
                if scores:
                    # 平均スコアを返す
                    return sum(scores) / len(scores)
        
        # 2. 英語版の評価データもチェック
        if 'review_by_client_en' in sample:
            review_data = sample['review_by_client_en']
            if isinstance(review_data, dict):
                scores = []
                for key, value in review_data.items():
                    if isinstance(value, (int, float)) and 0 <= value <= 5:
                        scores.append(float(value))
                
                if scores:
                    return sum(scores) / len(scores)
        
        # 3. 直接的なスコアキー（従来の方法）
        direct_score_keys = ['score', 'rating', 'satisfaction', 'evaluation', 'ground_truth']
        for key in direct_score_keys:
            if key in sample:
                try:
                    score = float(sample[key])
                    if 0 <= score <= 5:
                        return score
                except (ValueError, TypeError):
                    continue
        
        # 4. ネストしたスコアキー
        nested_keys = ['metadata', 'annotation', 'label', 'data', 'result']
        for key in nested_keys:
            if key in sample and isinstance(sample[key], dict):
                for sub_key in direct_score_keys:
                    if sub_key in sample[key]:
                        try:
                            score = float(sample[key][sub_key])
                            if 0 <= score <= 5:
                                return score
                        except (ValueError, TypeError):
                            continue
        
        # 5. kokorochat特有のキー
        kokorochat_keys = [
            'kokorochat_score', 'kokorochat_rating', 'kokorochat_evaluation',
            'human_score', 'human_rating', 'human_evaluation',
            'expert_score', 'expert_rating', 'expert_evaluation',
            'reference_score', 'reference_rating', 'reference_evaluation'
        ]
        
        for key in kokorochat_keys:
            if key in sample:
                try:
                    score = float(sample[key])
                    if 0 <= score <= 5:
                        return score
                except (ValueError, TypeError):
                    continue
        
        # 6. メッセージ内のスコア（最後の手段）
        if 'messages' in sample:
            messages = sample['messages']
            # 最後の数メッセージからスコアを探す
            for msg in reversed(messages[-5:]):
                if msg.get('role') == 'assistant':
                    content = msg.get('content', '')
                    # スコア関連のキーワードを含む場合のみ数値を探す
                    score_keywords = ['評価', 'スコア', '点', 'rating', 'score', 'satisfaction']
                    if any(keyword in content for keyword in score_keywords):
                        import re
                        match = re.search(r'(\d+(?:\.\d+)?)', content)
                        if match:
                            try:
                                score = float(match.group(1))
                                if 0 <= score <= 5:
                                    return score
                            except ValueError:
                                continue
        
        return None

    def analyze_test_data_structure(self, max_samples: int = 10):
        """
        テストデータの構造を詳細分析
        
        Args:
            max_samples: 分析するサンプル数
        """
        logger.info(f"テストデータの構造を詳細分析します（最大{max_samples}サンプル）")
        
        # テストデータファイルを探す
        test_files = list(self.output_dir.glob("test_data_*.jsonl"))
        if not test_files:
            logger.error("テストデータファイルが見つかりません")
            return
        
        latest_test_file = max(test_files, key=lambda x: x.stat().st_mtime)
        logger.info(f"分析対象ファイル: {latest_test_file}")
        
        # スコア比較CSVも読み込み
        csv_files = list(self.output_dir.glob("score_comparison_*.csv"))
        if csv_files:
            latest_csv = max(csv_files, key=lambda x: x.stat().st_mtime)
            score_df = pd.read_csv(latest_csv)
            logger.info(f"スコア比較CSV: {latest_csv}")
            logger.info(f"CSV行数: {len(score_df)}")
        else:
            score_df = None
            logger.warning("スコア比較CSVが見つかりません")
        
        try:
            with open(latest_test_file, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i >= max_samples:
                        break
                    
                    sample = json.loads(line.strip())
                    print(f"\n{'='*80}")
                    print(f"サンプル {i+1} の詳細分析")
                    print(f"{'='*80}")
                    
                    # 基本情報
                    print(f"キー数: {len(sample.keys())}")
                    print(f"キー一覧: {list(sample.keys())}")
                    
                    # 各キーの詳細分析
                    for key, value in sample.items():
                        self._analyze_key_value(key, value, i+1)
                    
                    # CSVとの対応確認
                    if score_df is not None and i < len(score_df):
                        csv_row = score_df.iloc[i]
                        print(f"\n--- CSV対応情報 ---")
                        print(f"CSV行: {i}")
                        print(f"予測スコア: {csv_row.get('predicted_score', 'N/A')}")
                        print(f"正解スコア: {csv_row.get('correct_score', 'N/A')}")
                        print(f"エラー: {csv_row.get('error', 'N/A')}")
                        
                        # 正解スコアが欠損している場合の詳細調査
                        if pd.isna(csv_row.get('correct_score')):
                            print(f"⚠️  正解スコアが欠損しています！")
                            self._investigate_missing_score(sample, i+1)
                    
                    print(f"\n{'='*80}")
                    
        except Exception as e:
            logger.error(f"テストデータの分析中にエラーが発生: {e}")
            import traceback
            traceback.print_exc()
    
    def analyze_messages_for_scores(self, max_samples: int = 10):
        """
        メッセージ内から正解スコアを探す詳細分析
        
        Args:
            max_samples: 分析するサンプル数
        """
        logger.info(f"メッセージ内から正解スコアを探す詳細分析（最大{max_samples}サンプル）")
        
        # テストデータファイルを探す
        test_files = list(self.output_dir.glob("test_data_*.jsonl"))
        if not test_files:
            logger.error("テストデータファイルが見つかりません")
            return
        
        latest_test_file = max(test_files, key=lambda x: x.stat().st_mtime)
        
        # スコア比較CSVも読み込み
        csv_files = list(self.output_dir.glob("score_comparison_*.csv"))
        if csv_files:
            latest_csv = max(csv_files, key=lambda x: x.stat().st_mtime)
            score_df = pd.read_csv(latest_csv)
        else:
            score_df = None
            logger.warning("スコア比較CSVが見つかりません")
        
        try:
            with open(latest_test_file, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i >= max_samples:
                        break
                    
                    sample = json.loads(line.strip())
                    print(f"\n{'='*80}")
                    print(f"サンプル {i+1} のメッセージ詳細分析")
                    print(f"{'='*80}")
                    
                    messages = sample['messages']
                    print(f"メッセージ数: {len(messages)}")
                    
                    # CSVの正解スコアを取得
                    csv_score = None
                    if score_df is not None and i < len(score_df):
                        csv_score = score_df.iloc[i].get('correct_score')
                        if not pd.isna(csv_score):
                            print(f"CSV正解スコア: {csv_score}")
                        else:
                            print(f"CSV正解スコア: 欠損")
                    
                    # 各メッセージを詳しく分析
                    for j, msg in enumerate(messages):
                        role = msg.get('role', 'unknown')
                        content = msg.get('content', '')
                        
                        # スコア関連のキーワードを探す
                        score_keywords = ['点', 'スコア', '評価', '満足度', 'rating', 'score', 'satisfaction']
                        has_score_keyword = any(keyword in content for keyword in score_keywords)
                        
                        # 数値を探す
                        import re
                        numbers = re.findall(r'\d+(?:\.\d+)?', content)
                        valid_scores = [n for n in numbers if 0 <= float(n) <= 5]
                        
                        # スコア関連の情報がある場合のみ詳細表示
                        if has_score_keyword or valid_scores or len(content) > 200:
                            print(f"\n  メッセージ {j} (role={role}):")
                            print(f"    内容: {content[:300]}...")
                            
                            if valid_scores:
                                print(f"    有効なスコア候補: {valid_scores}")
                            
                            if has_score_keyword:
                                print(f"    🔍 スコア関連キーワードを含む")
                            
                            # 長いメッセージの場合は後半も確認
                            if len(content) > 300:
                                print(f"    後半内容: {content[-200]}...")
                    
                    # 正解スコアが見つからない場合の特別調査
                    if csv_score is None or pd.isna(csv_score):
                        print(f"\n⚠️  正解スコアが欠損しているサンプルの特別調査:")
                        self._deep_search_for_scores(messages, i+1)
                    
                    print(f"\n{'='*80}")
                    
        except Exception as e:
            logger.error(f"メッセージ分析中にエラーが発生: {e}")
            import traceback
            traceback.print_exc()
    
    def analyze_message_structure_for_scores(self, max_samples: int = 5):
        """
        メッセージの構造から正解スコアを探す詳細分析
        
        Args:
            max_samples: 分析するサンプル数
        """
        logger.info(f"メッセージの構造から正解スコアを探す詳細分析（最大{max_samples}サンプル）")
        
        # テストデータファイルを探す
        test_files = list(self.output_dir.glob("test_data_*.jsonl"))
        if not test_files:
            logger.error("テストデータファイルが見つかりません")
            return
        
        latest_test_file = max(test_files, key=lambda x: x.stat().st_mtime)
        
        # スコア比較CSVも読み込み
        csv_files = list(self.output_dir.glob("score_comparison_*.csv"))
        if csv_files:
            latest_csv = max(csv_files, key=lambda x: x.stat().st_mtime)
            score_df = pd.read_csv(latest_csv)
        else:
            score_df = None
        
        try:
            with open(latest_test_file, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i >= max_samples:
                        break
                    
                    sample = json.loads(line.strip())
                    print(f"\n{'='*80}")
                    print(f"サンプル {i+1} のメッセージ構造詳細分析")
                    print(f"{'='*80}")
                    
                    messages = sample['messages']
                    print(f"メッセージ数: {len(messages)}")
                    
                    # CSVの正解スコアを取得
                    csv_score = None
                    if score_df is not None and i < len(score_df):
                        csv_score = score_df.iloc[i].get('correct_score')
                        if not pd.isna(csv_score):
                            print(f"CSV正解スコア: {csv_score}")
                        else:
                            print(f"CSV正解スコア: 欠損")
                    
                    # メッセージの構造を詳しく分析
                    print(f"\n--- メッセージ構造分析 ---")
                    
                    # 1. 最初の数メッセージ
                    print(f"最初の5メッセージ:")
                    for j in range(min(5, len(messages))):
                        msg = messages[j]
                        role = msg.get('role', 'unknown')
                        content = msg.get('content', '')
                        print(f"  {j}: role={role}, content={content[:100]}...")
                    
                    # 2. 最後の数メッセージ
                    print(f"\n最後の5メッセージ:")
                    for j in range(max(0, len(messages)-5), len(messages)):
                        msg = messages[j]
                        role = msg.get('role', 'unknown')
                        content = msg.get('content', '')
                        print(f"  {j}: role={role}, content={content[:100]}...")
                    
                    # 3. 特定の位置のメッセージ（中間、3/4位置など）
                    if len(messages) > 10:
                        mid_point = len(messages) // 2
                        three_quarter = (len(messages) * 3) // 4
                        
                        print(f"\n中間位置のメッセージ:")
                        for pos in [mid_point-1, mid_point, mid_point+1]:
                            if 0 <= pos < len(messages):
                                msg = messages[pos]
                                role = msg.get('role', 'unknown')
                                content = msg.get('content', '')
                                print(f"  {pos}: role={role}, content={content[:100]}...")
                        
                        print(f"\n3/4位置のメッセージ:")
                        for pos in [three_quarter-1, three_quarter, three_quarter+1]:
                            if 0 <= pos < len(messages):
                                msg = messages[pos]
                                role = msg.get('role', 'unknown')
                                content = msg.get('content', '')
                                print(f"  {pos}: role={role}, content={content[:100]}...")
                    
                    # 4. スコア関連のキーワードを含むメッセージを探す
                    print(f"\n--- スコア関連キーワード検索 ---")
                    score_keywords = ['評価', 'スコア', '点', 'rating', 'score', 'satisfaction', '満足度', '採点']
                    
                    for j, msg in enumerate(messages):
                        content = msg.get('content', '')
                        if any(keyword in content for keyword in score_keywords):
                            role = msg.get('role', 'unknown')
                            print(f"  メッセージ {j} (role={role}): スコア関連キーワード発見")
                            print(f"    内容: {content[:200]}...")
                    
                    # 5. 数値が含まれるメッセージの詳細分析
                    print(f"\n--- 数値を含むメッセージの詳細分析 ---")
                    for j, msg in enumerate(messages):
                        content = msg.get('content', '')
                        import re
                        numbers = re.findall(r'\d+(?:\.\d+)?', content)
                        
                        if numbers:
                            role = msg.get('role', 'unknown')
                            # 0-5の範囲の数値のみ表示
                            valid_scores = [n for n in numbers if 0 <= float(n) <= 5]
                            if valid_scores:
                                print(f"  メッセージ {j} (role={role}):")
                                print(f"    含まれる数値: {numbers}")
                                print(f"    有効なスコア候補: {valid_scores}")
                                print(f"    内容: {content[:150]}...")
                    
                    print(f"\n{'='*80}")
                    
        except Exception as e:
            logger.error(f"メッセージ構造分析中にエラーが発生: {e}")
            import traceback
            traceback.print_exc()
    
    def _deep_search_for_scores(self, messages: List[Dict[str, Any]], sample_num: int):
        """スコアを深く探す特別調査"""
        print(f"  サンプル{sample_num}の深層調査:")
        
        # 最後の数メッセージを詳しく調べる
        last_messages = messages[-10:] if len(messages) >= 10 else messages
        
        for j, msg in enumerate(last_messages):
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            
            # 数値が含まれているかチェック
            import re
            all_numbers = re.findall(r'\d+(?:\.\d+)?', content)
            
            if all_numbers:
                print(f"    メッセージ {len(messages)-len(last_messages)+j} (role={role}):")
                print(f"      含まれる数値: {all_numbers}")
                
                # 0-5の範囲の数値をチェック
                valid_scores = [n for n in all_numbers if 0 <= float(n) <= 5]
                if valid_scores:
                    print(f"      有効なスコア候補: {valid_scores}")
                
                # スコア関連の文脈を確認
                if any(keyword in content for keyword in ['点', 'スコア', '評価', '満足度']):
                    print(f"      🔍 スコア関連の文脈を含む")
                    print(f"      内容: {content[:200]}...")
    
    def _analyze_key_value(self, key: str, value: Any, sample_num: int):
        """キーと値の詳細分析"""
        print(f"\n【{key}】")
        print(f"  型: {type(value).__name__}")
        
        if isinstance(value, dict):
            print(f"  辞書サイズ: {len(value)}個のキー")
            print(f"  キー一覧: {list(value.keys())}")
            
            # 重要なキーの内容を詳細表示
            if key in ['review', 'evaluation', 'score', 'rating', 'annotation', 'metadata']:
                print(f"  内容:")
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, (int, float)):
                        print(f"    {sub_key}: {sub_value}")
                    elif isinstance(sub_value, str):
                        print(f"    {sub_key}: {sub_value[:100]}...")
                    else:
                        print(f"    {sub_key}: {type(sub_value).__name__}")
        
        elif isinstance(value, list):
            print(f"  リストサイズ: {len(value)}個の要素")
            
            if key == 'messages':
                print(f"  メッセージ詳細:")
                for j, msg in enumerate(value[:5]):  # 最初の5つ
                    role = msg.get('role', 'unknown')
                    content = msg.get('content', '')
                    print(f"    {j}: role={msg.get('role')}")
                    print(f"      content: {msg.get('content', '')[:200]}...")
                
                if len(value) > 5:
                    print(f"    ... 他 {len(value)-5}個のメッセージ")
            
            elif key in ['annotations', 'evaluations', 'scores']:
                print(f"  内容（最初の3つ）:")
                for j, item in enumerate(value[:3]):
                    if isinstance(item, dict):
                        print(f"    {j}: {list(item.keys())}")
                    else:
                        print(f"    {j}: {item}")
        
        elif isinstance(value, (int, float)):
            print(f"  値: {value}")
            if key.lower() in ['score', 'rating', 'satisfaction', 'evaluation']:
                print(f"  ⭐ スコア関連の可能性が高い！")
        
        elif isinstance(value, str):
            print(f"  値: {value[:200]}...")
            # 数値が含まれているかチェック
            import re
            numbers = re.findall(r'\d+(?:\.\d+)?', value)
            if numbers:
                print(f"  含まれる数値: {numbers}")
                # 0-5の範囲の数値をチェック
                valid_scores = [n for n in numbers if 0 <= float(n) <= 5]
                if valid_scores:
                    print(f"  有効なスコア候補: {valid_scores}")
    
    def _investigate_missing_score(self, sample: Dict[str, Any], sample_num: int):
        """正解スコアが欠損しているサンプルの詳細調査"""
        print(f"\n🔍 正解スコア欠損の原因調査（サンプル{sample_num}）")
        
        # 数値が含まれている可能性のあるキーを探す
        potential_scores = []
        
        for key, value in sample.items():
            if isinstance(value, (int, float)):
                if 0 <= value <= 5:
                    potential_scores.append((key, value, "直接的な数値"))
            
            elif isinstance(value, str):
                import re
                numbers = re.findall(r'\d+(?:\.\d+)?', value)
                for num in numbers:
                    if 0 <= float(num) <= 5:
                        potential_scores.append((key, float(num), f"文字列内の数値: {value[:50]}..."))
            
            elif isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, (int, float)) and 0 <= sub_value <= 5:
                        potential_scores.append((f"{key}.{sub_key}", sub_value, "ネストした数値"))
        
        if potential_scores:
            print(f"  🎯 スコア候補を発見:")
            for key, score, reason in potential_scores:
                print(f"    {key}: {score} ({reason})")
        else:
            print(f"  ❌ スコア候補が見つかりません")
        
        # サンプルの全体的な構造を再確認
        print(f"\n  📋 サンプル全体の構造:")
        for key, value in sample.items():
            if isinstance(value, dict):
                print(f"    {key}: {list(value.keys())}")
            elif isinstance(value, list):
                print(f"    {key}: {len(value)}個の要素")
            else:
                print(f"    {key}: {type(value).__name__}")
    
    def analyze_score_extraction_patterns(self):
        """正解スコア抽出パターンの分析（kokorochatの正しい構造に対応）"""
        logger.info("正解スコア抽出パターンを分析します")
        
        # テストデータファイルを探す
        test_files = list(self.output_dir.glob("test_data_*.jsonl"))
        if not test_files:
            logger.error("テストデータファイルが見つかりません")
            return
        
        latest_test_file = max(test_files, key=lambda x: x.stat().st_mtime)
        
        # 元のkokorochatデータファイルも確認
        original_files = list(self.output_dir.glob("*kokorochat*.jsonl"))
        
        # 成功・失敗パターンを分析
        success_patterns = []
        failure_patterns = []
        
        try:
            with open(latest_test_file, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i >= 50:  # 最初の50サンプルのみ
                        break
                    
                    sample = json.loads(line.strip())
                    
                    # 正解スコアを抽出してみる
                    score = self._extract_score_attempt(sample)
                    
                    if score is not None:
                        success_patterns.append({
                            'sample_index': i,
                            'score': score,
                            'keys': list(sample.keys()),
                            'structure': self._get_sample_structure_summary(sample)
                        })
                    else:
                        failure_patterns.append({
                            'sample_index': i,
                            'keys': list(sample.keys()),
                            'structure': self._get_sample_structure_summary(sample)
                        })
        
        except Exception as e:
            logger.error(f"パターン分析中にエラー: {e}")
            return
        
        # 結果を表示
        print(f"\n{'='*80}")
        print(f"正解スコア抽出パターン分析結果")
        print(f"{'='*80}")
        
        print(f"\n✅ 成功パターン: {len(success_patterns)}サンプル")
        for pattern in success_patterns[:5]:  # 最初の5つ
            print(f"  サンプル{pattern['sample_index']}: スコア{pattern['score']}")
            print(f"    キー: {pattern['keys']}")
            print(f"    構造: {pattern['structure']}")
        
        print(f"\n❌ 失敗パターン: {len(failure_patterns)}サンプル")
        for pattern in failure_patterns[:5]:  # 最初の5つ
            print(f"  サンプル{pattern['sample_index']}")
            print(f"    キー: {pattern['keys']}")
            print(f"    構造: {pattern['structure']}")
        
        # 成功・失敗の傾向分析
        if success_patterns:
            print(f"\n📊 成功パターンの傾向:")
            common_keys = set.intersection(*[set(p['keys']) for p in success_patterns])
            print(f"  共通キー: {common_keys}")
        
        if failure_patterns:
            print(f"\n📊 失敗パターンの傾向:")
            missing_keys = set.intersection(*[set(p['keys']) for p in failure_patterns])
            print(f"  共通キー: {missing_keys}")
        
        # 元のkokorochatデータとの比較
        if original_files:
            print(f"\n{'='*80}")
            print(f"元のkokorochatデータとの比較")
            print(f"{'='*80}")
            
            try:
                with open(original_files[0], 'r', encoding='utf-8') as f:
                    original_sample = json.loads(f.readline().strip())
                    print(f"元のデータ構造: {list(original_sample.keys())}")
                    
                    if 'review_by_client_jp' in original_sample:
                        review_data = original_sample['review_by_client_jp']
                        print(f"評価項目数: {len(review_data)}")
                        print(f"評価項目例:")
                        for i, (key, value) in enumerate(review_data.items()):
                            if i < 5:  # 最初の5項目
                                print(f"  {key}: {value}")
                            else:
                                break
                        
                        # スコアの統計
                        scores = [v for v in review_data.values() if isinstance(v, (int, float))]
                        if scores:
                            print(f"スコア統計:")
                            print(f"  最小値: {min(scores)}")
                            print(f"  最大値: {max(scores)}")
                            print(f"  平均値: {sum(scores)/len(scores):.2f}")
                            print(f"  スコア範囲: 0-5の範囲内" if all(0 <= s <= 5 for s in scores) else "スコア範囲: 0-5の範囲外")
                    
                    if 'dialogue' in original_sample:
                        dialogue = original_sample['dialogue']
                        print(f"対話データ: {len(dialogue)}個のターン")
                    
                    if 'topic' in original_sample:
                        topic = original_sample['topic']
                        print(f"トピック: {topic}")
                
            except Exception as e:
                print(f"元のデータ分析エラー: {e}")
        
        # 問題の特定と解決策
        print(f"\n{'='*80}")
        print(f"問題の特定と解決策")
        print(f"{'='*80}")
        
        if failure_patterns:
            print(f"❌ 問題: {len(failure_patterns)}サンプルで正解スコアの抽出に失敗")
            print(f"   原因: データ変換時にreview_by_client_jpが失われている")
            print(f"   影響: モデルの評価が不可能")
            
            print(f"\n💡 解決策:")
            print(f"   1. データ変換スクリプトの修正")
            print(f"      - review_by_client_jpを保持する")
            print(f"      - dialogueをmessagesに変換する際の処理を確認")
            print(f"   2. 正解スコア抽出ロジックの修正")
            print(f"      - review_by_client_jpからスコアを抽出")
            print(f"      - 各評価項目の平均スコアを計算")
            print(f"   3. データ品質の確認")
            print(f"      - 変換前後のデータ構造を比較")
            print(f"      - 正解スコアの範囲（0-5）を確認")
        else:
            print(f"✅ 問題なし: すべてのサンプルで正解スコアの抽出に成功")
    
    def _extract_score_attempt(self, sample: Dict[str, Any]) -> float:
        """サンプルからスコアを抽出してみる（テスト用）"""
        # 複数の方法でスコアを抽出
        methods = [
            self._extract_from_messages,
            self._extract_from_metadata,
            self._extract_from_content,
            self._extract_from_annotations,
            self._extract_from_kokorochat_specific
        ]
        
        for method in methods:
            try:
                score = method(sample)
                if score is not None:
                    return score
            except:
                continue
        
        return None
    
    def _get_sample_structure_summary(self, sample: Dict[str, Any]) -> str:
        """サンプルの構造を簡潔に要約"""
        summary = []
        for key, value in sample.items():
            if isinstance(value, dict):
                summary.append(f"{key}(dict:{len(value)})")
            elif isinstance(value, list):
                summary.append(f"{key}(list:{len(value)})")
            else:
                summary.append(f"{key}({type(value).__name__})")
        return ", ".join(summary)
    
    # 既存の抽出メソッド（analyze_finetuned_model.pyからコピー）
    def _extract_from_messages(self, sample: Dict[str, Any]) -> float:
        """メッセージから正解スコアを抽出"""
        if 'messages' not in sample:
            return None
        
        messages = sample['messages']
        
        # 最後のメッセージからスコアを探す
        for msg in reversed(messages):
            if msg.get('role') == 'assistant':
                content = msg.get('content', '')
                # 数値スコアを探す
                import re
                match = re.search(r'(\d+(?:\.\d+)?)', content)
                if match:
                    score = float(match.group(1))
                    if 0 <= score <= 5:
                        return score
        
        return None
    
    def _extract_from_metadata(self, sample: Dict[str, Any]) -> float:
        """メタデータから正解スコアを抽出"""
        if 'metadata' not in sample:
            return None
        
        metadata = sample['metadata']
        
        # スコア関連のキーを探す
        score_keys = ['score', 'rating', 'satisfaction', 'evaluation']
        for key in score_keys:
            if key in metadata:
                try:
                    score = float(metadata[key])
                    if 0 <= score <= 5:
                        return score
                except (ValueError, TypeError):
                    continue
        
        return None
    
    def _extract_from_content(self, sample: Dict[str, Any]) -> float:
        """コンテンツから正解スコアを抽出"""
        # サンプル全体のテキストからスコアを探す
        sample_text = json.dumps(sample, ensure_ascii=False)
        
        # 数値スコアを探す
        import re
        matches = re.findall(r'(\d+(?:\.\d+)?)', sample_text)
        for match in matches:
            try:
                score = float(match)
                if 0 <= score <= 5:
                    return score
            except ValueError:
                continue
        
        return None
    
    def _extract_from_annotations(self, sample: Dict[str, Any]) -> float:
        """アノテーションから正解スコアを抽出"""
        annotation_keys = ['annotation', 'label', 'ground_truth', 'reference']
        
        for key in annotation_keys:
            if key in sample:
                annotation = sample[key]
                if isinstance(annotation, dict):
                    # スコア関連のキーを探す
                    score_keys = ['score', 'rating', 'satisfaction', 'evaluation']
                    for score_key in score_keys:
                        if score_key in annotation:
                            try:
                                score = float(annotation[score_key])
                                if 0 <= score <= 5:
                                    return score
                            except (ValueError, TypeError):
                                continue
                elif isinstance(annotation, (int, float)):
                    if 0 <= annotation <= 5:
                        return float(annotation)
        
        return None
    
    def _extract_from_kokorochat_specific(self, sample: Dict[str, Any]) -> float:
        """kokorochat特有の構造から正解スコアを抽出"""
        # kokorochatでよく使われるキーを探す
        kokorochat_keys = [
            'kokorochat_score', 'kokorochat_rating', 'kokorochat_evaluation',
            'ground_truth_score', 'ground_truth_rating', 'ground_truth_evaluation',
            'human_score', 'human_rating', 'human_evaluation',
            'expert_score', 'expert_rating', 'expert_evaluation',
            'reference_score', 'reference_rating', 'reference_evaluation'
        ]
        
        for key in kokorochat_keys:
            if key in sample:
                try:
                    score = float(sample[key])
                    if 0 <= score <= 5:
                        return score
                except (ValueError, TypeError):
                    continue
        
        # ネストした構造も探す
        for key in ['data', 'annotation', 'evaluation', 'score']:
            if key in sample and isinstance(sample[key], dict):
                for sub_key in kokorochat_keys:
                    if sub_key in sample[key]:
                        try:
                            score = float(sample[key][sub_key])
                            if 0 <= score <= 5:
                                return score
                        except (ValueError, TypeError):
                            continue
        
        return None

    def _identify_data_conversion_issues(self):
        """データ変換の問題を特定"""
        logger.info("\n=== データ変換の問題特定 ===")
        
        # 元のデータと変換後のデータを比較
        original_files = list(self.output_dir.glob("*kokorochat*.jsonl"))
        converted_files = list(self.output_dir.glob("train_data_*.jsonl"))
        
        if original_files and converted_files:
            logger.info("データ変換の前後を比較します")
            
            # 元のデータの構造を確認
            original_structure = self._get_file_structure_summary(original_files[0])
            logger.info(f"元のデータ構造: {original_structure}")
            
            # 変換後のデータの構造を確認
            converted_structure = self._get_file_structure_summary(converted_files[0])
            logger.info(f"変換後のデータ構造: {converted_structure}")
            
            # 問題の特定
            if 'review_by_client_jp' in original_structure and 'review_by_client_jp' not in converted_structure:
                logger.error("❌ 重大な問題: review_by_client_jpが変換で失われています")
                logger.error("  これが正解スコアが抽出できない直接の原因です")
            
            if 'dialogue' in original_structure and 'dialogue' not in converted_structure:
                logger.warning("⚠️  dialogueキーが変換で失われています")
                logger.warning("  元の会話構造が失われています")
            
            if 'topic' in original_structure and 'topic' not in converted_structure:
                logger.warning("⚠️  topicキーが変換で失われています")
                logger.warning("  トピック情報が失われています")
            
            # 推奨される修正方法
            logger.info("\n--- 推奨される修正方法 ---")
            logger.info("1. データ変換スクリプトでreview_by_client_jpを保持する")
            logger.info("2. dialogueキーをmessagesに変換する際の処理を確認する")
            logger.info("3. 正解スコアの抽出ロジックを修正する")
        
        else:
            logger.warning("元のデータまたは変換後のデータが見つからないため、比較できません")
    
    def _get_file_structure_summary(self, file_path: Path) -> Dict[str, int]:
        """ファイルの構造を要約"""
        structure_summary = {}
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i >= 10:  # 最初の10サンプルのみ
                        break
                    
                    sample = json.loads(line.strip())
                    for key in sample.keys():
                        if key not in structure_summary:
                            structure_summary[key] = 0
                        structure_summary[key] += 1
        except Exception as e:
            logger.error(f"ファイル構造分析エラー: {e}")
        
        return structure_summary

def main():
    """メイン関数"""
    # 出力ディレクトリを確認
    output_dir = "openai_sft_outputs"
    if not os.path.exists(output_dir):
        logger.error(f"出力ディレクトリ {output_dir} が存在しません")
        return
    
    # 分析器を初期化
    analyzer = TestDataStructureAnalyzer(output_dir)
    
    # 1. データ構造の妥当性を検証
    logger.info("=== データ構造の妥当性検証 ===")
    analyzer.validate_data_structure()
    
    # 2. スコア比較用のCSVを生成
    logger.info("\n=== スコア比較用CSVの生成 ===")
    csv_path = analyzer.generate_score_comparison_csv()
    
    # 3. テストデータの構造を詳細分析
    logger.info("\n=== テストデータの構造詳細分析 ===")
    analyzer.analyze_test_data_structure(max_samples=5)
    
    # 4. メッセージ内から正解スコアを探す詳細分析
    logger.info("\n=== メッセージ内から正解スコアを探す詳細分析 ===")
    analyzer.analyze_messages_for_scores(max_samples=5)
    
    # 5. メッセージの構造から正解スコアを探す詳細分析
    logger.info("\n=== メッセージの構造から正解スコアを探す詳細分析 ===")
    analyzer.analyze_message_structure_for_scores(max_samples=5)
    
    # 6. 正解スコア抽出パターンの分析
    logger.info("\n=== 正解スコア抽出パターンの分析 ===")
    analyzer.analyze_score_extraction_patterns()

if __name__ == "__main__":
    main()