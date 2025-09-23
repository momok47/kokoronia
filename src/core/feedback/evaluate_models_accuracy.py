#!/usr/bin/env python3
"""
今回作成した4つのファインチューニング済みモデルのtestデータでの正解率を計算するスクリプト
会話の得点（0~5点）を予測し、正解の点数と比較する機能付き
"""

import os
import json
import logging
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
import time
from typing import List, Dict, Any, Tuple
import numpy as np
from datetime import datetime
import pandas as pd
import re

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

class ModelAccuracyEvaluator:
    """ファインチューニング済みモデルの正解率評価クラス"""
    
    def __init__(self, api_key: str):
        """
        初期化
        
        Args:
            api_key: OpenAI APIキー
        """
        self.client = OpenAI(api_key=api_key)
        # どの階層から実行されても正しくパスを指定できるように、スクリプト自身の場所を基準にする
        self.script_dir = Path(__file__).resolve().parent
        self.output_dir = self.script_dir / "openai_sft_outputs"
        logger.info(f"結果ディレクトリを設定: {self.output_dir}")
        self.output_dir.mkdir(exist_ok=True) # ディレクトリがなければ作成

    def _find_latest_results_file(self) -> Path:
        """最新のバッチ結果ファイルを見つける"""
        result_files = list(self.output_dir.glob("batch_fine_tuning_results_*.json"))
        if not result_files:
            raise FileNotFoundError(f"バッチ結果ファイルがディレクトリに見つかりません: {self.output_dir}")
        
        latest_file = max(result_files, key=lambda x: x.stat().st_mtime)
        logger.info(f"使用する結果ファイル: {latest_file}")
        return latest_file
    
    def load_test_data_and_models(self) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        最新の結果ファイルからモデルIDとテストデータを読み込む
        
        Returns:
            (モデルIDのリスト, testデータのリスト)
        """
        latest_results_file = self._find_latest_results_file()
        
        with open(latest_results_file, 'r', encoding='utf-8') as f:
            results_data = json.load(f)

        # モデルIDの読み込み - データ構造に応じて修正
        model_ids = []
        if 'batches' in results_data:
            # 新しい形式: batchesキーがある場合
            for batch in results_data['batches']:
                if 'fine_tuned_model' in batch:
                    model_ids.append(batch['fine_tuned_model'])
        elif 'batch_results' in results_data:
            # 古い形式: batch_resultsキーがある場合
            for batch in results_data['batch_results']:
                if batch.get('final_status') == 'succeeded' and 'final_model_id' in batch:
                    model_ids.append(batch['final_model_id'])
        else:
            # その他の形式を試す
            logger.warning(f"予期しないデータ構造です: {list(results_data.keys())}")
            # 直接モデルIDが含まれている可能性をチェック
            for key, value in results_data.items():
                if isinstance(value, list) and value:
                    for item in value:
                        if isinstance(item, dict):
                            if 'fine_tuned_model' in item:
                                model_ids.append(item['fine_tuned_model'])
                            elif 'final_model_id' in item:
                                model_ids.append(item['final_model_id'])
        
        if not model_ids:
            raise ValueError(f"結果ファイル {latest_results_file.name} からモデルIDを取得できませんでした。")
        
        logger.info(f"評価対象モデル数: {len(model_ids)}")
        for i, model_id in enumerate(model_ids):
            logger.info(f"  モデル {i+1}: {model_id}")

        # テストデータファイルのパスを取得 - 複数の方法を試す
        test_data_path = None
        
        # 方法1: test_data_fileキーから取得
        if 'test_data_file' in results_data:
            test_data_filename = results_data['test_data_file']
            test_data_path = self.output_dir / test_data_filename
            logger.info(f"test_data_fileキーから取得: {test_data_filename}")
        
        # 方法2: ファイル名から推測
        if not test_data_path or not test_data_path.exists():
            test_files = list(self.output_dir.glob("test_data_*.jsonl"))
            if test_files:
                # 最新のtestデータファイルを選択
                latest_test_file = max(test_files, key=lambda x: x.stat().st_mtime)
                test_data_path = latest_test_file
                logger.info(f"ファイル名から推測: {test_data_path.name}")
        
        # 方法3: ファイル名から推測（より柔軟に）
        if not test_data_path or not test_data_path.exists():
            test_files = list(self.output_dir.glob("*test*data*.jsonl"))
            if test_files:
                # 最新のtestデータファイルを選択
                latest_test_file = max(test_files, key=lambda x: x.stat().st_mtime)
                test_data_path = latest_test_file
                logger.info(f"柔軟な検索で発見: {test_data_path.name}")
        
        if not test_data_path or not test_data_path.exists():
            raise FileNotFoundError(f"テストデータファイルが見つかりません。ディレクトリ: {self.output_dir}")
        
        logger.info(f"使用するtestデータファイル: {test_data_path}")

        # テストデータの読み込み
        test_data = []
        try:
            with open(test_data_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:  # 空行をスキップ
                        try:
                            test_data.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            logger.warning(f"行 {line_num} のJSON解析に失敗: {e}")
                            continue
            
            logger.info(f"testデータ読み込み完了: {len(test_data)}サンプル")
            
            # データ構造の確認
            if test_data:
                sample_keys = list(test_data[0].keys())
                logger.info(f"テストデータの構造: {sample_keys}")
                
                # messagesキーの存在確認
                if 'messages' in test_data[0]:
                    first_messages = test_data[0]['messages']
                    if first_messages:
                        logger.info(f"最初のメッセージ構造: {[msg.get('role', 'unknown') for msg in first_messages]}")
        
        except Exception as e:
            logger.error(f"テストデータの読み込み中にエラー: {e}")
            raise
        
        return model_ids, test_data

    def extract_score_from_response(self, response_text: str) -> Tuple[float, str, bool, Dict[int, float]]:
        """
        応答テキストから確率分布と期待値を抽出
        """
        logger.debug(f"元のテキスト: {response_text}")
        
        probability_patterns = [
            r'(\d+)点\s*[:：]\s*(\d+(?:\.\d+)?)%',
            r'(\d+)点\s+(\d+(?:\.\d+)?)%',
        ]
        
        probabilities = {}
        for pattern in probability_patterns:
            matches = re.findall(pattern, response_text)
            for match in matches:
                point, prob = int(match[0]), float(match[1])
                if 0 <= point <= 5 and point not in probabilities:
                    probabilities[point] = prob
        
        total_probability = sum(probabilities.values())
        
        if len(probabilities) == 6 and abs(total_probability - 100.0) < 10.0:
            expected_value = sum(p * (pr / 100.0) for p, pr in probabilities.items())
            return expected_value, "抽出成功", False, probabilities
        else:
            logger.warning(f"確率分布の形式が不正: 合計={total_probability}%, 数={len(probabilities)}")
            return 0.0, "response error", True, {}

    def evaluate_model(self, model_id: str, test_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """1つのモデルを評価"""
        logger.info(f"モデル {model_id} の評価を開始...")
        score_comparison = []

        for i, sample in enumerate(test_data):
            try:
                # ユーザーメッセージを取得
                user_message = None
                for msg in sample['messages']:
                    if msg['role'] == 'user':
                        user_message = msg['content']
                        break
                
                if not user_message:
                    logger.warning(f"サンプル {i+1}: ユーザーメッセージが見つかりません")
                    continue
                
                # モデルに質問を送信（確率分布形式の得点予測用）
                score_response = self.client.chat.completions.create(
                    model=model_id,
                    messages=[
                        {"role": "system", "content": "あなたは心理カウンセリングの評価者です。相談者の満足度を0～5点で評価し、各点数の確率を出力してください。"},
                        {"role": "user", "content": f"""### 指示
以下の対話について「相談者の満足度」を相談者の視点で0～5点で評価し、各点数の確率を出力してください。

### 対話
{user_message}

### 出力形式（数値のみ）
0点: XX%
1点: XX%
2点: XX%
3点: XX%
4点: XX%
5点: XX%"""}
                    ],
                    max_tokens=200,
                    temperature=0.7
                )
                
                response_text = score_response.choices[0].message.content.strip()
                time.sleep(1)
            except Exception as e:
                logger.error(f"サンプル {i+1}/{len(test_data)}: API呼び出しエラー: {e}")
                response_text = "API error"

            expected_value, _, is_error, probabilities = self.extract_score_from_response(response_text)
            
            icon = "❌" if is_error else "✅"
            log_msg = f"予測得点: {expected_value:.1f}点" if not is_error else "response error"
            logger.info(f"サンプル {i+1}/{len(test_data)}: {icon} ({log_msg})")
            
            if is_error:
                 logger.info(f"   応答テキスト: {response_text}")

            try:
                correct_score_str = sample["messages"][-1]["content"]
                correct_score = float(re.search(r'(\d+(?:\.\d+)?)', correct_score_str).group(1))
            except (AttributeError, IndexError, ValueError):
                correct_score = -1

            score_comparison.append({
                "model_id": model_id, "sample_index": i,
                "predicted_score": expected_value if not is_error else None,
                "correct_score": correct_score, "is_error": is_error,
            })

        errors = sum(1 for s in score_comparison if s['is_error'])
        valid_preds = [s for s in score_comparison if not s['is_error']]
        mae = np.mean([abs(s['predicted_score'] - s['correct_score']) for s in valid_preds]) if valid_preds else 0
        
        return {"model_id": model_id, "total_samples": len(test_data), "response_errors": errors,
                "mean_absolute_error": mae, "score_comparison": score_comparison}

    def evaluate_all_models(self, max_test_samples: int = None):
        """全モデルを評価"""
        try:
            model_ids, test_data = self.load_test_data_and_models()
        except (FileNotFoundError, KeyError, IndexError) as e:
            logger.error(e)
            return
            
        if max_test_samples:
            test_data = test_data[:max_test_samples]
            logger.info(f"評価サンプル数を {max_test_samples} に制限")
        
        all_results = [self.evaluate_model(model_id, test_data) for i, model_id in enumerate(model_ids)]
        
        df = pd.DataFrame([item for res in all_results for item in res['score_comparison']])
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / f"accuracy_evaluation_results_{ts}.csv"
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        logger.info(f"評価結果を {output_path} に保存しました")

        self.print_summary(all_results)

    def print_summary(self, all_results: List[Dict[str, Any]]):
        """評価結果のサマリーを表示"""
        print("\n--- 📊 評価結果サマリー 📊 ---\n")
        summary = [{"Model ID": r['model_id'],
                      "MAE (平均絶対誤差)": f"{r['mean_absolute_error']:.3f}",
                      "Error率": f"{(r['response_errors']/r['total_samples']*100):.1f}%"}
                     for r in all_results]
        print(pd.DataFrame(summary).to_string(index=False))
        print("\n--------------------------\n")

def main():
    """メイン関数"""
    try:
        project_root = Path(__file__).resolve().parent.parent.parent.parent
        env_path = project_root / ".env"
        if env_path.exists():
            load_dotenv(env_path)
            logger.info(f".envファイルを読み込みました: {env_path}")
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI APIキーが.envファイルに設定されていません。")
        
        print("�� ファインチューニング済みモデルの正解率評価を開始します")
        evaluator = ModelAccuracyEvaluator(api_key)
        evaluator.evaluate_all_models()
        print(f"\n✅ 評価が完了しました！")

    except Exception as e:
        logger.error(f"処理中にエラーが発生しました: {e}", exc_info=True)

if __name__ == "__main__":
    main()