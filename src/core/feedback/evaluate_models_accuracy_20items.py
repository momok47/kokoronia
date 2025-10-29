#!/usr/bin/env python3
"""
20項目の会話印象評価に対する予測精度計算スクリプト（evaluate_models_accuracy.pyベース）
Kokorochatデータから150件を抽出し、各項目ごとにMAE、RMSE、誤差1での正解率を計算する機能付き
"""

import os
import json
import logging
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
import time
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from datetime import datetime
import pandas as pd
import re
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datasets import load_dataset
import random
import argparse

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

# 20項目の評価指標
EVALUATION_ITEMS = [
    "聴いてもらえた、わかってもらえたと感じた",
    "尊重されたと感じた",
    "新しい気づきや体験があった",
    "希望や期待を感じられた",
    "取り組みたかったことを扱えた",
    "一緒に考えながら取り組めた",
    "やりとりのリズムがあっていた",
    "居心地のよいやりとりだった",
    "全体として適切でよかった",
    "今回の相談は価値があった",
    "相談開始の円滑さ",
    "相談終了のタイミング（不必要に聴きすぎていないか）、円滑さ",
    "受容・共感",
    "肯定・承認",
    "的確な質問による会話の促進",
    "要約",
    "問題の明確化",
    "この相談での目標の明確化",
    "次の行動につながる提案",
    "勇気づけ・希望の喚起"
]


class MultiItemModelAccuracyEvaluator:
    """20項目評価対応のファインチューニング済みモデル正解率評価クラス"""
    
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

    def load_kokorochat_test_dataset(self, max_samples: Optional[int] = 1500, seed: int = 42) -> List[Dict]:
        """
        Kokoro Chatデータセットを読み込み、テストデータのみを抽出する（1500件の10%=150件）
        
        Args:
            max_samples: 最大サンプル数（デフォルト: 1500）
            seed: 再現性のための乱数シード
            
        Returns:
            test_data: テストデータのリスト
        """
        logger.info("KokoroChat データセットを読み込み中...")
        dataset = load_dataset("UEC-InabaLab/KokoroChat", split="train")
        
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
            logger.info(f"デバッグ用に{max_samples}サンプルに制限")
        
        logger.info(f"元データセットサイズ: {len(dataset)}")
        
        # 有効なデータのインデックスを取得（review_by_client_jpが存在するもの）
        valid_indices = []
        for i, sample in enumerate(dataset):
            if 'review_by_client_jp' in sample and sample['review_by_client_jp']:
                review_data = sample['review_by_client_jp']
                # 20項目の評価データが存在するかチェック
                valid_items = sum(1 for item in EVALUATION_ITEMS if item in review_data and isinstance(review_data[item], (int, float)))
                if valid_items >= 15:  # 20項目中15項目以上あれば有効とする
                    valid_indices.append(i)
        
        logger.info(f"有効なサンプル数: {len(valid_indices)}")
        
        if len(valid_indices) == 0:
            raise ValueError("有効なデータが見つかりませんでした。")
        
        # インデックスをシャッフル
        random.seed(seed)
        shuffled_indices = random.sample(valid_indices, len(valid_indices))
        
        # テストデータのみを抽出（全体の10%）
        total_size = len(shuffled_indices)
        train_size = int(total_size * 0.8)
        test_size = int(total_size * 0.1)
        
        # テストデータのインデックスを取得
        test_indices = shuffled_indices[train_size:train_size + test_size]
        test_dataset = dataset.select(test_indices)
        
        logger.info("=== テストデータ抽出結果 ===")
        logger.info(f"有効データ: {total_size} サンプル")
        logger.info(f"テストデータ: {len(test_dataset)} サンプル ({len(test_dataset)/total_size*100:.1f}%)")
        logger.info(f"ランダムシード: {seed}")
        
        # テストデータをリスト形式に変換
        test_data = self._convert_dataset_to_list(test_dataset)
        
        return test_data

    def _convert_dataset_to_list(self, dataset) -> List[Dict[str, Any]]:
        """Hugging Faceデータセットをリスト形式に変換"""
        converted_data = []
        for sample in dataset:
            # 会話テキストを抽出
            conversation_text = self._extract_conversation_text(sample["dialogue"])
            
            # 正解スコアを抽出
            correct_scores = self._extract_correct_scores(sample)
            
            if correct_scores:
                converted_data.append({
                    "conversation_text": conversation_text,
                    "correct_scores": correct_scores,
                    "original_sample": sample
                })
        
        return converted_data

    def _extract_conversation_text(self, dialogue: List[Dict]) -> str:
        """対話データから会話テキストを抽出"""
        conversation_parts = []
        for turn in dialogue:
            role = "相談者" if turn["role"] == "client" else "カウンセラー"
            conversation_parts.append(f"{role}: {turn['utterance']}")
        return "\n".join(conversation_parts)

    def _extract_correct_scores(self, sample: Dict[str, Any]) -> Dict[str, float]:
        """サンプルから20項目の正解スコアを抽出"""
        correct_scores = {}
        review_data = sample.get('review_by_client_jp', {})
        
        for item in EVALUATION_ITEMS:
            if item in review_data and isinstance(review_data[item], (int, float)):
                correct_scores[item] = float(review_data[item])
        
        return correct_scores

    def _find_latest_results_file(self) -> Path:
        """最新のバッチ結果ファイルを見つける"""
        result_files = list(self.output_dir.glob("batch_fine_tuning_results_*.json"))
        if not result_files:
            raise FileNotFoundError(f"バッチ結果ファイルがディレクトリに見つかりません: {self.output_dir}")
        
        latest_file = max(result_files, key=lambda x: x.stat().st_mtime)
        logger.info(f"使用する結果ファイル: {latest_file}")
        return latest_file
    
    def load_test_data_and_models(self, use_kokorochat: bool = True, max_samples: int = 1500, seed: int = 42) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        モデルIDとテストデータを読み込む
        
        Args:
            use_kokorochat: Kokorochatデータを直接使用するか
            max_samples: 最大サンプル数
            seed: ランダムシード
            
        Returns:
            (モデルIDのリスト, testデータのリスト)
        """
        if use_kokorochat:
            # Kokorochatデータを直接使用
            logger.info("Kokorochatデータを直接使用します")
            test_data = self.load_kokorochat_test_dataset(max_samples=max_samples, seed=seed)
            
            # デフォルトモデルIDを設定（実際のファインチューニング済みモデルがない場合）
            model_ids = ["gpt-4o-mini"]  # デフォルトモデル
            
            return model_ids, test_data
        else:
            # 従来の方法（バッチ結果ファイルから読み込み）
            return self._load_from_batch_results()

    def _load_from_batch_results(self) -> Tuple[List[str], List[Dict[str, Any]]]:
        """バッチ結果ファイルからモデルIDとテストデータを読み込む（従来の方法）"""
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

    def evaluate_model_on_all_items(self, model_id: str, test_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """1つのモデルを20項目すべてで評価"""
        logger.info(f"モデル {model_id} の20項目評価を開始...")
        
        all_predictions = []
        
        for sample_idx, sample in enumerate(test_data):
            try:
                # 会話テキストを取得
                conversation_text = sample.get("conversation_text", "")
                if not conversation_text:
                    logger.warning(f"サンプル {sample_idx+1}: 会話テキストが見つかりません")
                    continue
                
                # 正解データを取得
                correct_scores = sample.get("correct_scores", {})
                if not correct_scores:
                    logger.warning(f"サンプル {sample_idx+1}: 正解データが取得できませんでした")
                    continue
                
                sample_predictions = {
                    "sample_index": sample_idx,
                    "conversation_text": conversation_text[:200] + "...",  # 表示用に短縮
                    "predictions": {},
                    "correct_scores": correct_scores,
                    "errors": {}
                }
                
                # 各評価項目について予測を実行
                for item_idx, evaluation_item in enumerate(EVALUATION_ITEMS):
                    if evaluation_item not in correct_scores:
                        logger.debug(f"項目 '{evaluation_item}' の正解データがありません")
                        continue
                        
                    try:
                        logger.info(f"サンプル {sample_idx+1}/{len(test_data)}, 項目 {item_idx+1}/{len(EVALUATION_ITEMS)}: {evaluation_item}")
                        
                        # モデルに質問を送信
                        response = self.client.chat.completions.create(
                            model=model_id,
                            messages=[
                                {"role": "system", "content": "あなたは心理カウンセリングの評価者です。相談者の満足度を0～5点で評価し、各点数の確率を出力してください。"},
                                {"role": "user", "content": f"""### 指示
以下の対話について「{evaluation_item}」の満足度を相談者の視点で0～5点で評価し、各点数の確率を出力してください。

### 対話
{conversation_text}

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
                        
                        response_text = response.choices[0].message.content.strip()
                        expected_value, _, is_error, probabilities = self.extract_score_from_response(response_text)
                        
                        if not is_error:
                            sample_predictions["predictions"][evaluation_item] = expected_value
                        else:
                            sample_predictions["errors"][evaluation_item] = response_text
                            logger.warning(f"項目 '{evaluation_item}' の予測に失敗")
                        
                        time.sleep(1)  # API制限対策
                        
                    except Exception as e:
                        logger.error(f"項目 '{evaluation_item}' の評価中にエラー: {e}")
                        sample_predictions["errors"][evaluation_item] = str(e)
                
                all_predictions.append(sample_predictions)
                
            except Exception as e:
                logger.error(f"サンプル {sample_idx+1} の処理中にエラー: {e}")
                continue
        
        return {
            "model_id": model_id,
            "total_samples": len(test_data),
            "predictions": all_predictions
        }

    def calculate_metrics_per_item(self, predictions_data: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """項目ごとにMAE、RMSE、誤差1での正解率を計算"""
        logger.info("項目ごとの精度指標を計算中...")
        
        metrics_per_item = {}
        
        for evaluation_item in EVALUATION_ITEMS:
            predicted_scores = []
            correct_scores = []
            
            # 各サンプルから該当項目の予測値と正解値を収集
            for sample_pred in predictions_data["predictions"]:
                if evaluation_item in sample_pred["predictions"] and evaluation_item in sample_pred["correct_scores"]:
                    predicted_scores.append(sample_pred["predictions"][evaluation_item])
                    correct_scores.append(sample_pred["correct_scores"][evaluation_item])
            
            if len(predicted_scores) == 0:
                logger.warning(f"項目 '{evaluation_item}' の有効な予測データがありません")
                metrics_per_item[evaluation_item] = {
                    "mae": float('nan'),
                    "rmse": float('nan'),
                    "accuracy_within_1": float('nan'),
                    "sample_count": 0
                }
                continue
            
            # MAE（平均絶対誤差）
            mae = mean_absolute_error(correct_scores, predicted_scores)
            
            # RMSE（二乗平均平方根誤差）
            rmse = np.sqrt(mean_squared_error(correct_scores, predicted_scores))
            
            # 誤差1での正解率
            errors = np.abs(np.array(predicted_scores) - np.array(correct_scores))
            accuracy_within_1 = np.mean(errors <= 1.0) * 100  # パーセンテージ
            
            metrics_per_item[evaluation_item] = {
                "mae": mae,
                "rmse": rmse,
                "accuracy_within_1": accuracy_within_1,
                "sample_count": len(predicted_scores)
            }
            
            logger.info(f"項目 '{evaluation_item}': MAE={mae:.3f}, RMSE={rmse:.3f}, 誤差1正解率={accuracy_within_1:.1f}%")
        
        return metrics_per_item

    def evaluate_all_models(self, max_test_samples: int = None, use_kokorochat: bool = True, max_samples: int = 1500, seed: int = 42):
        """全モデルを20項目で評価"""
        try:
            model_ids, test_data = self.load_test_data_and_models(
                use_kokorochat=use_kokorochat, 
                max_samples=max_samples, 
                seed=seed
            )
        except (FileNotFoundError, KeyError, IndexError) as e:
            logger.error(e)
            return
            
        if max_test_samples:
            test_data = test_data[:max_test_samples]
            logger.info(f"評価サンプル数を {max_test_samples} に制限")
        
        all_results = []
        all_metrics = {}
        
        for model_id in model_ids:
            # モデルごとの評価実行
            predictions_data = self.evaluate_model_on_all_items(model_id, test_data)
            
            # 項目ごとの精度指標計算
            metrics_per_item = self.calculate_metrics_per_item(predictions_data)
            
            all_results.append(predictions_data)
            all_metrics[model_id] = metrics_per_item
        
        # 結果の保存
        self.save_multi_item_results(all_results, all_metrics)
        
        # サマリー表示
        self.print_multi_item_summary(all_metrics)

    def save_multi_item_results(self, all_results: List[Dict[str, Any]], all_metrics: Dict[str, Dict[str, Dict[str, float]]]):
        """20項目評価結果を保存"""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 詳細結果の保存
        detailed_output_path = self.output_dir / f"multi_item_detailed_results_{ts}.json"
        with open(detailed_output_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        logger.info(f"詳細結果を {detailed_output_path} に保存しました")
        
        # 精度指標の保存（CSV形式）
        metrics_rows = []
        for model_id, metrics_per_item in all_metrics.items():
            for evaluation_item, metrics in metrics_per_item.items():
                metrics_rows.append({
                    "model_id": model_id,
                    "evaluation_item": evaluation_item,
                    "mae": metrics["mae"],
                    "rmse": metrics["rmse"],
                    "accuracy_within_1": metrics["accuracy_within_1"],
                    "sample_count": metrics["sample_count"]
                })
        
        metrics_df = pd.DataFrame(metrics_rows)
        metrics_output_path = self.output_dir / f"multi_item_metrics_{ts}.csv"
        metrics_df.to_csv(metrics_output_path, index=False, encoding='utf-8-sig')
        logger.info(f"精度指標を {metrics_output_path} に保存しました")

    def print_multi_item_summary(self, all_metrics: Dict[str, Dict[str, Dict[str, float]]]):
        """20項目評価結果のサマリーを表示"""
        print("\n" + "="*80)
        print("📊 20項目評価結果サマリー 📊")
        print("="*80)
        
        for model_id, metrics_per_item in all_metrics.items():
            print(f"\n🤖 モデル: {model_id}")
            print("-" * 60)
            
            # 各項目の結果を表示
            valid_metrics = {k: v for k, v in metrics_per_item.items() if not np.isnan(v["mae"])}
            
            if not valid_metrics:
                print("❌ 有効な評価結果がありません")
                continue
            
            # 平均値を計算
            avg_mae = np.mean([m["mae"] for m in valid_metrics.values()])
            avg_rmse = np.mean([m["rmse"] for m in valid_metrics.values()])
            avg_accuracy = np.mean([m["accuracy_within_1"] for m in valid_metrics.values()])
            
            print(f"📈 全体平均:")
            print(f"   MAE (平均絶対誤差): {avg_mae:.3f}")
            print(f"   RMSE (二乗平均平方根誤差): {avg_rmse:.3f}")
            print(f"   誤差1での正解率: {avg_accuracy:.1f}%")
            print(f"   有効項目数: {len(valid_metrics)}/{len(EVALUATION_ITEMS)}")
            
            # 最良・最悪項目
            sorted_by_mae = sorted(valid_metrics.items(), key=lambda x: x[1]["mae"])
            
            print(f"\n🏆 MAE最良項目 TOP3:")
            for item, metrics in sorted_by_mae[:3]:
                print(f"   {metrics['mae']:.3f} - {item}")
            
            print(f"\n⚠️ MAE改善項目 TOP3:")
            for item, metrics in sorted_by_mae[-3:]:
                print(f"   {metrics['mae']:.3f} - {item}")
        
        print("\n" + "="*80)


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="20項目評価精度計算スクリプト（Kokorochatデータ対応）")
    parser.add_argument("--max-samples", type=int, default=1500, help="Kokorochatから抽出する最大サンプル数（デフォルト: 1500）")
    parser.add_argument("--max-test-samples", type=int, help="評価に使用するテストサンプル数の上限")
    parser.add_argument("--seed", type=int, default=42, help="ランダムシード（デフォルト: 42）")
    parser.add_argument("--use-batch-results", action="store_true", help="バッチ結果ファイルを使用（デフォルトはKokorochatデータを直接使用）")
    parser.add_argument("--debug", action="store_true", help="デバッグモード")
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        project_root = Path(__file__).resolve().parent.parent.parent.parent
        env_path = project_root / ".env"
        if env_path.exists():
            load_dotenv(env_path)
            logger.info(f".envファイルを読み込みました: {env_path}")
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI APIキーが.envファイルに設定されていません。")
        
        print("🚀 20項目評価精度計算を開始します")
        print(f"📊 Kokorochatサンプル数: {args.max_samples}")
        print(f"🎲 ランダムシード: {args.seed}")
        print(f"📋 評価項目数: {len(EVALUATION_ITEMS)}")
        if args.max_test_samples:
            print(f"📈 テストサンプル上限: {args.max_test_samples}")
        
        evaluator = MultiItemModelAccuracyEvaluator(api_key)
        evaluator.evaluate_all_models(
            max_test_samples=args.max_test_samples,
            use_kokorochat=not args.use_batch_results,
            max_samples=args.max_samples,
            seed=args.seed
        )
        
        print(f"\n✅ 20項目評価が完了しました！")
        print("📁 結果ファイルは以下に保存されました:")
        print(f"   - 詳細結果: multi_item_detailed_results_*.json")
        print(f"   - 精度指標: multi_item_metrics_*.csv")

    except Exception as e:
        logger.error(f"処理中にエラーが発生しました: {e}", exc_info=True)


if __name__ == "__main__":
    main()
