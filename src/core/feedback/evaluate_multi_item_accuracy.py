#!/usr/bin/env python3
"""
20項目の会話印象評価に対する予測精度計算スクリプト
各項目ごとにMAE、RMSE、誤差1での正解率を計算する機能付き
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
from sklearn.metrics import mean_absolute_error, mean_squared_error
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


class MultiItemModelEvaluator:
    """20項目の評価予測に対応したモデル評価クラス"""
    
    def __init__(self, api_key: str):
        """
        初期化
        
        Args:
            api_key: OpenAI APIキー
        """
        self.client = OpenAI(api_key=api_key)
        self.script_dir = Path(__file__).resolve().parent
        self.output_dir = self.script_dir / "openai_sft_outputs"
        logger.info(f"結果ディレクトリを設定: {self.output_dir}")
        self.output_dir.mkdir(exist_ok=True)

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

        # モデルIDの読み込み
        model_ids = []
        if 'batches' in results_data:
            for batch in results_data['batches']:
                if 'fine_tuned_model' in batch:
                    model_ids.append(batch['fine_tuned_model'])
        elif 'batch_results' in results_data:
            for batch in results_data['batch_results']:
                if batch.get('final_status') == 'succeeded' and 'final_model_id' in batch:
                    model_ids.append(batch['final_model_id'])
        
        if not model_ids:
            raise ValueError(f"結果ファイル {latest_results_file.name} からモデルIDを取得できませんでした。")
        
        logger.info(f"評価対象モデル数: {len(model_ids)}")
        for i, model_id in enumerate(model_ids):
            logger.info(f"  モデル {i+1}: {model_id}")

        # テストデータファイルのパスを取得
        test_data_path = None
        if 'test_data_file' in results_data:
            test_data_filename = results_data['test_data_file']
            test_data_path = self.output_dir / test_data_filename
        
        if not test_data_path or not test_data_path.exists():
            test_files = list(self.output_dir.glob("test_data_*.jsonl"))
            if test_files:
                latest_test_file = max(test_files, key=lambda x: x.stat().st_mtime)
                test_data_path = latest_test_file
        
        if not test_data_path or not test_data_path.exists():
            raise FileNotFoundError(f"テストデータファイルが見つかりません。ディレクトリ: {self.output_dir}")
        
        logger.info(f"使用するtestデータファイル: {test_data_path}")

        # テストデータの読み込み
        test_data = []
        with open(test_data_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        test_data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.warning(f"行 {line_num} のJSON解析に失敗: {e}")
                        continue
        
        logger.info(f"testデータ読み込み完了: {len(test_data)}サンプル")
        return model_ids, test_data

    def extract_score_from_response(self, response_text: str) -> Tuple[float, str, bool, Dict[int, float]]:
        """応答テキストから確率分布と期待値を抽出"""
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

    def parse_correct_scores_from_sample(self, sample: Dict[str, Any]) -> Dict[str, float]:
        """
        サンプルから正解スコアを抽出（KokoChatデータ構造に対応）
        
        Args:
            sample: KokoChatデータのサンプル
            
        Returns:
            各評価項目の正解スコア辞書
        """
        correct_scores = {}
        
        try:
            # KokoChatデータ構造の場合: review_by_client_jpから正解データを取得
            if 'review_by_client_jp' in sample:
                review_data = sample['review_by_client_jp']
                for item in EVALUATION_ITEMS:
                    if item in review_data:
                        score_value = review_data[item]
                        if isinstance(score_value, (int, float)) and score_value != "":
                            correct_scores[item] = float(score_value)
                        else:
                            logger.warning(f"項目 '{item}' のスコアが無効: {score_value}")
                
                if correct_scores:
                    return correct_scores
            
            # ファインチューニング用のmessages形式の場合
            elif 'messages' in sample:
                assistant_message = sample["messages"][-1]["content"]
                
                # JSON形式で各項目のスコアが含まれている場合
                try:
                    score_data = json.loads(assistant_message)
                    if isinstance(score_data, dict):
                        for item in EVALUATION_ITEMS:
                            if item in score_data:
                                correct_scores[item] = float(score_data[item])
                        if correct_scores:
                            return correct_scores
                except json.JSONDecodeError:
                    pass
                
                # 各項目が個別の行で記載されている場合
                lines = assistant_message.split('\n')
                for line in lines:
                    for item in EVALUATION_ITEMS:
                        if item in line:
                            # スコアを抽出（例: "項目名: 3.5点" の形式）
                            score_match = re.search(r'(\d+(?:\.\d+)?)', line)
                            if score_match:
                                correct_scores[item] = float(score_match.group(1))
                                break
            
            # どの形式でも解析できなかった場合のエラー
            if not correct_scores:
                logger.error(f"正解データを抽出できませんでした。サンプル構造: {list(sample.keys())}")
                return {}
                    
        except (AttributeError, IndexError, ValueError) as e:
            logger.warning(f"正解データの解析に失敗: {e}")
            
        return correct_scores

    def evaluate_model_on_all_items(self, model_id: str, test_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """1つのモデルを20項目すべてで評価"""
        logger.info(f"モデル {model_id} の20項目評価を開始...")
        
        all_predictions = []
        
        for sample_idx, sample in enumerate(test_data):
            try:
                # 会話テキストを取得（KokoChatデータ構造に対応）
                conversation_text = None
                
                if 'dialogue' in sample:
                    # KokoChatの対話データから会話テキストを構築
                    dialogue_parts = []
                    for turn in sample['dialogue']:
                        role = turn.get('role', 'unknown')
                        utterance = turn.get('utterance', '')
                        if role == 'counselor':
                            dialogue_parts.append(f"カウンセラー: {utterance}")
                        elif role == 'client':
                            dialogue_parts.append(f"相談者: {utterance}")
                    conversation_text = '\n'.join(dialogue_parts)
                elif 'messages' in sample:
                    # messages形式の場合
                    for msg in sample['messages']:
                        if msg['role'] == 'user':
                            conversation_text = msg['content']
                            break
                
                if not conversation_text:
                    logger.warning(f"サンプル {sample_idx+1}: 会話テキストが見つかりません")
                    continue
                
                # 正解データを取得
                correct_scores = self.parse_correct_scores_from_sample(sample)
                
                if not correct_scores:
                    logger.warning(f"サンプル {sample_idx+1}: 正解データが取得できませんでした")
                    continue
                
                sample_predictions = {
                    "sample_index": sample_idx,
                    "conversation_text": conversation_text,
                    "predictions": {},
                    "correct_scores": correct_scores,
                    "errors": {}
                }
                
                # 各評価項目について予測を実行
                for item_idx, evaluation_item in enumerate(EVALUATION_ITEMS):
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
                logger.debug(f"サンプル構造: {list(sample.keys()) if isinstance(sample, dict) else type(sample)}")
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

    def load_kokorochat_data_directly(self, max_test_samples: int = None) -> List[Dict[str, Any]]:
        """KokoChatデータを直接読み込み"""
        from datasets import load_dataset
        
        logger.info("KokoChatデータセットを直接使用します")
        logger.info("KokoroChat データセットを読み込み中...")
        
        # KokoChatデータセットを読み込み
        dataset = load_dataset("UEC-InabaLab/KokoroChat", split="train")
        
        if max_test_samples:
            dataset = dataset.select(range(min(max_test_samples, len(dataset))))
            logger.info(f"デバッグ用に{max_test_samples}サンプルに制限")
        
        logger.info(f"元データセットサイズ: {len(dataset)}")
        
        # 有効なサンプルのみを抽出（review_by_client_jpが存在するもの）
        valid_samples = []
        for sample in dataset:
            if 'review_by_client_jp' in sample and sample['review_by_client_jp']:
                valid_samples.append(sample)
        
        logger.info(f"有効なサンプル数: {len(valid_samples)}")
        
        # テストデータを抽出（全体の10%をテストデータとして使用）
        import random
        random.seed(42)  # 再現性のため
        
        test_size = int(len(valid_samples) * 0.1)
        test_data = random.sample(valid_samples, min(test_size, 150))  # 最大150サンプル
        
        logger.info(f"=== テストデータ抽出結果 ===")
        logger.info(f"有効データ: {len(valid_samples)} サンプル")
        logger.info(f"テストデータ: {len(test_data)} サンプル ({len(test_data)/len(valid_samples)*100:.1f}%)")
        logger.info(f"ランダムシード: 42")
        
        return test_data

    def _find_latest_results_file(self) -> Path:
        """最新のバッチ結果ファイルを見つける"""
        result_files = list(self.output_dir.glob("batch_fine_tuning_results_*.json"))
        if not result_files:
            logger.warning(f"バッチ結果ファイルがディレクトリに見つかりません: {self.output_dir}")
            return None
        
        latest_file = max(result_files, key=lambda x: x.stat().st_mtime)
        logger.info(f"使用する結果ファイル: {latest_file}")
        return latest_file

    def load_fine_tuned_model_ids(self) -> List[str]:
        """ファインチューニング済みモデルIDを読み込む（OpenAI APIから直接取得）"""
        try:
            from openai import OpenAI
            import os
            from dotenv import load_dotenv
            
            # .envファイル読み込み
            load_dotenv()
            api_key = os.getenv('OPENAI_API_KEY')
            
            if not api_key:
                logger.warning("OpenAI APIキーが見つかりません。ベースモデルを使用します。")
                return ["gpt-4o-mini"]
            
            client = OpenAI(api_key=api_key)
            
            # ファインチューニング済みモデル一覧を取得
            logger.info("OpenAI APIからファインチューニング済みモデルを取得中...")
            jobs = client.fine_tuning.jobs.list(limit=20)
            
            model_ids = []
            for job in jobs.data:
                if job.status == 'succeeded' and hasattr(job, 'fine_tuned_model') and job.fine_tuned_model:
                    # gpt-4o-miniベースのモデルを優先
                    if 'gpt-4o-mini' in job.fine_tuned_model:
                        model_ids.append(job.fine_tuned_model)
            
            # gpt-4o-miniベースがない場合、他の成功したモデルを追加
            if not model_ids:
                for job in jobs.data:
                    if job.status == 'succeeded' and hasattr(job, 'fine_tuned_model') and job.fine_tuned_model:
                        model_ids.append(job.fine_tuned_model)
            
            if not model_ids:
                logger.warning("有効なファインチューニング済みモデルが見つかりません。ベースモデルを使用します。")
                return ["gpt-4o-mini"]
            
            # 最新のモデルのみを使用（最初の1つ）
            selected_model = model_ids[0]
            
            logger.info(f"ファインチューニング済みモデルを使用: {selected_model}")
            return [selected_model]
            
        except Exception as e:
            logger.error(f"ファインチューニングモデル取得エラー: {e}")
            logger.info("ベースモデルを使用します。")
            return ["gpt-4o-mini"]

    def evaluate_all_models_multi_item(self, max_test_samples: int = None):
        """全モデルを20項目で評価（ファインチューニング済みモデルまたはKokoChatデータを直接使用）"""
        try:
            # KokoChatデータを直接読み込み
            test_data = self.load_kokorochat_data_directly(max_test_samples)
            
            # ファインチューニング済みモデルIDを読み込み
            model_ids = self.load_fine_tuned_model_ids()
            
        except Exception as e:
            logger.error(f"データ読み込みエラー: {e}")
            return
        
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
    parser = argparse.ArgumentParser(description="20項目評価精度計算スクリプト")
    parser.add_argument("--max-samples", type=int, help="評価サンプル数の上限")
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
        print(f"📋 評価項目数: {len(EVALUATION_ITEMS)}")
        if args.max_samples:
            print(f"📊 最大サンプル数: {args.max_samples}")
        
        evaluator = MultiItemModelEvaluator(api_key)
        evaluator.evaluate_all_models_multi_item(max_test_samples=args.max_samples)
        
        print(f"\n✅ 20項目評価が完了しました！")
        print("📁 結果ファイルは以下に保存されました:")
        print(f"   - 詳細結果: multi_item_detailed_results_*.json")
        print(f"   - 精度指標: multi_item_metrics_*.csv")

    except Exception as e:
        logger.error(f"処理中にエラーが発生しました: {e}", exc_info=True)


if __name__ == "__main__":
    main()
