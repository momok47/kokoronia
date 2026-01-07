#!/usr/bin/env python3
"""
0〜6段階評価の分布を棒グラフで可視化するスクリプト
正解、チューニング済みモデル、チューニングなしモデルの分布を比較
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Tuple
import logging
from datetime import datetime
import re
from collections import defaultdict

# 日本語フォント設定
plt.rcParams['font.family'] = ['DejaVu Sans', 'Hiragino Sans', 'Yu Gothic', 'Meiryo', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

class ScoreDistributionVisualizer:
    """スコア分布可視化クラス"""
    
    def __init__(self, output_dir: Path = None):
        """
        初期化
        
        Args:
            output_dir: 結果ディレクトリ（デフォルトはスクリプトと同じ場所のopenai_sft_outputs）
        """
        if output_dir is None:
            self.output_dir = Path(__file__).resolve().parent / "openai_sft_outputs"
        else:
            self.output_dir = output_dir
        
        logger.info(f"結果ディレクトリ: {self.output_dir}")
        
        # 出力ディレクトリの確認
        if not self.output_dir.exists():
            raise FileNotFoundError(f"結果ディレクトリが見つかりません: {self.output_dir}")

    def load_ground_truth_distribution(self, model_id: str = None) -> Dict[int, int]:
        """正解データの分布を取得"""
        logger.info("正解データの分布を取得中...")
        
        # 詳細結果ファイルから正解データを取得
        result_files = list(self.output_dir.glob("multi_item_detailed_results_*.json"))
        if not result_files:
            raise FileNotFoundError("詳細結果ファイルが見つかりません")
        
        ground_truth_scores = []
        
        # チューニングなしモデルのファイルから正解データを取得（より多くのサンプルがあるため）
        target_file = None
        for result_file in sorted(result_files, key=lambda x: x.stat().st_mtime):
            try:
                with open(result_file, 'r', encoding='utf-8') as f:
                    results_data = json.load(f)
                
                # gpt-4o-mini（チューニングなし）のファイルを探す
                for result in results_data:
                    if result.get('model_id') == 'gpt-4o-mini':
                        target_file = result_file
                        logger.info(f"使用する結果ファイル: {target_file.name}")
                        
                        # 正解スコアを取得
                        for sample_pred in result['predictions']:
                            for item, correct_score in sample_pred['correct_scores'].items():
                                # 0-5の範囲に丸める
                                score = max(0, min(5, round(correct_score)))
                                ground_truth_scores.append(score)
                        break
                
                if target_file:
                    break
                    
            except Exception as e:
                logger.warning(f"ファイル {result_file.name} の読み込みに失敗: {e}")
                continue
        
        if not target_file:
            # フォールバック：最新ファイルから取得
            latest_result_file = max(result_files, key=lambda x: x.stat().st_mtime)
            logger.info(f"フォールバック: {latest_result_file.name}")
            
            with open(latest_result_file, 'r', encoding='utf-8') as f:
                results_data = json.load(f)
            
            if results_data:
                first_model_data = results_data[0]
                for sample_pred in first_model_data['predictions']:
                    for item, correct_score in sample_pred['correct_scores'].items():
                        # 0-5の範囲に丸める
                        score = max(0, min(5, round(correct_score)))
                        ground_truth_scores.append(score)
        
        if not ground_truth_scores:
            raise ValueError("正解データが見つかりません")
        
        # 分布を計算
        distribution = {}
        for score in range(6):  # 0-5
            distribution[score] = ground_truth_scores.count(score)
        
        total_samples = len(ground_truth_scores)
        logger.info(f"正解データ読み込み完了: {total_samples}サンプル")
        logger.info(f"正解分布: {distribution}")
        
        return distribution

    def load_model_predictions_distribution(self, model_id: str) -> Dict[int, int]:
        """指定されたモデルの予測分布を取得"""
        logger.info(f"モデル {model_id} の予測分布を取得中...")
        
        # 詳細結果ファイルを探す
        result_files = list(self.output_dir.glob("multi_item_detailed_results_*.json"))
        if not result_files:
            raise FileNotFoundError("詳細結果ファイルが見つかりません")
        
        # 全てのファイルをチェックしてモデルIDを探す
        model_data = None
        used_file = None
        
        for result_file in sorted(result_files, key=lambda x: x.stat().st_mtime, reverse=True):
            try:
                with open(result_file, 'r', encoding='utf-8') as f:
                    results_data = json.load(f)
                
                # モデルIDに対応するデータを探す
                for result in results_data:
                    if result.get('model_id') == model_id:
                        model_data = result
                        used_file = result_file
                        break
                
                if model_data:
                    break
                    
            except Exception as e:
                logger.warning(f"ファイル {result_file.name} の読み込みに失敗: {e}")
                continue
        
        if not model_data:
            raise ValueError(f"モデル {model_id} のデータが見つかりません")
        
        logger.info(f"使用する結果ファイル: {used_file.name}")
        
        try:
            # 予測スコアを収集
            predicted_scores = []
            for sample_pred in model_data['predictions']:
                for item, prediction in sample_pred['predictions'].items():
                    # 0-5の範囲に丸める
                    score = max(0, min(5, round(prediction)))
                    predicted_scores.append(score)
            
            # 分布を計算
            distribution = {}
            for score in range(6):  # 0-5
                distribution[score] = predicted_scores.count(score)
            
            total_predictions = len(predicted_scores)
            logger.info(f"モデル {model_id} の予測読み込み完了: {total_predictions}予測")
            logger.info(f"予測分布: {distribution}")
            
            return distribution
            
        except Exception as e:
            logger.error(f"モデル {model_id} の予測読み込み中にエラー: {e}")
            raise

    def create_distribution_comparison_plot(self, 
                                         ground_truth_dist: Dict[int, int],
                                         tuned_model_dist: Dict[int, int],
                                         untuned_model_dist: Dict[int, int],
                                         output_path: Path = None):
        """分布比較の棒グラフを作成"""
        logger.info("分布比較グラフを作成中...")
        
        # データを準備
        scores = list(range(6))  # 0-5
        
        # 正規化（パーセンテージ）
        total_gt = sum(ground_truth_dist.values())
        total_tuned = sum(tuned_model_dist.values())
        total_untuned = sum(untuned_model_dist.values())
        
        gt_percentages = [ground_truth_dist[score] / total_gt * 100 for score in scores]
        tuned_percentages = [tuned_model_dist[score] / total_tuned * 100 for score in scores]
        untuned_percentages = [untuned_model_dist[score] / total_untuned * 100 for score in scores]
        
        # グラフの設定
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # バーの幅と位置
        bar_width = 0.25
        x_pos = np.arange(len(scores))
        
        # 棒グラフを描画
        bars1 = ax.bar(x_pos - bar_width, gt_percentages, bar_width, 
                      label='正解', color='#61C5FF', alpha=0.8)
        bars2 = ax.bar(x_pos, tuned_percentages, bar_width,
                      label='チューニング済みモデル', color='#FFB550', alpha=0.8)
        bars3 = ax.bar(x_pos + bar_width, untuned_percentages, bar_width,
                      label='チューニングなしモデル', color='#FFF5AE', alpha=0.8)
        
        # グラフの装飾
        ax.set_xlabel('評価スコア', fontsize=24, fontweight='bold')
        ax.set_ylabel('分布 (%)', fontsize=24, fontweight='bold')
        ax.set_title('0〜5段階評価の分布比較', fontsize=28, fontweight='bold', pad=20)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f'{i}点' for i in scores], fontsize=20)
        ax.legend(fontsize=20)
        ax.grid(True, alpha=0.3)
        
        
        # レイアウト調整
        plt.tight_layout()
        
        # 保存
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.output_dir / f"score_distribution_comparison_{timestamp}.png"
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"分布比較グラフを保存: {output_path}")
        
        # 表示
        plt.show()
        
        return output_path



    def print_distribution_summary(self, 
                                 ground_truth_dist: Dict[int, int],
                                 tuned_model_dist: Dict[int, int],
                                 untuned_model_dist: Dict[int, int]):
        """分布のサマリーを表示"""
        print("\n" + "="*60)
        print("📊 スコア分布サマリー 📊")
        print("="*60)
        
        print(f"\n🎯 正解データ:")
        total_gt = sum(ground_truth_dist.values())
        for score in range(6):
            count = ground_truth_dist[score]
            percentage = count / total_gt * 100
            print(f"  {score}点: {count:3d}件 ({percentage:5.1f}%)")
        
        print(f"\n🤖 チューニング済みモデル:")
        total_tuned = sum(tuned_model_dist.values())
        for score in range(6):
            count = tuned_model_dist[score]
            percentage = count / total_tuned * 100
            print(f"  {score}点: {count:3d}件 ({percentage:5.1f}%)")
        
        print(f"\n🔧 チューニングなしモデル:")
        total_untuned = sum(untuned_model_dist.values())
        for score in range(6):
            count = untuned_model_dist[score]
            percentage = count / total_untuned * 100
            print(f"  {score}点: {count:3d}件 ({percentage:5.1f}%)")
        
        print("\n" + "="*60)




    def visualize_all_distributions(self, tuned_model_id: str, untuned_model_id: str = None):
        """全ての分布可視化を実行"""
        try:
            # 分布データを取得
            ground_truth_dist = self.load_ground_truth_distribution()
            tuned_model_dist = self.load_model_predictions_distribution(tuned_model_id)
            
            # チューニングなしモデルのデータがあるかチェック
            untuned_model_dist = None
            if untuned_model_id:
                try:
                    untuned_model_dist = self.load_model_predictions_distribution(untuned_model_id)
                except ValueError as e:
                    logger.warning(f"チューニングなしモデルのデータが見つかりません: {e}")
                    untuned_model_dist = None
            
            # サマリー表示
            self.print_distribution_summary(ground_truth_dist, tuned_model_dist, untuned_model_dist)
            
            # 比較グラフ作成
            comparison_path = self.create_distribution_comparison_plot(
                ground_truth_dist, tuned_model_dist, untuned_model_dist
            )
            
            print(f"\n✅ 分布可視化が完了しました！")
            print(f"📁 結果は {self.output_dir} に保存されました")
            
        except Exception as e:
            logger.error(f"分布可視化中にエラー: {e}")
            raise

def main():
    """メイン関数"""
    try:
        # 結果ディレクトリを設定
        script_dir = Path(__file__).resolve().parent
        output_dir = script_dir / "openai_sft_outputs"
        
        if not output_dir.exists():
            raise FileNotFoundError(f"結果ディレクトリが見つかりません: {output_dir}")
        
        # 可視化実行
        visualizer = ScoreDistributionVisualizer(output_dir)
        
        # モデルIDを指定（実際のモデルIDに変更してください）
        tuned_model_id = "ft:gpt-4o-mini-2024-07-18:personal::CAZPNbKA"  # 実際のチューニング済みモデルID
        untuned_model_id = "gpt-4o-mini"  # チューニングなしモデルのID
        
        visualizer.visualize_all_distributions(tuned_model_id, untuned_model_id)
        
    except Exception as e:
        logger.error(f"処理中にエラーが発生しました: {e}", exc_info=True)

if __name__ == "__main__":
    main()
