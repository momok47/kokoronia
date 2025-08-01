import os
import json
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from datasets import load_dataset, Dataset
import logging
import random

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 17項目の評価指標
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

def load_and_split_dataset():
    """データセットを読み込み、8:1:1に分割"""
    # データセットの読み込み
    ds = load_dataset("UEC-InabaLab/KokoroChat")
    df_all = ds['train']

    print("=== データセット情報 ===")
    print(f"データセットサイズ: {len(df_all)}")
    print(f"カラム: {df_all.column_names}")

    # train:test:valid = 8:1:1 に分割
    print("\n=== データ分割 ===")
    total_size = len(df_all)
    indices = list(range(total_size))
    random.shuffle(indices)

    train_end = int(total_size * 0.8)
    test_end = int(total_size * 0.9)

    train_indices = indices[:train_end]
    test_indices = indices[train_end:test_end]
    valid_indices = indices[test_end:]

    train_data = df_all.select(train_indices)
    test_data = df_all.select(test_indices)
    valid_data = df_all.select(valid_indices)

    print(f"トレーニングデータサイズ: {len(train_data)} ({len(train_data)/total_size*100:.1f}%)")
    print(f"テストデータサイズ: {len(test_data)} ({len(test_data)/total_size*100:.1f}%)")
    print(f"検証データサイズ: {len(valid_data)} ({len(valid_data)/total_size*100:.1f}%)")

    return train_data, test_data, valid_data

def calculate_weighted_average(turn_scores: list) -> float:
    """ターンスコアの加重平均を計算"""
    if not turn_scores:
        raise ValueError("ターンスコアリストが空です。データが正しく読み込まれているか確認してください。")
    
    # 後半のターンにより重みを付ける（会話の進行に応じて）
    weights = []
    for i in range(len(turn_scores)):
        weight = 1.0 + (i * 0.2)
        weights.append(weight)
    
    # 正規化
    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights]
    
    # 加重平均を計算
    weighted_average = sum(score * weight for score, weight in zip(turn_scores, normalized_weights))
    return weighted_average

def calculate_weighted_average_probabilities(turn_probabilities: List[List[float]]) -> List[float]:
    """
    確率分布の加重平均を計算
    
    Args:
        turn_probabilities: 各ターンの確率分布のリスト
                          [[p0, p1, p2, p3, p4, p5], ...]
    
    Returns:
        加重平均された確率分布 [p0, p1, p2, p3, p4, p5]
    """
    if not turn_probabilities:
        # 空の場合は均等な確率分布を返すなど、適切なデフォルト値を設定
        return [1/6] * 6

    # 後半のターンにより重みを付ける（会話の進行に応じて）
    weights = [1.0 + (i * 0.2) for i in range(len(turn_probabilities))]
    
    # 正規化
    total_weight = sum(weights)
    if total_weight == 0:
        # 重みの合計が0になることは通常ないが、念のため均等な重みに
        normalized_weights = [1/len(weights)] * len(weights)
    else:
        normalized_weights = [w / total_weight for w in weights]
    
    # 各スコア（0-5）について加重平均を計算
    weighted_probabilities = [0.0] * 6
    
    for score_idx in range(6):
        for turn_idx, turn_probs in enumerate(turn_probabilities):
            if len(turn_probs) > score_idx:
                weighted_probabilities[score_idx] += turn_probs[score_idx] * normalized_weights[turn_idx]
    
    # 確率の合計を1.0に正規化
    total_prob = sum(weighted_probabilities)
    if total_prob > 0:
        weighted_probabilities = [p / total_prob for p in weighted_probabilities]
    else:
        # 🚨【重要】合計が0の場合、均等な確率分布を返す
        logger.warning("加重平均後の確率合計が0でした。均等分布を返します。")
        weighted_probabilities = [1/6] * 6
    
    return weighted_probabilities

def probability_to_expected_score(probabilities: List[float]) -> float:
    """
    確率分布から期待値を計算
    Args:
        probabilities: [p0, p1, p2, p3, p4, p5]
    
    Returns:
        期待値スコア (0.0-5.0)
    """
    expected_score = sum(i * p for i, p in enumerate(probabilities))
    return expected_score 