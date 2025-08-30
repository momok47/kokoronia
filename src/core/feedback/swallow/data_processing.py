# -*- coding: utf-8 -*-
# data_processing.py

import os
import json
import torch
import numpy as np
try:
    from typing import Dict, List, Optional, Tuple
except ImportError:
    # 古いPythonバージョン用のフォールバック
    Dict = dict
    List = list
    Optional = object
    Tuple = tuple
from datasets import load_dataset, Dataset
import logging
import random

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 20項目の評価指標 (コメントを修正)
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

def load_and_split_dataset(seed=42):
    """
    データセットを読み込み、8:1:1に分割する。
    再現性のために乱数シードを固定。
    """
    # 再現性のためにシードを設定
    random.seed(seed)
    
    logger.info("UEC-InabaLab/KokoroChat データセットを読み込んでいます...")
    ds = load_dataset("UEC-InabaLab/KokoroChat")
    df_all = ds['train']

    logger.info("=== データセット情報 ===")
    logger.info("データセットサイズ: {}".format(len(df_all)))
    logger.info("カラム: {}".format(df_all.column_names))

    # train:test:valid = 8:1:1 に分割
    logger.info("\n=== データ分割 (8:1:1) ===")
    shuffled_ds = df_all.shuffle(seed=seed)
    
    train_test_split = shuffled_ds.train_test_split(test_size=0.2)
    train_data = train_test_split['train']
    
    test_valid_split = train_test_split['test'].train_test_split(test_size=0.5)
    test_data = test_valid_split['train']
    valid_data = test_valid_split['test']

    logger.info("トレーニングデータサイズ: {}".format(len(train_data)))
    logger.info("テストデータサイズ: {}".format(len(test_data)))
    logger.info("検証データサイズ: {}".format(len(valid_data)))
    
    return train_data, test_data, valid_data

def probability_to_expected_score(probabilities, score_range=(0, 5)):
    """
    確率分布から期待値を計算する。
    例: [0.1, 0.2, 0.7] -> 0*0.1 + 1*0.2 + 2*0.7 = 1.6
    """
    if not probabilities:
        return 0.0
    scores = np.arange(score_range[0], score_range[1] + 1)
    if len(scores) != len(probabilities):
        logger.warning("スコア範囲({})と確率分布の長さ({})が一致しません。".format(len(scores), len(probabilities)))
        return 0.0
        
    expected_value = np.dot(scores, probabilities)
    return float(expected_value)


def save_data(data, file_path):
    """
    処理済みデータをJSON Lines形式で保存する。
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        logger.info("データを {} に保存しました。".format(file_path))
    except IOError as e:
        logger.error("ファイル保存中にエラーが発生しました: {}".format(e))

if __name__ == '__main__':
    # このスクリプトを直接実行した際のテスト用動作
    train_ds, test_ds, valid_ds = load_and_split_dataset()
    print("\n--- 最初の学習データサンプル ---")
    print(train_ds[0])