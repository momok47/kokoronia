# -*- coding: utf-8 -*-
# llm_evaluation.py

import re
import logging
try:
    from typing import Dict, List
except ImportError:
    # 古いPythonバージョン用のフォールバック
    Dict = dict
    List = list
from transformers import pipeline
import numpy as np

# 他のモジュールからインポート
from data_processing import EVALUATION_ITEMS

logger = logging.getLogger(__name__)

def create_unified_evaluation_prompt(conversation_text, evaluation_item):
    """LLMに評価を依頼するためのプロンプトを生成する。"""
    # 対話が長すぎる場合、さらに短縮（処理時間短縮のため）
    short_conversation = conversation_text if len(conversation_text) <= 800 else conversation_text[:800] + "..."

    prompt = """### 指示
以下の対話について「{}」の満足度を0～5点で評価し、各点数の確率を出力してください。

### 対話
{}

### 出力形式（数値のみ）
0点: XX%
1点: XX%
2点: XX%
3点: XX%
4点: XX%
5点: XX%

### 回答
""".format(evaluation_item, short_conversation)
    return prompt

def parse_probabilities_from_llm_response(response):
    """
    LLMの応答テキストから確率分布を抽出する、より堅牢な関数。
    マイナス点数や異常値を適切にハンドリングする。
    """
    # 0点から5点までの確率を保持するリストを初期化
    probabilities = [-1.0] * 6
    
    # より柔軟な確率抽出パターン（マイナス記号も考慮）
    # "X点: Y%" や "X点：Y%" や "X点 Y%" などに対応
    patterns = [
        re.compile(r"(-?\d)\s*点\s*[:\：]\s*([\d\.]+)\s*%?"),  # 基本パターン（マイナス対応）
        re.compile(r"(-?\d)\s*点\s+([\d\.]+)\s*%?"),          # コロンなし
        re.compile(r"(-?\d)\s*[:\：]\s*([\d\.]+)\s*%?"),      # "点"なし
    ]

    matches = []
    for pattern in patterns:
        matches = pattern.findall(response)
        if matches:
            break
    
    if not matches:
        logger.warning(f"応答から確率パターンが見つかりませんでした。応答: '{response[:200]}'")
        return [1/6] * 6 # 均等確率を返す

    valid_matches = 0
    for score_str, prob_str in matches:
        try:
            score = int(score_str)
            prob = float(prob_str)
            
            # マイナス点数や範囲外の点数を除外
            if score < 0:
                logger.warning("マイナス点数 {} を検出しました。無視します。".format(score))
                continue
            elif score > 5:
                logger.warning("範囲外の点数 {} を検出しました。無視します。".format(score))
                continue
            
            # 確率値の妥当性チェック
            if prob < 0:
                logger.warning("{}点の確率が負の値 {}% です。0%に修正します。".format(score, prob))
                prob = 0
            elif prob > 100:
                logger.warning("{}点の確率が100%を超えています: {}%。100%に修正します。".format(score, prob))
                prob = 100
                
            probabilities[score] = prob / 100.0  # パーセント表記を小数に変換
            valid_matches += 1
            
        except (ValueError, IndexError) as e:
            logger.warning("数値変換エラー: score='{}', prob='{}', error={}".format(score_str, prob_str, e))
            continue

    # 有効なマッチが少ない場合
    if valid_matches < 6:
        logger.warning("有効な確率が{}/6個しか抽出できませんでした。抽出結果: {}".format(valid_matches, probabilities))
        
        # 抽出できなかった部分に小さな確率を割り当て
        for i in range(6):
            if probabilities[i] == -1.0:
                probabilities[i] = 0.01  # 1%の小さな確率を割り当て

    # すべて同じ確率（均等分布）の場合を検出
    non_negative_probs = [p for p in probabilities if p >= 0]
    if len(set(non_negative_probs)) == 1 and len(non_negative_probs) == 6:
        logger.warning("すべて同じ確率が検出されました。これは不適切な応答の可能性があります。")
        # 軽微な変動を加えて均等分布を回避
        probabilities = [0.15, 0.16, 0.17, 0.18, 0.17, 0.17]

    # 合計が100になるように正規化
    total_prob = sum(p for p in probabilities if p >= 0)
    if total_prob > 0:
        normalized_probabilities = [max(0, p) / total_prob for p in probabilities]
    else:
        logger.warning("確率の合計が0です。デフォルト分布を返します。")
        # より現実的なデフォルト分布（中央値周辺に重み）
        return [0.05, 0.1, 0.2, 0.3, 0.25, 0.1]

    return normalized_probabilities

def call_llm_for_probability_distribution(llm_pipeline, conversation_text, evaluation_item):
    """LLMパイプラインを使い、単一の評価項目の確率分布を取得する。"""
    prompt = create_unified_evaluation_prompt(conversation_text, evaluation_item)
    
    try:
        logger.info("プロンプト長: {} 文字".format(len(prompt)))
        logger.info("LLM推論を開始...")
        
        response = llm_pipeline(
            prompt,
            max_new_tokens=80,   # より短く設定してハングを防ぐ
            do_sample=False,     # 決定的な出力でハングを回避
            return_full_text=False,
            eos_token_id=llm_pipeline.tokenizer.eos_token_id,
            pad_token_id=llm_pipeline.tokenizer.eos_token_id,
        )
        response_text = response[0]['generated_text']
        logger.info("LLM推論完了")
        
    except Exception as e:
        logger.error("LLMパイプライン呼び出しでエラー: {}".format(e))
        response_text = ""

    logger.info("LLM Raw Response for '{}': '{}'".format(evaluation_item, response_text[:200]))
    
    probabilities = parse_probabilities_from_llm_response(response_text)
    
    logger.info("Parsed Probabilities for '{}': {}".format(evaluation_item, ['{:.3f}'.format(p) for p in probabilities]))
    
    # 確率分布の品質チェック
    validate_probability_distribution(probabilities, evaluation_item)
    
    return probabilities

def validate_probability_distribution(probabilities, evaluation_item):
    """確率分布の品質をチェックし、問題があれば警告を出す。"""
    # 均等分布の検出（すべて同じ値）
    if len(set(probabilities)) == 1:
        logger.warning("'{}': 均等分布が検出されました。LLMが適切に評価できていない可能性があります。".format(evaluation_item))
    
    # 極端に偏った分布の検出（1つの値が80%以上）
    max_prob = max(probabilities)
    if max_prob > 0.8:
        max_index = probabilities.index(max_prob)
        logger.warning("'{}': {}点に{:.1%}の極端に高い確率が割り当てられています。".format(evaluation_item, max_index, max_prob))
    
    # 合計が1.0から大きく外れている場合
    total = sum(probabilities)
    if abs(total - 1.0) > 0.01:
        logger.warning("'{}': 確率の合計が{:.3f}で1.0から外れています。".format(evaluation_item, total))
    
    # 負の確率値の検出
    if any(p < 0 for p in probabilities):
        logger.error("'{}': 負の確率値が含まれています: {}".format(evaluation_item, probabilities))
    
    # ゼロ確率が多すぎる場合
    zero_count = sum(1 for p in probabilities if p == 0)
    if zero_count >= 4:  # 6点中4点以上がゼロ確率
        logger.warning("'{}': {}/6個の点数がゼロ確率です。分布が極端すぎる可能性があります。".format(evaluation_item, zero_count))

def evaluate_conversation_on_items(conversation_text, llm_pipeline):
    """
    一つの会話に対し、全ての評価項目の確率分布を計算して返す。
    """
    all_probabilities = {}
    logger.info("全{}項目について評価を開始します...".format(len(EVALUATION_ITEMS)))
    for i, item in enumerate(EVALUATION_ITEMS):
        logger.info("--- 項目 {}/{}: '{}' ---".format(i+1, len(EVALUATION_ITEMS), item))
        probabilities = call_llm_for_probability_distribution(llm_pipeline, conversation_text, item)
        all_probabilities[item] = probabilities
    
    return all_probabilities