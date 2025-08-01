import re
import logging
from typing import Dict, List
from transformers import pipeline

# 相対インポートを試行、失敗した場合は絶対インポート
try:
    from .data_processing import EVALUATION_ITEMS
except ImportError:
    from data_processing import EVALUATION_ITEMS

logger = logging.getLogger(__name__)

def create_unified_evaluation_prompt(turn_list: list, review: str, item: str = None) -> str:
    """
    Args:
        turn_list: ターンの要素リスト
        review: クライアントの評価
        item: 評価項目名（Noneの場合は全項目）
    
    Returns:
        統一されたプロンプト
    """
    # counselorとclientの発言を分けて整理
    counselor_text = ""
    client_text = ""
    
    for turn in turn_list:
        if turn['role'] == 'counselor':
            counselor_text += f"カウンセラー: {turn['utterance']}\n"
        elif turn['role'] == 'client':
            client_text += f"クライアント: {turn['utterance']}\n"
    
    # 統一されたプロンプト
    prompt = f"""以下のカウンセリング会話について、評価を行ってください。

                    会話内容:
                    カウンセラーの発言:
                    {counselor_text}

                    クライアントの発言:
                    {client_text}

                    クライアントの評価:
                    {review}

                    評価基準:
                    0=非常に悪い, 1=悪い, 2=普通, 3=良い, 4=非常に良い, 5=最高

                    【重要】以下の形式で必ず回答してください。他の説明は不要です。"""
    if item:
        # 特定項目の評価
        prompt += f"""
{item}の観点でのクライアントの評価確率分布を0.0-1.0の範囲で回答してください（合計1.0になるように）。

必ず以下の形式で回答してください：
0点の確率: [数値]
1点の確率: [数値]
2点の確率: [数値]
3点の確率: [数値]
4点の確率: [数値]
5点の確率: [数値]

例：
0点の確率: 0.05
1点の確率: 0.15
2点の確率: 0.25
3点の確率: 0.35
4点の確率: 0.15
5点の確率: 0.05"""
    else:
        # 全項目の評価
        prompt += """
各評価項目について確率分布を回答してください。

必ず以下の形式で回答してください：
[項目名]:
0点の確率: [数値]
1点の確率: [数値]
2点の確率: [数値]
3点の確率: [数値]
4点の確率: [数値]
5点の確率: [数値]

例：
共感:
0点の確率: 0.05
1点の確率: 0.15
2点の確率: 0.25
3点の確率: 0.35
4点の確率: 0.15
5点の確率: 0.05"""
    
    return prompt

def call_llm_for_probability_distribution(prompt: str, llm_pipeline) -> List[float]:
    """
    LLMを呼び出して0-5の確率分布を取得
    Args:
        prompt: LLMへのプロンプト
        llm_pipeline: LLMパイプライン
    Returns:
        0-5の各スコアの確率分布 [p0, p1, p2, p3, p4, p5]
    """
    if llm_pipeline is None:
        logger.warning("LLMパイプラインが利用できません。デフォルト確率分布を返します。")
        return [0.0, 0.0, 0.1, 0.8, 0.1, 0.0]
    
    try:
        # LLMを呼び出し
        response = llm_pipeline(prompt, max_new_tokens=100)[0]['generated_text']
        
        # プロンプト部分を除去
        response_only = response[len(prompt):].strip()
        
        # 確率を抽出
        probabilities = parse_probabilities_from_llm_response(response_only)
        
        logger.info(f"LLM確率分布応答: {response_only[:100]}... -> 確率: {probabilities}")
        return probabilities
        
    except Exception as e:
        logger.error(f"LLM確率分布呼び出しエラー: {e}")
        logger.warning("デフォルト確率分布を返します。")
        return [0.0, 0.0, 0.1, 0.8, 0.1, 0.0]

def parse_probabilities_from_llm_response(response: str) -> List[float]:
    """
    LLMの応答から確率分布を抽出
    Args:
        response: LLMの応答テキスト
    Returns:
        0-5の各スコアの確率分布 [p0, p1, p2, p3, p4, p5]
    """
    try:
        # 複数の形式に対応するパターン
        patterns = [
            # 標準形式（半角コロン）
            r'0点の確率:\s*([0-9]*\.?[0-9]+)',
            r'1点の確率:\s*([0-9]*\.?[0-9]+)',
            r'2点の確率:\s*([0-9]*\.?[0-9]+)',
            r'3点の確率:\s*([0-9]*\.?[0-9]+)',
            r'4点の確率:\s*([0-9]*\.?[0-9]+)',
            r'5点の確率:\s*([0-9]*\.?[0-9]+)'
        ]
        
        # フォールバック用のパターン（全角コロン対応）
        fallback_patterns = [
            r'0点の確率：\s*([0-9]*\.?[0-9]+)',
            r'1点の確率：\s*([0-9]*\.?[0-9]+)',
            r'2点の確率：\s*([0-9]*\.?[0-9]+)',
            r'3点の確率：\s*([0-9]*\.?[0-9]+)',
            r'4点の確率：\s*([0-9]*\.?[0-9]+)',
            r'5点の確率：\s*([0-9]*\.?[0-9]+)'
        ]
        
        probabilities = []
        for i, pattern in enumerate(patterns):
            match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
            if match:
                try:
                    prob = float(match.group(1))
                    probabilities.append(prob)
                except ValueError:
                    probabilities.append(0.0)
            else:
                # フォールバックパターンを試行
                fallback_match = re.search(fallback_patterns[i], response, re.IGNORECASE | re.MULTILINE)
                if fallback_match:
                    try:
                        prob = float(fallback_match.group(1))
                        probabilities.append(prob)
                    except ValueError:
                        probabilities.append(0.0)
                else:
                    probabilities.append(0.0)
        
        # 確率の合計を正規化
        total = sum(probabilities)
        if total > 0:
            probabilities = [p / total for p in probabilities]
        else:
            # デフォルト値
            probabilities = [0.0, 0.0, 0.1, 0.8, 0.1, 0.0]
        
        return probabilities
     
    except Exception as e:
        logger.warning(f"確率抽出エラー: {e}, レスポンス: {response}")
        raise ValueError(f"確率分布抽出中にエラーが発生しました: {e}")

def evaluate_turn_on_items(turn_list: list, review: str, llm_pipeline) -> Dict[str, List[float]]:
    """
    ターンの各評価項目について確率分布を計算
    Args:
        turn_list: ターンの要素リスト [{'role': 'counselor', 'utterance': '...'}, ...]
        review: クライアントの評価
        llm_pipeline: LLMパイプライン
    Returns:
        各評価項目の確率分布辞書
    """
    evaluation_probabilities = {}
    
    for item in EVALUATION_ITEMS:
        probabilities = calculate_item_probabilities(turn_list, item, review, llm_pipeline)
        evaluation_probabilities[item] = probabilities
    
    return evaluation_probabilities

def calculate_item_probabilities(turn_list: list, item: str, review: str, llm_pipeline) -> List[float]:
    """
    特定の評価項目についてクライアントの確率分布を計算（LLMベース）
    
    Args:
        turn_list: ターンの要素リスト
        item: 評価項目名
        review: クライアントの評価
        llm_pipeline: LLMパイプライン
    
    Returns:
        クライアントの確率分布 [p0, p1, p2, p3, p4, p5]
    """
    # 統一されたプロンプトを使用
    prompt = create_unified_evaluation_prompt(turn_list, review, item)
    
    # LLMを使用した確率分布取得
    probabilities = call_llm_for_probability_distribution(prompt, llm_pipeline)
    return probabilities

def create_emotion_prompt(dialogue: str, review: str, llm_pipeline) -> str:
    """感情評価用のプロンプトを作成"""
    try:
        from .turn_segmentation import segment_turns, create_turn_text, create_turn_list
    except ImportError:
        from turn_segmentation import segment_turns, create_turn_text, create_turn_list
    
    try:
        from .data_processing import calculate_weighted_average_probabilities, probability_to_expected_score
    except ImportError:
        from data_processing import calculate_weighted_average_probabilities, probability_to_expected_score
    
    if isinstance(dialogue, dict) and 'dialogue' in dialogue:
        # ターン分割を実行
        turns = dialogue['dialogue']
        counselor_turns, client_turns, max_turns = segment_turns(turns)
        
        # ターンごとの要素リストを作成
        turn_list = create_turn_list(counselor_turns, client_turns, max_turns)
        turn_evaluations = []
        
        # 各ターンの17項目評価を計算
        for i, current_turn in enumerate(turn_list):
            # 17項目の確率分布を計算
            evaluation_probabilities = evaluate_turn_on_items(current_turn, review, llm_pipeline)
            turn_evaluations.append(evaluation_probabilities)
            
            print(f"ターン {i+1} の評価:")
            print(f"  17項目確率分布:")
            for item, probabilities in evaluation_probabilities.items():
                expected_score = probability_to_expected_score(probabilities)
                print(f"    {item}: 期待値 {expected_score:.2f} (確率分布: {probabilities})")
            print()
        
        # turn_segmentationモジュールのcreate_turn_textを使用してテキストを作成
        conversation_text = create_turn_text(counselor_turns, client_turns, max_turns)
        
    else:
        conversation_text = str(dialogue)
        turn_evaluations = []
    
    # 17項目それぞれについて確率分布の加重平均を計算
    item_weighted_probabilities = {}
    for item in EVALUATION_ITEMS:
        item_probabilities = []
        for turn_eval in turn_evaluations:
            item_probabilities.append(turn_eval.get(item, [0.0, 0.0, 0.1, 0.8, 0.1, 0.0]))
        
        if item_probabilities:
            weighted_probs = calculate_weighted_average_probabilities(item_probabilities)
            item_weighted_probabilities[item] = weighted_probs
        else:
            item_weighted_probabilities[item] = [0.0, 0.0, 0.1, 0.8, 0.1, 0.0]
    
    # 17項目の評価確率分布をプロンプトに含める
    evaluation_prompt = ""
    for i, turn_eval in enumerate(turn_evaluations):
        evaluation_prompt += f"\n--- ターン {i+1} の17項目確率分布 ---\n"
        for item, probabilities in turn_eval.items():
            expected_score = probability_to_expected_score(probabilities)
            evaluation_prompt += f"{item}: 期待値 {expected_score:.2f} (確率: {probabilities})\n"
    
    # 17項目の加重平均確率分布をプロンプトに含める
    weighted_averages_prompt = ""
    for item, probabilities in item_weighted_probabilities.items():
        expected_score = probability_to_expected_score(probabilities)
        weighted_averages_prompt += f"{item}: 期待値 {expected_score:.2f} (確率: {probabilities})\n"
    
    # 統一されたプロンプトを使用（全項目評価用）
    prompt = create_unified_evaluation_prompt(turn_list, review)
    
    # 評価結果を追加
    prompt += f"\n\n評価結果:\n{evaluation_prompt}\n\n加重平均結果:\n{weighted_averages_prompt}"
    
    return prompt 