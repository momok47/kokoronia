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

def create_unified_evaluation_prompt(conversation_text: str, turn_index: int) -> str:
    """
    LLMに渡すプロンプトを生成する（Few-shotプロンプティング版）
    """
    # Few-shotプロンプティングのための完璧な「お手本」を用意する
    example_input = (
        "counselor: こんにちは！今日はどのようなお話を聞かせていただけますか？\n"
        "client: 最近、仕事でストレスが溜まっていて...\n"
        "counselor: お疲れ様です。そのストレスについて、もう少し詳しく教えていただけますか？\n"
        "client: はい、ありがとうございます。上司との関係で悩んでいて...\n"
        "counselor: それは大変でしたね。上司との関係で具体的にどのようなことが起きているのでしょうか？\n"
        "client: とても助かりました！話を聞いてもらえて、気持ちが楽になりました。"
    )

    example_output = "0点: 0%, 1点: 0%, 2点: 10%, 3点: 30%, 4点: 40%, 5点: 20%"

    # 実際のプロンプトを組み立てる
    prompt = (
        "あなたは対話評価の専門家です。提示された対話を分析し、会話全体のポジティブさを0点から5点の6段階で評価し、その確率分布を算出してください。\n\n"
        "--- お手本 ---\n"
        "【分析対象の対話】:\n"
        f"{example_input}\n"
        "【出力】:\n"
        f"{example_output}\n\n"
        "--- 本番 ---\n"
        "【分析対象の対話】:\n"
        f"{conversation_text}\n"
        "【出力】:\n"
    )
    
    return prompt



def call_llm_for_probability_distribution(tokenizer, model, conversation_text: str) -> List[float]:
    """
    LLMを呼び出して、会話のテキストから確率分布を取得する
    """
    prompt = create_unified_evaluation_prompt(conversation_text, 0)
    
    # モデルへの入力を準備
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

    # 🚨【重要】テキスト生成パラメータを調整して、LLMの応答を制御
    response_ids = model.generate(
        input_ids,
        max_new_tokens=100,         # 生成する最大トークン数
        do_sample=True,             # 👈 サンプリングを有効にし、多様な出力を促す
        temperature=0.7,            # 👈 出力のランダム性を制御 (創造性を少し加える)
        top_p=0.95,                 # 👈 上位95%の確率を持つ単語からサンプリング
        repetition_penalty=1.15,    # 👈 繰り返しを抑制するためのペナルティ
        pad_token_id=tokenizer.eos_token_id  # pad_token_id を eos_token_id に設定
    )
    
    # 応答をデコード
    response_only = tokenizer.decode(response_ids[0][input_ids.shape[-1]:], skip_special_tokens=True)
    
    logger.info(f"LLM Raw Response: '{response_only}'")
    
    # 確率を抽出
    probabilities = parse_probabilities_from_llm_response(response_only)
    logger.info(f"Parsed Probabilities: {probabilities}")

    return probabilities

def parse_probabilities_from_llm_response(response: str) -> List[float]:
    """
    LLMの応答テキスト（多少の揺れや余計な文章があっても対応可能）から
    確率分布を抽出する、より頑健な関数。
    """
    # 期待する確率分布の行をすべて見つけるための正規表現
    # "0点: 10.5%" や " 1 点 : 5 % " のような表記の揺れにも対応
    pattern = r"(\d)\s*点\s*:\s*([\d\.]+)\s*%"
    
    try:
        matches = re.findall(pattern, response)
        
        if not matches:
            logger.warning(f"応答から確率のパターンが見つかりませんでした。応答: '{response}'")
            return [1/6] * 6

        # 抽出した確率を格納する辞書を初期化
        probabilities_dict = {i: 0.0 for i in range(6)}
        
        for score_str, prob_str in matches:
            score = int(score_str)
            prob = float(prob_str)
            if 0 <= score <= 5:
                probabilities_dict[score] = prob / 100.0  # パーセントを小数に変換

        # 0点から5点のリスト形式に変換
        probabilities = [probabilities_dict[i] for i in range(6)]

        # 合計が0、または合計が極端にずれている場合は正規化する
        total_prob = sum(probabilities)
        if total_prob <= 0:
            logger.warning(f"抽出した確率の合計が0です。均等分布を返します。抽出結果: {probabilities}")
            return [1/6] * 6
        
        # 合計が1になるように正規化（LLMの計算ミスを補正）
        probabilities = [p / total_prob for p in probabilities]
        
        return probabilities

    except Exception as e:
        logger.error(f"確率のパース中に予期せぬエラーが発生しました: {e}\n応答: '{response}'")
        return [1/6] * 6

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
    # ターンリストから会話テキストを生成
    conversation_text = ""
    for turn in turn_list:
        role = turn.get('role', 'unknown')
        utterance = turn.get('utterance', '')
        conversation_text += f"{role}: {utterance}\n"
    
    # LLMを使用した確率分布取得
    # llm_pipelineからtokenizerとmodelを取得
    tokenizer = llm_pipeline.tokenizer
    model = llm_pipeline.model
    
    probabilities = call_llm_for_probability_distribution(tokenizer, model, conversation_text)
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
    conversation_text = ""
    for turn in turn_list:
        role = turn.get('role', 'unknown')
        utterance = turn.get('utterance', '')
        conversation_text += f"{role}: {utterance}\n"
    
    prompt = create_unified_evaluation_prompt(conversation_text, 0)
    
    # 評価結果を追加
    prompt += f"\n\n評価結果:\n{evaluation_prompt}\n\n加重平均結果:\n{weighted_averages_prompt}"
    
    return prompt 