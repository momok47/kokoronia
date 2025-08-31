#!/usr/bin/env python3
"""
ファインチューニング済みモデルを使用して会話の評価点数を予測するスクリプト

このスクリプトは、カウンセリング対話に対する満足度評価点数（0-5点）を予測します。

使用方法:
1. 保存されたモデル一覧表示:
   python use_saved_model.py --list

2. 会話テキストの評価点数予測:
   python use_saved_model.py --model-id ft:gpt-4o-mini-2024-07-18:personal::CAJ6PxFB --evaluate-conversation "対話テキスト" --evaluation-item "聴いてもらえた、わかってもらえたと感じた"

3. インタラクティブ評価モード:
   python use_saved_model.py --interactive --model-id ft:gpt-4o-mini-2024-07-18:personal::CAJ6PxFB

4. 最新のモデルで評価:
   python use_saved_model.py --use-latest --evaluate-conversation "対話テキスト"

5. 複数評価項目での一括評価:
   python use_saved_model.py --comprehensive-evaluation --model-id ft:gpt-4o-mini-2024-07-18:personal::CAJ6PxFB --conversation-file "conversation.txt"
"""

import os
import argparse
import logging
from pathlib import Path
from dotenv import load_dotenv

from openai_sft import OpenAISFT
import re
import json

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 評価項目リスト
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

def create_evaluation_prompt(conversation_text: str, evaluation_item: str) -> str:
    """
    評価用プロンプトを作成
    
    Args:
        conversation_text: 対話テキスト
        evaluation_item: 評価項目
        
    Returns:
        評価用プロンプト
    """
    prompt = f"""### 指示
以下の対話について「{evaluation_item}」の満足度を相談者の視点で0～5点で評価し、各点数の確率を出力してください。

### 対話
{conversation_text}

### 出力形式（数値のみ）
0点: XX%
1点: XX%
2点: XX%
3点: XX%
4点: XX%
5点: XX%"""
    
    return prompt

def parse_evaluation_response(response: str) -> dict:
    """
    評価応答をパースして点数と確率を抽出
    
    Args:
        response: モデルの応答
        
    Returns:
        点数と確率の辞書
    """
    scores = {}
    lines = response.strip().split('\n')
    
    for line in lines:
        # 「X点: XX%」の形式を抽出
        match = re.search(r'(\d+)点:\s*(\d+)%', line)
        if match:
            score = int(match.group(1))
            probability = int(match.group(2))
            scores[score] = probability
    
    return scores

def calculate_expected_score(scores: dict) -> float:
    """
    期待値スコアを計算
    
    Args:
        scores: 点数と確率の辞書
        
    Returns:
        期待値スコア
    """
    if not scores:
        return 0.0
    
    total_probability = sum(scores.values())
    if total_probability == 0:
        return 0.0
    
    expected_score = sum(score * (prob / total_probability) for score, prob in scores.items())
    return expected_score

def evaluate_conversation(sft: OpenAISFT, model_id: str, conversation_text: str, evaluation_item: str = None):
    """
    会話の評価点数を予測
    
    Args:
        sft: OpenAISFTインスタンス
        model_id: 使用するモデルID
        conversation_text: 評価する対話テキスト
        evaluation_item: 評価項目（指定しない場合はデフォルト項目を使用）
    """
    if evaluation_item is None:
        evaluation_item = "聴いてもらえた、わかってもらえたと感じた"
    
    logger.info(f"対話評価実行: {model_id}")
    logger.info(f"評価項目: {evaluation_item}")
    
    # 評価プロンプト作成
    prompt = create_evaluation_prompt(conversation_text, evaluation_item)
    
    try:
        # モデルに送信
        from openai import OpenAI
        client = OpenAI(api_key=sft.api_key)
        
        response = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.3  # 評価では一貫性を重視して低めに設定
        )
        
        response_text = response.choices[0].message.content
        
        # 応答をパース
        scores = parse_evaluation_response(response_text)
        expected_score = calculate_expected_score(scores)
        
        # 結果表示
        print(f"\n=== 対話評価結果 ===")
        print(f"評価項目: {evaluation_item}")
        print(f"期待値スコア: {expected_score:.2f}/5.0")
        print(f"\n確率分布:")
        for score in range(6):
            prob = scores.get(score, 0)
            print(f"  {score}点: {prob:2d}%")
        
        print(f"\n生の応答:")
        print(response_text)
        
        return {
            'evaluation_item': evaluation_item,
            'expected_score': expected_score,
            'probability_distribution': scores,
            'raw_response': response_text
        }
        
    except Exception as e:
        logger.error(f"評価エラー: {e}")
        return None

def comprehensive_evaluation(sft: OpenAISFT, model_id: str, conversation_text: str):
    """
    複数の評価項目で包括的評価を実行
    
    Args:
        sft: OpenAISFTインスタンス
        model_id: 使用するモデルID
        conversation_text: 評価する対話テキスト
    """
    logger.info(f"包括的評価実行: {model_id}")
    logger.info(f"評価項目数: {len(EVALUATION_ITEMS)}")
    
    results = {}
    
    print(f"\n=== 包括的対話評価 ===")
    print(f"モデル: {model_id}")
    print(f"評価項目数: {len(EVALUATION_ITEMS)}")
    print("=" * 60)
    
    for i, item in enumerate(EVALUATION_ITEMS, 1):
        print(f"\n[{i}/{len(EVALUATION_ITEMS)}] {item}")
        
        result = evaluate_conversation(sft, model_id, conversation_text, item)
        if result:
            results[item] = result
            print(f"期待値: {result['expected_score']:.2f}/5.0")
        else:
            print("評価失敗")
    
    # 総合結果表示
    if results:
        print(f"\n=== 総合結果 ===")
        total_score = sum(r['expected_score'] for r in results.values())
        average_score = total_score / len(results)
        print(f"平均スコア: {average_score:.2f}/5.0")
        
        # 上位・下位項目
        sorted_results = sorted(results.items(), key=lambda x: x[1]['expected_score'], reverse=True)
        
        print(f"\n🏆 高評価項目 TOP3:")
        for item, result in sorted_results[:3]:
            print(f"  {result['expected_score']:.2f} - {item}")
        
        print(f"\n⚠️ 改善項目 TOP3:")
        for item, result in sorted_results[-3:]:
            print(f"  {result['expected_score']:.2f} - {item}")
    
    return results

def interactive_evaluation_mode(sft: OpenAISFT, model_id: str):
    """
    インタラクティブ評価モード
    
    Args:
        sft: OpenAISFTインスタンス  
        model_id: 使用するモデルID
    """
    logger.info(f"インタラクティブ評価モード開始: {model_id}")
    logger.info("対話テキストを入力してください（'quit'または'exit'で終了）")
    
    while True:
        try:
            print(f"\n{'='*50}")
            print("対話テキストを入力してください:")
            print("（複数行の場合は、最後に空行を入力してください）")
            
            lines = []
            while True:
                line = input()
                if line.strip() == "":
                    break
                lines.append(line)
            
            conversation_text = '\n'.join(lines).strip()
            
            if conversation_text.lower() in ['quit', 'exit', 'q']:
                logger.info("評価を終了します")
                break
                
            if not conversation_text:
                continue
            
            # 評価項目選択
            print(f"\n評価項目を選択してください:")
            print("0. 全項目で包括評価")
            for i, item in enumerate(EVALUATION_ITEMS[:10], 1):  # 最初の10項目を表示
                print(f"{i}. {item}")
            print("11. その他（カスタム項目）")
            
            choice = input("\n番号を入力 (0-11): ").strip()
            
            if choice == "0":
                # 包括評価
                comprehensive_evaluation(sft, model_id, conversation_text)
            elif choice.isdigit() and 1 <= int(choice) <= 10:
                # 特定項目評価
                item_index = int(choice) - 1
                evaluation_item = EVALUATION_ITEMS[item_index]
                evaluate_conversation(sft, model_id, conversation_text, evaluation_item)
            elif choice == "11":
                # カスタム項目
                custom_item = input("カスタム評価項目を入力してください: ").strip()
                if custom_item:
                    evaluate_conversation(sft, model_id, conversation_text, custom_item)
            else:
                print("無効な選択です")
                
        except KeyboardInterrupt:
            logger.info("\n評価が中断されました")
            break
        except Exception as e:
            logger.error(f"評価エラー: {e}")

def load_conversation_from_file(file_path: str) -> str:
    """
    ファイルから会話テキストを読み込み
    
    Args:
        file_path: ファイルパス
        
    Returns:
        会話テキスト
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        logger.error(f"ファイル読み込みエラー: {e}")
        return ""

def main():
    parser = argparse.ArgumentParser(description='ファインチューニング済みモデルで対話評価')
    
    # 動作モード
    parser.add_argument('--list', action='store_true', help='保存されたモデル一覧表示')
    parser.add_argument('--interactive', action='store_true', help='インタラクティブ評価モード')
    parser.add_argument('--use-latest', action='store_true', help='最新のモデルを自動選択')
    parser.add_argument('--comprehensive-evaluation', action='store_true', help='全評価項目での包括評価')
    
    # モデル指定
    parser.add_argument('--model-id', type=str, help='使用するモデルID')
    parser.add_argument('--model-file', type=str, help='モデル情報ファイルのパス')
    
    # 評価パラメータ
    parser.add_argument('--evaluate-conversation', type=str, help='評価する対話テキスト')
    parser.add_argument('--conversation-file', type=str, help='対話テキストファイルのパス')
    parser.add_argument('--evaluation-item', type=str, help='評価項目（指定しない場合はデフォルト）')
    
    # API設定
    parser.add_argument('--api-key', type=str, help='OpenAI APIキー')
    
    args = parser.parse_args()
    
    # プロジェクトルートの.envファイルを読み込み
    project_root = Path(__file__).parent.parent.parent.parent
    env_path = project_root / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        logger.info(f".envファイルを読み込みました: {env_path}")
    
    # APIキーの確認
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OpenAI APIキーが設定されていません。")
        logger.error(".envファイルにOPENAI_API_KEY=your-api-key-here を設定するか、")
        logger.error("--api-key オプションを使用してください。")
        return 1
    
    try:
        # SFTインスタンス作成
        sft = OpenAISFT(api_key=api_key)
        
        if args.list:
            # モデル一覧表示
            logger.info("=== 保存されているモデル一覧 ===")
            models = sft.list_saved_models()
            
            if not models:
                logger.info("保存されているモデルがありません")
                return 0
            
            for i, model in enumerate(models, 1):
                print(f"{i}. {model['model_id']}")
                print(f"   タイムスタンプ: {model['timestamp']}")
                print(f"   学習パラメータ: {model['training_params']}")
                print(f"   ファイル: {model['file_path']}")
                print()
            
            return 0
        
        # モデルIDの決定
        model_id = None
        
        if args.use_latest:
            # 最新のモデルを自動選択
            models = sft.list_saved_models()
            if models:
                model_id = models[0]['model_id']  # 最新のモデル
                logger.info(f"最新のモデルを選択: {model_id}")
            else:
                logger.error("保存されているモデルがありません")
                return 1
                
        elif args.model_file:
            # モデル情報ファイルから読み込み
            model_info = sft.load_model_from_file(args.model_file)
            model_id = model_info['model_id']
            
        elif args.model_id:
            # 直接指定
            model_id = args.model_id
            
        else:
            logger.error("モデルを指定してください (--model-id, --model-file, または --use-latest)")
            return 1
        
        # 対話テキストの取得
        conversation_text = None
        if args.evaluate_conversation:
            conversation_text = args.evaluate_conversation
        elif args.conversation_file:
            conversation_text = load_conversation_from_file(args.conversation_file)
            if not conversation_text:
                logger.error("ファイルから対話テキストを読み込めませんでした")
                return 1
        
        if args.interactive:
            # インタラクティブ評価モード
            interactive_evaluation_mode(sft, model_id)
            
        elif args.comprehensive_evaluation:
            # 包括的評価
            if not conversation_text:
                logger.error("--evaluate-conversation または --conversation-file を指定してください")
                return 1
            comprehensive_evaluation(sft, model_id, conversation_text)
            
        elif conversation_text:
            # 単一項目評価
            evaluate_conversation(sft, model_id, conversation_text, args.evaluation_item)
            
        else:
            logger.error("--interactive, --comprehensive-evaluation, または --evaluate-conversation を指定してください")
            return 1
        
        return 0
        
    except Exception as e:
        logger.error(f"エラーが発生しました: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
