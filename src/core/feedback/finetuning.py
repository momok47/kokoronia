import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments, 
    Trainer,
    pipeline
)
from datasets import Dataset
from typing import Dict, List
import logging
from .data_processing import EVALUATION_ITEMS
from .llm_evaluation import evaluate_turn_on_items
from .turn_segmentation import segment_turns, create_turn_list

logger = logging.getLogger(__name__)

def prepare_supervised_finetuning_data(data, llm_pipeline):
    """教師ありファインチューニング用のデータを準備"""
    finetuning_data = []
    
    print("=== 教師ありファインチューニングデータ準備 ===")
    
    for i in range(len(data)):
        if i % 100 == 0:
            print(f"処理中: {i}/{len(data)}")
        
        dialogue = data[i]['dialogue']
        review = data[i]['review_by_client_jp']
        
        # ターン分割を実行
        if isinstance(dialogue, dict) and 'dialogue' in dialogue:
            turns = dialogue['dialogue']
            counselor_turns, client_turns, max_turns = segment_turns(turns)
            turn_list = create_turn_list(counselor_turns, client_turns, max_turns)
            
            # 各ターンに対して17項目の評価スコアを計算
            for turn_idx, turn in enumerate(turn_list):
                # ターンのテキストを作成
                turn_text = ""
                for utterance in turn:
                    role = utterance['role']
                    text = utterance['utterance']
                    turn_text += f"{role}: {text}\n"
                
                # 17項目の確率分布を計算（LLM使用）
                evaluation_probabilities = evaluate_turn_on_items(turn, review, llm_pipeline)
                
                # 各評価項目についてプロンプトと応答のペアを作成
                for item in EVALUATION_ITEMS:
                    probabilities = evaluation_probabilities.get(item, [0.0, 0.0, 0.1, 0.8, 0.1, 0.0])
                    # 確率分布から期待値を計算
                    from .data_processing import probability_to_expected_score
                    score = probability_to_expected_score(probabilities)
                    
                    # プロンプトを作成
                    counselor_text = ""
                    client_text = ""
                    for utterance in turn:
                        if utterance['role'] == 'counselor':
                            counselor_text += f"カウンセラー: {utterance['utterance']}\n"
                        elif utterance['role'] == 'client':
                            client_text += f"クライアント: {utterance['utterance']}\n"
                    
                    prompt = f"""以下のカウンセリング会話について、{item}の観点からクライアントの評価スコアを0~5で予測してください。

会話内容:
カウンセラーの発言:
{counselor_text}

クライアントの発言:
{client_text}

クライアントの評価:
{review}

評価基準:
0=非常に悪い, 1=悪い, 2=普通, 3=良い, 4=非常に良い, 5=最高

{item}の観点でのクライアントの評価スコア（0-5の整数）:"""
                    
                    # 応答を作成
                    response = f" {int(score)}"
                    
                    finetuning_data.append({
                        "prompt": prompt,
                        "response": response,
                        "score": score,
                        "item": item,
                        "turn_idx": turn_idx
                    })
    
    print(f"ファインチューニングデータ準備完了: {len(finetuning_data)}件")
    return finetuning_data

class SupervisedFinetuningDataCollator:
    """教師ありファインチューニング用のデータコレーター"""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, batch):
        # プロンプトと応答を結合
        texts = []
        for item in batch:
            full_text = item["prompt"] + item["response"]
            texts.append(full_text)
        
        # トークン化
        tokenized = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # ラベルを作成（プロンプト部分は-100、応答部分はトークンID）
        labels = []
        for i, item in enumerate(batch):
            prompt_tokens = self.tokenizer(
                item["prompt"], 
                return_tensors="pt",
                add_special_tokens=False
            )["input_ids"][0]
            
            response_tokens = self.tokenizer(
                item["response"], 
                return_tensors="pt",
                add_special_tokens=False
            )["input_ids"][0]
            
            # プロンプト部分は-100、応答部分はトークンID
            label = torch.cat([
                torch.full((len(prompt_tokens),), -100),
                response_tokens
            ])
            
            # パディング
            if len(label) < 512:
                label = torch.cat([label, torch.full((512 - len(label),), -100)])
            else:
                label = label[:512]
            
            labels.append(label)
        
        labels = torch.stack(labels)
        
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": labels
        }

class SupervisedFinetuningTrainer(Trainer):
    """教師ありファインチューニング用のカスタムトレーナー"""
    def compute_loss(self, model, inputs, return_outputs=False):
        # モデルの出力を取得
        outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        
        # 損失を計算（-100のラベルは無視）
        loss = outputs.loss
        
        return (loss, outputs) if return_outputs else loss 