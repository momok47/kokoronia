import torch
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from .data_processing import load_and_split_dataset
from .llm_evaluation import create_emotion_prompt
from .finetuning import (
    prepare_supervised_finetuning_data,
    SupervisedFinetuningDataCollator,
    SupervisedFinetuningTrainer
)

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_model_and_pipeline():
    """モデルとパイプラインを初期化"""
    print("\n=== モデル読み込み ===")
    model_name = "tokyotech-llm/Swallow-7b-instruct-hf"
    print(f"読み込み中: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # パディングトークンの設定
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"eos_token: {tokenizer.eos_token}")
    print(f"pad_token: {tokenizer.pad_token}")

    # LLMパイプラインの初期化
    print("\n=== LLMパイプライン初期化 ===")
    try:
        llm_pipeline = pipeline(
            "text-generation",
            model=model_name,
            tokenizer=tokenizer,
            device=0 if torch.cuda.is_available() else -1,
            max_length=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
        print("LLMパイプライン初期化完了")
    except Exception as e:
        print(f"LLMパイプライン初期化エラー: {e}")
        llm_pipeline = None

    return tokenizer, model, llm_pipeline

def run_supervised_finetuning(tokenizer, model, llm_pipeline, train_data):
    """教師ありファインチューニングを実行"""
    print("\n=== 教師ありファインチューニング開始 ===")
    
    # ファインチューニングデータを準備
    finetuning_data = prepare_supervised_finetuning_data(train_data, llm_pipeline)
    
    # データセットに変換
    from datasets import Dataset
    finetuning_dataset = Dataset.from_list(finetuning_data)
    
    print(f"ファインチューニングデータセットサイズ: {len(finetuning_dataset)}")
    
    # データコレーターを初期化
    data_collator = SupervisedFinetuningDataCollator(tokenizer)
    
    # トレーニング引数を設定
    from transformers import TrainingArguments
    training_args = TrainingArguments(
        output_dir="./supervised_finetuned_model",
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        eval_steps=500,
        save_steps=1000,
        warmup_steps=100,
        prediction_loss_only=True,
        logging_dir="./logs_supervised",
        logging_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
    )
    
    # トレーナーを初期化
    trainer = SupervisedFinetuningTrainer(
        model=model,
        args=training_args,
        train_dataset=finetuning_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # ファインチューニングを実行
    print("ファインチューニング開始...")
    trainer.train()
    
    # モデルを保存
    trainer.save_model()
    print("ファインチューニング完了！モデルを保存しました。")
    
    return trainer

def main():
    """メイン実行関数"""
    # データセットを読み込み
    train_data, test_data, valid_data = load_and_split_dataset()
    
    # モデルとパイプラインを初期化
    tokenizer, model, llm_pipeline = initialize_model_and_pipeline()
    
    # 教師ありファインチューニングを実行
    try:
        trainer = run_supervised_finetuning(tokenizer, model, llm_pipeline, train_data)
        print("教師ありファインチューニングが正常に完了しました。")
    except Exception as e:
        print(f"ファインチューニングエラー: {e}")
        print("LLMベースの評価システムを使用します。")

if __name__ == "__main__":
    main()