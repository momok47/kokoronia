import torch
import logging
import os
import sys
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, Trainer
from transformers import TrainingArguments
from datasets import Dataset
from sklearn.model_selection import train_test_split

# 相対インポートを試行、失敗した場合は絶対インポート
try:
    from .data_processing import load_and_split_dataset
    from .llm_evaluation import create_emotion_prompt
except ImportError:
    from data_processing import load_and_split_dataset
    from llm_evaluation import create_emotion_prompt

# ログ設定
def setup_logging():
    """ログ設定を初期化"""
    # ログディレクトリを作成
    log_dir = "./logs_supervised"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # ログファイル名にタイムスタンプを追加
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{log_dir}/training_{timestamp}.log"
    
    # ログ設定
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    print(f"ログファイル: {log_file}")
    return logging.getLogger(__name__)

logger = setup_logging()

def create_output_directories():
    """出力用ディレクトリを作成"""
    directories = [
        "./supervised_finetuned_model",
        "./logs_supervised",
        "./model_checkpoints"
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"ディレクトリ作成: {directory}")
        else:
            logger.info(f"ディレクトリ既存: {directory}")
    
    return directories

def prepare_supervised_finetuning_data(data, llm_pipeline, max_samples=None):
    """教師ありファインチューニング用のデータを準備（全ターンを1つのプロンプトに）"""
    finetuning_data = []
    
    logger.info("=== 教師ありファインチューニングデータ準備 ===")
    logger.info(f"📊 データ件数: {len(data)}")
    
    # データサンプリング（処理時間短縮のため）
    if max_samples and len(data) > max_samples:
        import random
        random.seed(42)  # 再現性のため
        data = random.sample(data, max_samples)
        logger.info(f"📊 サンプリング後データ件数: {len(data)}")
    
    processed_count = 0
    
    for i in range(len(data)):
        if i % 10 == 0:  # 10件ごとに進捗を表示
            logger.info(f"🔄 処理中: {i}/{len(data)} ({i/len(data)*100:.1f}%)")
        
        try:
            data_item = data[i]
        except Exception as e:
            logger.error(f"data[{i}] アクセス失敗: {e}")
            continue

        dialogue = data_item['dialogue']
        review = data_item['review_by_client_jp']
        
        # ターン分割を実行
        turns = None
        if isinstance(dialogue, dict) and 'dialogue' in dialogue:
            turns = dialogue['dialogue']
        elif isinstance(dialogue, list):
            turns = dialogue
        else:
            continue
        
        try:
            from .turn_segmentation import segment_turns, create_turn_list
        except ImportError:
            from turn_segmentation import segment_turns, create_turn_list
        
        counselor_turns, client_turns, max_turns = segment_turns(turns)
        turn_list = create_turn_list(counselor_turns, client_turns, max_turns)
        
        # 全ターンの会話テキストを作成
        full_conversation_text = ""
        for turn in turn_list:
            role = turn.get('role', 'unknown')
            utterance = turn.get('utterance', '')
            full_conversation_text += f"{role}: {utterance}\n"
        
        # 17項目の確率分布を計算（全ターンを1つのプロンプトで）
        try:
            from .llm_evaluation import evaluate_conversation_on_items
        except ImportError:
            from llm_evaluation import evaluate_conversation_on_items
        
        evaluation_probabilities = evaluate_conversation_on_items(full_conversation_text, review, llm_pipeline)
        
        # 各評価項目についてプロンプトと応答のペアを作成
        try:
            from .data_processing import EVALUATION_ITEMS
        except ImportError:
            from data_processing import EVALUATION_ITEMS
        
        for item in EVALUATION_ITEMS:
            probabilities = evaluation_probabilities.get(item, [0.0, 0.0, 0.1, 0.8, 0.1, 0.0])
            
            # 確率分布から期待値を計算
            try:
                from .data_processing import probability_to_expected_score
            except ImportError:
                from data_processing import probability_to_expected_score
            score = probability_to_expected_score(probabilities)
            
            # プロンプトを作成（全会話を含む）
            prompt = f"""Rate {item}:

Conversation:
{full_conversation_text[:500]}...

【重要】必ず以下の形式で回答してください。他の説明は一切不要です：

0点の確率: [数値]
1点の確率: [数値]
2点の確率: [数値]
3点の確率: [数値]
4点の確率: [数値]
5点の確率: [数値]

Answer:"""
            
            # 応答を作成（確率分布形式）
            response = f"""0点の確率: {probabilities[0]:.3f}
1点の確率: {probabilities[1]:.3f}
2点の確率: {probabilities[2]:.3f}
3点の確率: {probabilities[3]:.3f}
4点の確率: {probabilities[4]:.3f}
5点の確率: {probabilities[5]:.3f}"""
            
            # LLMを実際に呼び出して応答を取得
            try:
                from .llm_evaluation import call_llm_for_probability_distribution
                llm_response = call_llm_for_probability_distribution(prompt, llm_pipeline)
                if llm_response and len(llm_response) == 6:
                    # LLMの応答を使用
                    response = f"""0点の確率: {llm_response[0]:.3f}
1点の確率: {llm_response[1]:.3f}
2点の確率: {llm_response[2]:.3f}
3点の確率: {llm_response[3]:.3f}
4点の確率: {llm_response[4]:.3f}
5点の確率: {llm_response[5]:.3f}"""
                    logger.info(f"✅ LLM応答成功: {item} - データ{i}")
                else:
                    logger.warning(f"❌ LLM応答失敗: {item} - データ{i} - デフォルト確率分布を使用")
            except Exception as e:
                logger.error(f"❌ LLM呼び出しエラー: {item} - データ{i} - {e}")
                # デフォルトの応答を使用
            
            finetuning_data.append({
                "prompt": prompt,
                "response": response,
                "probabilities": probabilities,
                "expected_score": score,
                "item": item,
                "data_index": i
            })
        
        processed_count += 1
    
    logger.info(f"✅ ファインチューニングデータ準備完了:")
    logger.info(f"   - 処理済みデータ: {processed_count}件")
    logger.info(f"   - 生成されたサンプル数: {len(finetuning_data)}件")
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
    """教師ありファインチューニング用のカスタムトレーナー（MSE損失）"""
    def compute_loss(self, model, inputs, return_outputs=False):
        # モデルの出力を取得
        outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        
        # 平均二乗誤差（MSE）損失を計算
        logits = outputs.logits
        
        # ラベルから有効なトークンのみを抽出（-100以外）
        labels = inputs["labels"]
        active_loss = labels.view(-1) != -100
        active_logits = logits.view(-1, logits.size(-1))
        active_labels = labels.view(-1)[active_loss]
        
        # MSE損失を計算
        loss_fct = torch.nn.MSELoss()
        loss = loss_fct(active_logits, active_labels.float())
        
        return (loss, outputs) if return_outputs else loss

def initialize_model_and_pipeline():
    """モデルとパイプラインを初期化"""
    print("\n=== モデル読み込み ===")
    model_name = "tokyotech-llm/Swallow-7b-instruct-hf"
    print(f"読み込み中: {model_name}")

    # SentencePieceの依存を回避するための環境変数を設定
    import os
    import sys
    
    # システムレベルのSentencePieceを利用するための環境変数を設定
    os.environ["PKG_CONFIG_PATH"] = "/opt/homebrew/lib/pkgconfig:" + os.environ.get("PKG_CONFIG_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = "/opt/homebrew/lib:" + os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["DYLD_LIBRARY_PATH"] = "/opt/homebrew/lib:" + os.environ.get("DYLD_LIBRARY_PATH", "")
    
    # システムレベルのPythonパッケージを追加
    sys.path.append('/Users/shirakawamomoko/Library/Python/3.11/lib/python/site-packages')
    
    # SentencePieceが利用可能かどうかを確認
    try:
        import sentencepiece
        print("SentencePiece利用可能")
    except ImportError:
        print("SentencePiece利用不可 - システムレベルのインストールを確認してください")

    try:
        # Swallowモデルを読み込み
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            local_files_only=False,
            revision="main",
            use_fast=True,  # 高速トークナイザーを使用
            legacy=False,  # 新しいトークナイザー実装を使用
            padding_side="left"  # パディングを左側に配置
        )
        
        # Swallowモデルのchat_templateを設定
        if not hasattr(tokenizer, 'chat_template') or tokenizer.chat_template is None:
            # Swallowモデルの独自chat_templateを設定
            tokenizer.chat_template = """{% for message in messages %}{% if message['role'] == 'system' %}{{ message['content'] + '\n\n'}}{% elif message['role'] == 'user' %}{{ '### 指示:\n' + message['content'] + '\n\n### 応答:\n' }}{% endif %}{% endfor %}"""
            print("Swallowモデルのchat_templateを設定しました")
        
        # モデルの読み込み
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,  # 公式推奨のデータ型
            low_cpu_mem_usage=True,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
            local_files_only=False,
            revision="main"
        )
        
        print("モデル読み込み成功")
        
    except Exception as e:
        print(f"モデル読み込みエラー: {e}")
        # フォールバック: 高速トークナイザーを無効にして再試行
        try:
            print("フォールバック: 高速トークナイザーを無効にして再試行")
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                use_fast=False,
                trust_remote_code=True,
                local_files_only=False,
                revision="main",
                legacy=True,  # レガシーモードで試行
                padding_side="left"
            )
            
            # Swallowモデルのchat_templateを設定
            if not hasattr(tokenizer, 'chat_template') or tokenizer.chat_template is None:
                tokenizer.chat_template = """{% for message in messages %}{% if message['role'] == 'system' %}{{ message['content'] + '\n\n'}}{% elif message['role'] == 'user' %}{{ '### 指示:\n' + message['content'] + '\n\n### 応答:\n' }}{% endif %}{% endfor %}"""
                print("Swallowモデルのchat_templateを設定しました")
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
                local_files_only=False,
                revision="main"
            )
            print("フォールバック成功")
        except Exception as e2:
            print(f"フォールバックも失敗: {e2}")
            # 最終フォールバック: 基本的な設定で再試行
            try:
                print("最終フォールバック: 基本的な設定で再試行")
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    trust_remote_code=True
                )
                
                # Swallowモデルのchat_templateを設定
                if not hasattr(tokenizer, 'chat_template') or tokenizer.chat_template is None:
                    tokenizer.chat_template = """{% for message in messages %}{% if message['role'] == 'system' %}{{ message['content'] + '\n\n'}}{% elif message['role'] == 'user' %}{{ '### 指示:\n' + message['content'] + '\n\n### 応答:\n' }}{% endif %}{% endfor %}"""
                    print("Swallowモデルのchat_templateを設定しました")
                
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    trust_remote_code=True
                )
                print("最終フォールバック成功")
            except Exception as e3:
                print(f"最終フォールバックも失敗: {e3}")
                raise e3

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
            model=model,
            tokenizer=tokenizer,
            device=0 if torch.cuda.is_available() else -1,
            max_length=512,
            do_sample=True,
            temperature=1.0,  # 温度を最大に
            top_p=1.0,  # top_pも最大に
            repetition_penalty=1.0,  # 繰り返しペナルティを無効化
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            # より確実に応答するための設定
            max_new_tokens=100,  # トークン数を減らす
            num_return_sequences=1,
            # early_stoppingを削除（無効なフラグ）
        )
        print("LLMパイプライン初期化完了")
    except Exception as e:
        print(f"LLMパイプライン初期化エラー: {e}")
        llm_pipeline = None

    return tokenizer, model, llm_pipeline

def run_supervised_finetuning(tokenizer, model, llm_pipeline, train_data, valid_data, max_samples=100):
    logger.info("\n=== 教師ありファインチューニング開始 ===")
    
    # ファインチューニングデータを準備（train_dataは既に8割のデータ）
    logger.info("📊 学習データの準備を開始...")
    train_finetuning_data = prepare_supervised_finetuning_data(train_data, llm_pipeline, max_samples)
    
    # 検証データを準備（valid_dataは既に1割のデータ）
    logger.info("📊 検証データの準備を開始...")
    val_finetuning_data = prepare_supervised_finetuning_data(valid_data, llm_pipeline, max_samples//4)
    
    logger.info(f"✅ データ準備完了:")
    logger.info(f"   - 学習データ: {len(train_finetuning_data)}件")
    logger.info(f"   - 検証データ: {len(val_finetuning_data)}件")
    
    if len(train_finetuning_data) == 0:
        logger.error("❌ 学習データが0件です。データ処理に問題があります。")
        raise ValueError("学習データが0件です")
    
    # データセットに変換
    from datasets import Dataset
    train_dataset = Dataset.from_list(train_finetuning_data)
    val_dataset = Dataset.from_list(val_finetuning_data)
    
    # データコレーターを初期化
    data_collator = SupervisedFinetuningDataCollator(tokenizer)
    
    # トレーニング引数を設定
    from transformers import TrainingArguments
    training_args = TrainingArguments(
        output_dir="./supervised_finetuned_model",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=3,
        learning_rate=2e-5,
        warmup_steps=100,
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=100,
        save_steps=500,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_async_init_persistent_workers=False,
        dataloader_async_init_timeout_override=None,
        dataloader_async_init_batch_size_override=None,
        dataloader_async_init_num_workers_override=None,
        dataloader_async_init_pin_memory_override=None,
        dataloader_async_init_prefetch_factor_override=None,
        dataloader_async_init_persistent_workers_override=None,
        dataloader_async_init_timeout_override_override=None,
        dataloader_async_init_batch_size_override_override=None,
        dataloader_async_init_num_workers_override_override=None,
        dataloader_async_init_pin_memory_override_override=None,
        dataloader_async_init_prefetch_factor_override_override=None,
        dataloader_async_init_persistent_workers_override_override=None,
    )
    
    # カスタムトレーナーを初期化
    trainer = SupervisedFinetuningTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # ファインチューニングを実行
    logger.info("🚀 ファインチューニング開始...")
    logger.info(f"   - 総ステップ数: {len(train_dataset) // training_args.per_device_train_batch_size * training_args.num_train_epochs}")
    logger.info(f"   - エポック数: {training_args.num_train_epochs}")
    logger.info(f"   - バッチサイズ: {training_args.per_device_train_batch_size}")
    logger.info(f"   - 学習率: {training_args.learning_rate}")
    
    trainer.train()
    
    # モデルを保存
    logger.info("💾 モデルを保存中...")
    trainer.save_model()
    tokenizer.save_pretrained("./supervised_finetuned_model")
    logger.info("✅ ファインチューニング完了！モデルを保存しました。")
    
    return trainer, tokenizer

def evaluate_finetuned_model(trainer, tokenizer, test_data, llm_pipeline, max_samples=50):
    """ファインチューニングされたモデルの評価"""
    logger.info("\n=== モデル評価開始 ===")
    
    # テストデータの準備
    test_finetuning_data = prepare_supervised_finetuning_data(test_data, llm_pipeline, max_samples)
    
    # 評価結果
    results = {
        "model_predictions": [],
        "llm_predictions": [],
        "ground_truth": []
    }
    
    # 各テストデータについて予測
    for i, data in enumerate(test_finetuning_data):
        if i % 10 == 0:
            logger.info(f"評価中: {i}/{len(test_finetuning_data)}")
        
        # ファインチューニングされたモデルでの予測
        inputs = tokenizer(data["prompt"], return_tensors="pt", truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = trainer.model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        model_response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        
        # 結果を記録
        results["model_predictions"].append(model_response)
        results["llm_predictions"].append(data["response"])
        results["ground_truth"].append(data["expected_score"])
    
    logger.info("評価完了！")
    return results

def main():
    """メイン実行関数"""
    
    # 出力用ディレクトリを作成
    logger.info("\n=== ディレクトリ準備 ===")
    create_output_directories()
    
    # データセットを読み込み
    logger.info("\n=== データセット読み込み ===")
    train_data, test_data, valid_data = load_and_split_dataset()
    
    # モデルとパイプラインを初期化
    tokenizer, model, llm_pipeline = initialize_model_and_pipeline()
    
    # 教師ありファインチューニングを実行
    try:
        # Phase 1: 100件でテスト実行
        logger.info("=== Phase 1: 100件でテスト実行 ===")
        trainer, tokenizer = run_supervised_finetuning(tokenizer, model, llm_pipeline, train_data, valid_data, max_samples=100)
        logger.info("✅ 100件でのテスト実行が正常に完了しました。")
        
        # モデル評価を実行
        try:
            results = evaluate_finetuned_model(trainer, tokenizer, test_data, llm_pipeline, max_samples=50)
            logger.info("✅ モデル評価が正常に完了しました。")
            logger.info(f"評価サンプル数: {len(results['model_predictions'])}")
            
            # 結果の要約
            logger.info("=== テスト実行結果 ===")
            logger.info(f"学習データ: 100件")
            logger.info(f"検証データ: 25件")
            logger.info(f"テストデータ: 50件")
            logger.info("✅ Phase 1完了！100件でのテスト実行が正常に終了しました。")
            logger.info("📝 次のステップ: 5000件での本格実行に進む場合は、max_samplesを5000に変更してください。")
            
        except Exception as e:
            logger.error(f"モデル評価エラー: {e}")
            
    except Exception as e:
        logger.error(f"ファインチューニングエラー: {e}")
        import traceback
        logger.error("詳細なエラー情報:")
        logger.error(traceback.format_exc())
        logger.info("LLMベースの評価システムを使用します。")

# Phase 2: 5000件での本格実行用設定
# 以下の行のコメントを外して、max_samplesを5000に変更してください
# trainer, tokenizer = run_supervised_finetuning(tokenizer, model, llm_pipeline, train_data, valid_data, max_samples=5000)
# results = evaluate_finetuned_model(trainer, tokenizer, test_data, llm_pipeline, max_samples=500)

if __name__ == "__main__":
    main()