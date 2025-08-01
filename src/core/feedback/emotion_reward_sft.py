import torch
import logging
import os
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
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

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
            print(f"ディレクトリ作成: {directory}")
        else:
            print(f"ディレクトリ既存: {directory}")
    
    return directories

def prepare_supervised_finetuning_data(data, llm_pipeline):
    """教師ありファインチューニング用のデータを準備"""
    finetuning_data = []
    
    print("=== 教師ありファインチューニングデータ準備 ===")
    print(f"📊 データ件数: {len(data)}")
    
    # データの構造をデバッグ
    print(f"🔍 データの型: {type(data)}")
    if len(data) > 0:
        print(f"🔍 最初のデータの型: {type(data[0])}")
        print(f"🔍 最初のデータのキー: {data[0].keys() if hasattr(data[0], 'keys') else 'Not a dict'}")
    

    
    processed_count = 0
    total_turns = 0
    
    for i in range(len(data)):
        if i % 50 == 0:
            print(f"🔄 処理中: {i}/{len(data)} ({i/len(data)*100:.1f}%)")
        try:
            data_item = data[i]
        except Exception as e:
            print(f"data[{i}] アクセス失敗: {e}")
            print(f"dataの型: {type(data)}")
            print(f"dataのrepr: {repr(data)}")
            raise

        dialogue = data_item['dialogue']
        review = data_item['review_by_client_jp']
        

        
        # ターン分割を実行 - dialogueがlist型の場合も処理
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
        

        
        total_turns += len(turn_list)
        
        # 各ターンに対して17項目の評価スコアを計算
        for turn_idx, turn in enumerate(turn_list):
            if turn_idx % 10 == 0:
                print(f"=== ターン {turn_idx + 1}/{len(turn_list)} の処理 ===")
            
            # 17項目の確率分布を計算（LLM使用）
            try:
                from .llm_evaluation import evaluate_turn_on_items
            except ImportError:
                from llm_evaluation import evaluate_turn_on_items
            evaluation_probabilities = evaluate_turn_on_items(turn, review, llm_pipeline)
            
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
                
        
                
                # プロンプトを作成
                counselor_text = ""
                client_text = ""
                for utterance in turn:
                    if utterance['role'] == 'counselor':
                        counselor_text += f"カウンセラー: {utterance['utterance']}\n"
                    elif utterance['role'] == 'client':
                        client_text += f"クライアント: {utterance['utterance']}\n"
                
                prompt = f"""Rate {item}:

C: {counselor_text[:20]}...
U: {client_text[:20]}...

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
                        print(f"✅ LLM応答成功: {item} - 確率分布: {llm_response}")
                    else:
                        print(f"❌ LLM応答失敗: {item} - デフォルト確率分布を使用")
                except Exception as e:
                    print(f"❌ LLM呼び出しエラー: {item} - {e}")
                    # デフォルトの応答を使用
                
                finetuning_data.append({
                    "prompt": prompt,
                    "response": response,
                    "probabilities": probabilities,
                    "expected_score": score,
                    "item": item,
                    "turn_idx": turn_idx
                })
        
        processed_count += 1
    
    print(f"✅ ファインチューニングデータ準備完了:")
    print(f"   - 処理済みデータ: {processed_count}件")
    print(f"   - 総ターン数: {total_turns}")
    print(f"   - 生成されたサンプル数: {len(finetuning_data)}件")
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

def run_supervised_finetuning(tokenizer, model, llm_pipeline, train_data, valid_data):
    print("\n=== 教師ありファインチューニング開始 ===")
    
    # ファインチューニングデータを準備（train_dataは既に8割のデータ）
    print("📊 学習データの準備を開始...")
    train_finetuning_data = prepare_supervised_finetuning_data(train_data, llm_pipeline)
    
    # 検証データを準備（valid_dataは既に1割のデータ）
    print("📊 検証データの準備を開始...")
    val_finetuning_data = prepare_supervised_finetuning_data(valid_data, llm_pipeline)
    
    print(f"✅ データ準備完了:")
    print(f"   - 学習データ: {len(train_finetuning_data)}件")
    print(f"   - 検証データ: {len(val_finetuning_data)}件")
    
    if len(train_finetuning_data) == 0:
        print("❌ 学習データが0件です。データ処理に問題があります。")
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
        overwrite_output_dir=True,
        num_train_epochs=10,                    # エポック数10
        per_device_train_batch_size=32,         # バッチサイズ32（4GPUで128）
        per_device_eval_batch_size=32,          # 評価バッチサイズ32
        eval_steps=200,                         # 200ステップごとに評価
        save_steps=500,                         # 500ステップごとに保存
        warmup_steps=200,                       # ウォームアップステップ200
        learning_rate=2e-5,                     # 学習率2e-5
        weight_decay=0.01,                      # 重み減衰0.01
        logging_dir="./logs_supervised",        # ログディレクトリ
        logging_steps=100,                      # 100ステップごとにログ
        evaluation_strategy="steps",            # ステップごとに評価
        save_strategy="steps",                  # ステップごとに保存
        load_best_model_at_end=True,           # 最良モデルをロード
        metric_for_best_model="eval_loss",     # 評価指標
        greater_is_better=False,               # 小さい方が良い
        report_to=None,                        # レポート無効
        remove_unused_columns=False,           # 未使用カラムを削除しない
        dataloader_pin_memory=False,           # ピンメモリ無効
        dataloader_num_workers=0,              # ワーカー数0
        gradient_accumulation_steps=1,         # 勾配蓄積ステップ1
        fp16=False,                            # FP16無効
        bf16=True,                             # BF16有効
        optim="adamw_torch",                   # オプティマイザー
        lr_scheduler_type="cosine",            # スケジューラー
        warmup_ratio=0.1,                      # ウォームアップ比率
        max_grad_norm=1.0,                     # 勾配クリッピング
        seed=42,                               # シード
        data_seed=42,                          # データシード
        group_by_length=True,                  # 長さでグループ化
        length_column_name="length",           # 長さカラム名
        ignore_data_skip=False,                # データスキップ無視
        label_names=["labels"],                # ラベル名
        ddp_find_unused_parameters=False,      # DDP未使用パラメータ無視
        ddp_bucket_cap_mb=25,                 # DDPバケットサイズ
        dataloader_drop_last=False,            # 最後のバッチを削除しない
        eval_accumulation_steps=None,          # 評価蓄積ステップ
        eval_delay=0,                          # 評価遅延
        save_on_each_node=False,               # 各ノードに保存しない
        save_total_limit=None,                 # 保存制限なし
        save_only_model=False,                 # モデルのみ保存
        use_cpu=False,                         # CPU使用しない
        dataloader_prefetch_factor=None,       # プリフェッチ係数
        dataloader_persistent_workers=False,   # 永続ワーカー無効
        dataloader_prefetch_factor_override=None,  # プリフェッチ係数オーバーライド
        dataloader_pin_memory_device="",       # ピンメモリデバイス
        dataloader_async_init=False,           # 非同期初期化無効
        dataloader_async_init_timeout=0,       # 非同期初期化タイムアウト
        dataloader_async_init_batch_size=0,    # 非同期初期化バッチサイズ
        dataloader_async_init_num_workers=0,   # 非同期初期化ワーカー数
        dataloader_async_init_pin_memory=False,  # 非同期初期化ピンメモリ無効
        dataloader_async_init_prefetch_factor=None,  # 非同期初期化プリフェッチ係数
        dataloader_async_init_persistent_workers=False,  # 非同期初期化永続ワーカー無効
        dataloader_async_init_timeout_override=None,  # 非同期初期化タイムアウトオーバーライド
        dataloader_async_init_batch_size_override=None,  # 非同期初期化バッチサイズオーバーライド
        dataloader_async_init_num_workers_override=None,  # 非同期初期化ワーカー数オーバーライド
        dataloader_async_init_pin_memory_override=None,  # 非同期初期化ピンメモリオーバーライド
        dataloader_async_init_prefetch_factor_override=None,  # 非同期初期化プリフェッチ係数オーバーライド
        dataloader_async_init_persistent_workers_override=None,  # 非同期初期化永続ワーカーオーバーライド
        dataloader_async_init_timeout_override_override=None,  # 非同期初期化タイムアウトオーバーライドオーバーライド
        dataloader_async_init_batch_size_override_override=None,  # 非同期初期化バッチサイズオーバーライドオーバーライド
        dataloader_async_init_num_workers_override_override=None,  # 非同期初期化ワーカー数オーバーライドオーバーライド
        dataloader_async_init_pin_memory_override_override=None,  # 非同期初期化ピンメモリオーバーライドオーバーライド
        dataloader_async_init_prefetch_factor_override_override=None,  # 非同期初期化プリフェッチ係数オーバーライドオーバーライド
        dataloader_async_init_persistent_workers_override_override=None,  # 非同期初期化永続ワーカーオーバーライドオーバーライド
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
    print("🚀 ファインチューニング開始...")
    print(f"   - 総ステップ数: {len(train_dataset) // training_args.per_device_train_batch_size * training_args.num_train_epochs}")
    print(f"   - エポック数: {training_args.num_train_epochs}")
    print(f"   - バッチサイズ: {training_args.per_device_train_batch_size}")
    print(f"   - 学習率: {training_args.learning_rate}")
    
    trainer.train()
    
    # モデルを保存
    print("💾 モデルを保存中...")
    trainer.save_model()
    tokenizer.save_pretrained("./supervised_finetuned_model")
    print("✅ ファインチューニング完了！モデルを保存しました。")
    
    return trainer, tokenizer

def evaluate_finetuned_model(trainer, tokenizer, test_data, llm_pipeline):
    """ファインチューニングされたモデルの評価"""
    print("\n=== モデル評価開始 ===")
    
    # テストデータの準備
    test_finetuning_data = prepare_supervised_finetuning_data(test_data, llm_pipeline)
    
    # 評価結果
    results = {
        "model_predictions": [],
        "llm_predictions": [],
        "ground_truth": []
    }
    
    # 各テストデータについて予測
    for i, data in enumerate(test_finetuning_data):
        if i % 10 == 0:
            print(f"評価中: {i}/{len(test_finetuning_data)}")
        
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
    
    print("評価完了！")
    return results

def main():
    """メイン実行関数"""
    
    # 出力用ディレクトリを作成
    print("\n=== ディレクトリ準備 ===")
    create_output_directories()
    
    # データセットを読み込み
    print("\n=== データセット読み込み ===")
    train_data, test_data, valid_data = load_and_split_dataset()
    
    # モデルとパイプラインを初期化
    tokenizer, model, llm_pipeline = initialize_model_and_pipeline()
    
    # 教師ありファインチューニングを実行
    try:
        # 本格的なファインチューニング実行
        trainer, tokenizer = run_supervised_finetuning(tokenizer, model, llm_pipeline, train_data, valid_data)
        print("教師ありファインチューニングが正常に完了しました。")
        
        # モデル評価を実行
        try:
            results = evaluate_finetuned_model(trainer, tokenizer, test_data, llm_pipeline)
            print("モデル評価が正常に完了しました。")
            print(f"評価サンプル数: {len(results['model_predictions'])}")
        except Exception as e:
            print(f"モデル評価エラー: {e}")
            
    except Exception as e:
        print(f"ファインチューニングエラー: {e}")
        import traceback
        print("詳細なエラー情報:")
        traceback.print_exc()
        print("LLMベースの評価システムを使用します。")

if __name__ == "__main__":
    main()