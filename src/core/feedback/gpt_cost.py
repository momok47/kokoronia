# pip install datasets tiktoken
from datasets import load_dataset
import tiktoken

# ===== 設定 =====
ds = load_dataset("UEC-InabaLab/KokoroChat", split="train")

# 現在のコード設定
CURRENT_MAX_SAMPLES = 1500          # 現在使用するサンプル数
CURRENT_BATCH_SPLIT_SIZE = 300      # バッチ分割サイズ
CURRENT_EPOCHS = 70                 # エポック数
CURRENT_BATCH_SIZE = 16             # 学習用バッチサイズ
CURRENT_MAX_TOKENS = 12000          # 最大トークン数制限
CURRENT_MAX_MESSAGES = 250          # 最大メッセージ数制限

USD_TO_JPY = 146.82
TRAIN_USD_PER_M = 0.40     # 学習: $0.40 / 1M tok (推論入力価格を使用)
IN_USD_PER_M = 0.40        # 推論 入力: $0.40 / 1M tok
OUT_USD_PER_M = 1.60       # 推論 出力: $1.60 / 1M tok

# ===== トークナイザー =====
try:
    enc = tiktoken.encoding_for_model("gpt-4o-mini")  # 現在使用するモデル
except Exception:
    enc = tiktoken.get_encoding("cl100k_base")

# ===== 現在の設定でのコスト計算 =====
def count_tokens_with_limits(batch):
    """制限付きでトークンをカウント"""
    total_tokens_per_sample = []
    for dialogue in batch["dialogue"]:
        sample_tokens = 0
        message_count = 0
        
        for turn in dialogue:
            if "utterance" in turn and isinstance(turn["utterance"], str):
                message_count += 1
                if message_count > CURRENT_MAX_MESSAGES:
                    break
                    
                utterance_tokens = len(enc.encode(turn["utterance"]))
                if sample_tokens + utterance_tokens > CURRENT_MAX_TOKENS:
                    break
                    
                sample_tokens += utterance_tokens
        
        total_tokens_per_sample.append(sample_tokens)
    return {"n_tokens": total_tokens_per_sample}

# 制限付きでトークンカウント
tok_ds = ds.map(count_tokens_with_limits, batched=True, batch_size=1000, desc="制限付きトークンカウント")

# 現在の設定でのサンプル数制限
limited_ds = tok_ds.select(range(min(CURRENT_MAX_SAMPLES, len(tok_ds))))
limited_tokens = int(sum(limited_ds["n_tokens"]))
limited_mtok = limited_tokens / 1_000_000

# 元のデータセット全体
total_tokens = int(sum(tok_ds["n_tokens"]))
total_mtok = total_tokens / 1_000_000

# ===== 円換算 =====
train_jpy_per_m = TRAIN_USD_PER_M * USD_TO_JPY
in_jpy_per_m = IN_USD_PER_M * USD_TO_JPY
out_jpy_per_m = OUT_USD_PER_M * USD_TO_JPY

def yen(x): return int(round(x))

print(f"===== 現在のコード設定でのコスト計算 =====")
print(f"設定:")
print(f"  - 最大サンプル数: {CURRENT_MAX_SAMPLES:,}")
print(f"  - バッチ分割サイズ: {CURRENT_BATCH_SPLIT_SIZE:,}")
print(f"  - エポック数: {CURRENT_EPOCHS}")
print(f"  - 学習用バッチサイズ: {CURRENT_BATCH_SIZE}")
print(f"  - 最大トークン数: {CURRENT_MAX_TOKENS:,}")
print(f"  - 最大メッセージ数: {CURRENT_MAX_MESSAGES:,}")
print(f"")

print(f"===== データセット情報 =====")
print(f"元データセット: KokoroChat ({len(ds):,}件の会話データ)")
print(f"制限適用後: {len(limited_ds):,}件の会話データ")
print(f"使用率: {len(limited_ds)/len(ds)*100:.1f}%")
print(f"")

print(f"===== トークン数情報 =====")
print(f"元データセット総トークン数: {total_tokens:,} tokens ({total_mtok:.2f}M)")
print(f"制限適用後総トークン数: {limited_tokens:,} tokens ({limited_mtok:.2f}M)")
print(f"トークン数使用率: {limited_tokens/total_tokens*100:.1f}%")
print(f"")

# ===== バッチ分割でのコスト計算 =====
batches = []
for i in range(0, len(limited_ds), CURRENT_BATCH_SPLIT_SIZE):
    end_idx = min(i + CURRENT_BATCH_SPLIT_SIZE, len(limited_ds))
    batch_samples = limited_ds.select(range(i, end_idx))
    batch_tokens = int(sum(batch_samples["n_tokens"]))
    batch_mtok = batch_tokens / 1_000_000
    
    batches.append({
        'batch_id': len(batches) + 1,
        'samples': len(batch_samples),
        'tokens': batch_tokens,
        'mtok': batch_mtok
    })

print(f"===== バッチ分割情報 =====")
print(f"総バッチ数: {len(batches)}")
for batch in batches:
    print(f"バッチ {batch['batch_id']}: {batch['samples']:,}サンプル, {batch['tokens']:,}トークン ({batch['mtok']:.2f}M)")

# ===== 学習コスト計算 =====
print(f"\n===== 学習コスト計算 =====")
print(f"各バッチの学習コスト（{CURRENT_EPOCHS}エポック）:")

total_training_cost = 0
for batch in batches:
    batch_cost = yen(batch['mtok'] * train_jpy_per_m * CURRENT_EPOCHS)
    total_training_cost += batch_cost
    print(f"バッチ {batch['batch_id']}: {batch_cost:,}円")

print(f"総学習コスト: {total_training_cost:,}円")

# ===== 推論コスト計算 =====
print(f"\n===== 推論コスト計算 =====")
inference_input_ratio = 1.0   # 入力トークン比率
inference_output_ratio = 0.15 # 出力トークン比率（入力の15%と仮定）

inference_input_cost = yen(limited_mtok * in_jpy_per_m * inference_input_ratio)
inference_output_cost = yen(limited_mtok * out_jpy_per_m * inference_output_ratio)
total_inference_cost = inference_input_cost + inference_output_cost

print(f"推論コスト参考値:")
print(f"  入力: {inference_input_cost:,}円")
print(f"  出力: {inference_output_cost:,}円")
print(f"  合計: {total_inference_cost:,}円")

# ===== 総コスト計算 =====
total_cost = total_training_cost + total_inference_cost
print(f"\n===== 総コスト =====")
print(f"学習コスト: {total_training_cost:,}円")
print(f"推論コスト: {total_inference_cost:,}円")
print(f"総コスト: {total_cost:,}円")

# ===== 従来の方法との比較 =====
print(f"\n===== 従来の方法との比較 =====")
print(f"従来（全データ使用）:")
print(f"  学習コスト（{CURRENT_EPOCHS}エポック）: {yen(total_mtok * train_jpy_per_m * CURRENT_EPOCHS):,}円")
print(f"  推論コスト: {yen(total_mtok * (in_jpy_per_m + out_jpy_per_m * inference_output_ratio)):,}円")
print(f"  総コスト: {yen(total_mtok * (train_jpy_per_m * CURRENT_EPOCHS + in_jpy_per_m + out_jpy_per_m * inference_output_ratio)):,}円")

print(f"\n現在の方法（制限適用）:")
print(f"  学習コスト（{CURRENT_EPOCHS}エポック）: {total_training_cost:,}円")
print(f"  推論コスト: {total_inference_cost:,}円")
print(f"  総コスト: {total_cost:,}円")

cost_reduction = (1 - total_cost / (total_mtok * (train_jpy_per_m * CURRENT_EPOCHS + in_jpy_per_m + out_jpy_per_m * inference_output_ratio))) * 100
print(f"\nコスト削減率: {cost_reduction:.1f}%")
