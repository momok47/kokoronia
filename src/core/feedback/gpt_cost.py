# pip install datasets tiktoken
from datasets import load_dataset
import tiktoken

# ===== 設定 =====
ds = load_dataset("UEC-InabaLab/KokoroChat", split="train")


USD_TO_JPY = 146.82
TRAIN_USD_PER_M = 0.40     # 学習: $0.40 / 1M tok (推論入力価格を使用)
IN_USD_PER_M = 0.40        # 推論 入力: $0.40 / 1M tok
OUT_USD_PER_M = 1.60       # 推論 出力: $1.60 / 1M tok

# ===== トークナイザー =====
try:
    enc = tiktoken.encoding_for_model("gpt-4.1-mini")
except Exception:
    enc = tiktoken.get_encoding("cl100k_base")

# ===== 時刻は使わない。dialogue内のutteranceだけ数える =====
def count_tokens(batch):
    total_tokens_per_sample = []
    for dialogue in batch["dialogue"]:
        sample_tokens = 0
        for turn in dialogue:
            if "utterance" in turn and isinstance(turn["utterance"], str):
                sample_tokens += len(enc.encode(turn["utterance"]))
        total_tokens_per_sample.append(sample_tokens)
    return {"n_tokens": total_tokens_per_sample}

tok_ds = ds.map(count_tokens, batched=True, batch_size=1000, desc="Tokenizing utterances")
total_tokens = int(sum(tok_ds["n_tokens"]))
mtok = total_tokens / 1_000_000

# ===== 円換算 =====
train_jpy_per_m = TRAIN_USD_PER_M * USD_TO_JPY
in_jpy_per_m = IN_USD_PER_M * USD_TO_JPY
out_jpy_per_m = OUT_USD_PER_M * USD_TO_JPY

def yen(x): return int(round(x))

print(f"===== KokoroNiaデータセット全体のコスト計算 =====")
print(f"データセット: KokoroChat ({len(ds):,}件の会話データ)")
print(f"総トークン数（時刻無視）: {total_tokens:,} tokens ({mtok:.2f}M)")
print(f"※これは全{len(ds):,}件の会話データ全体を入力として利用した場合のトークン数です")
print(f"")
print(f"学習コスト: 1ep={yen(mtok*train_jpy_per_m):,}円  2ep={yen(mtok*train_jpy_per_m*2):,}円  "
      f"3ep={yen(mtok*train_jpy_per_m*3):,}円  4ep={yen(mtok*train_jpy_per_m*4):,}円")
print(f"推論単価: 入力 {yen(in_jpy_per_m):,}円/100万tok, 出力 {yen(out_jpy_per_m):,}円/100万tok")

# ===== エポック数10、バッチサイズ128の場合のAPIコスト計算 =====
epochs = 10
batch_size = 128

# 学習コスト（エポック数10）
training_cost_10ep = yen(mtok * train_jpy_per_m * epochs)

# データセットサイズを推定（総サンプル数）
# バッチサイズ128でのバッチ数を計算するため、データセット長を取得
dataset_size = len(ds)
batches_per_epoch = (dataset_size + batch_size - 1) // batch_size  # 切り上げ
total_batches = batches_per_epoch * epochs

print(f"\n===== エポック数10、バッチサイズ128の場合 =====")
print(f"データセットサイズ: {dataset_size:,} サンプル")
print(f"バッチサイズ: {batch_size}")
print(f"1エポックあたりのバッチ数: {batches_per_epoch:,}")
print(f"総バッチ数（{epochs}エポック）: {total_batches:,}")
print(f"学習コスト（{epochs}エポック）: {training_cost_10ep:,}円")

# 推論コストの参考値（入力・出力の比率を仮定）
# 一般的に出力は入力の10-20%程度と仮定
inference_input_ratio = 1.0   # 入力トークン比率
inference_output_ratio = 0.15 # 出力トークン比率（入力の15%と仮定）

inference_input_cost = yen(mtok * in_jpy_per_m * inference_input_ratio)
inference_output_cost = yen(mtok * out_jpy_per_m * inference_output_ratio)
total_inference_cost = inference_input_cost + inference_output_cost

print(f"推論コスト参考値（入力:{inference_input_cost:,}円 + 出力:{inference_output_cost:,}円）: {total_inference_cost:,}円")

# ===== 総コスト計算 =====
total_cost = training_cost_10ep + total_inference_cost
print(f"\n===== 総コスト =====")
print(f"学習コスト（10エポック）: {training_cost_10ep:,}円")
print(f"推論コスト: {total_inference_cost:,}円")
print(f"総コスト: {total_cost:,}円")
