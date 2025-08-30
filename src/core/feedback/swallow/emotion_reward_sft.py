# -*- coding: utf-8 -*-
# emotion_reward_sft.py

import torch
import logging
import os
import sys
import json
import threading
import time
import psutil
from datetime import datetime
from tqdm import tqdm

# リモート接続のタイムアウト設定
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '3600'  # 1時間
os.environ['REQUESTS_TIMEOUT'] = '3600'  # 1時間
os.environ['HF_HUB_OFFLINE'] = '0'  # オフラインモードを無効化
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    pipeline
)
# from dataclasses import dataclass  # 通常のクラスを使用するためコメントアウト
try:
    from typing import Any, Dict, List, Union
except ImportError:
    # Python 3.5未満の場合のフォールバック
    Any = object
    Dict = dict
    List = list
    Union = object
from peft import LoraConfig, get_peft_model
import torch.nn as nn
import torch.nn.functional as F

# --- 他の自作モジュールからインポート ---
from data_processing import load_and_split_dataset, EVALUATION_ITEMS
from turn_segmentation import create_turn_list
# from llm_evaluation import evaluate_conversation_on_items  # 実際の正解ラベル使用のため不要

# --- ログ設定 ---
def setup_logging(log_dir="./logs_sft"):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, "sft_training_{}.log".format(timestamp))

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file, encoding='utf-8'), logging.StreamHandler(sys.stdout)]
    )
    logger = logging.getLogger(__name__)
    logger.info("ログは {} に記録されます。".format(log_file))
    return logger

logger = setup_logging()

# --- 実験管理ツール ---
try:
    from experiment_tracker import ExperimentTracker, create_experiment_tracker
    from experiment_config import ExperimentConfig
    EXPERIMENT_TRACKING_AVAILABLE = True
except ImportError as e:
    logger.warning("実験管理ツールが利用できません: {}".format(e))
    EXPERIMENT_TRACKING_AVAILABLE = False

# --- メモリ監視システム ---
class MemoryGuard:
    def __init__(self, threshold=90, check_interval=30):
        self.threshold = threshold
        self.check_interval = check_interval
        self.monitoring = False
        self.monitor_thread = None
        self.log_file = "memory_usage_{}.csv".format(datetime.now().strftime("%Y%m%d_%H%M%S"))
        self._init_log_file()
    
    def _init_log_file(self):
        """ログファイルの初期化"""
        try:
            with open(self.log_file, 'w') as f:
                f.write("timestamp,memory_usage_percent,memory_used_gb,memory_available_gb,swap_usage_percent\n")
            logger.info("📊 メモリ使用量ログファイル作成: {}".format(self.log_file))
        except Exception as e:
            logger.warning("ログファイル作成エラー: {}".format(e))
    
    def _log_usage(self, memory_percent, memory_used_gb, memory_available_gb, swap_percent):
        """使用量をログファイルに記録"""
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(self.log_file, 'a') as f:
                f.write("{},{:.1f},{:.1f},{:.1f},{:.1f}\n".format(
                    timestamp, memory_percent, memory_used_gb, memory_available_gb, swap_percent))
        except Exception as e:
            logger.warning("ログ記録エラー: {}".format(e))
        
    def get_memory_usage(self):
        """メモリ使用率を取得"""
        try:
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            # ログに記録
            self._log_usage(
                memory.percent,
                memory.used / (1024**3),  # GB
                memory.available / (1024**3),  # GB
                swap.percent
            )
            
            return memory.percent
        except Exception as e:
            logger.warning("メモリ使用率取得エラー: {}".format(e))
            return 0
    
    def emergency_stop(self):
        """緊急停止処理"""
        logger.error("🚨 メモリ使用率{}%超過！緊急停止を実行します".format(self.threshold))
        
        # 現在のプロセスを停止
        current_process = psutil.Process()
        logger.error("🛑 プロセス停止: PID {}".format(current_process.pid))
        
        # 強制終了
        os._exit(1)
    
    def monitor_memory(self):
        """メモリ監視ループ"""
        logger.info("🛡️ メモリ監視開始（閾値: {}%）".format(self.threshold))
        
        while self.monitoring:
            try:
                usage = self.get_memory_usage()
                
                if usage >= self.threshold:
                    self.emergency_stop()
                elif usage >= self.threshold - 5:
                    logger.warning("⚠️ メモリ使用率警告: {:.1f}%".format(usage))
                elif usage >= self.threshold - 10:
                    logger.info("📊 メモリ使用率注意: {:.1f}%".format(usage))
                
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error("メモリ監視エラー: {}".format(e))
                time.sleep(self.check_interval)
    
    def start_monitoring(self):
        """監視開始"""
        if not self.monitoring:
            self.monitoring = True
            self.monitor_thread = threading.Thread(target=self.monitor_memory, daemon=True)
            self.monitor_thread.start()
            logger.info("✅ メモリ監視システム開始")
    
    def stop_monitoring(self):
        """監視停止"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("🔚 メモリ監視システム停止")

# グローバルメモリガード
memory_guard = MemoryGuard(threshold=90, check_interval=30)

# --- トークナイザー読み込み用ヘルパー関数 ---
def load_tokenizer_with_fallback(model_name, force_swallow=False):
    """
    トークナイザーを複数の方法で試行して読み込む
    force_swallow=Trueの場合、Swallowモデル以外のフォールバックを無効化
    """
    if force_swallow and "Swallow" not in model_name:
        raise ValueError("Swallowモデル強制使用モードでは、Swallowモデル以外は使用できません")
    
    # Swallowモデル専用の読み込み方法
    swallow_methods = [
        # 方法1: use_fast=False
        lambda: AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False),
        # 方法2: LlamaTokenizer直接指定（Swallowはllamaベース）
        lambda: __import__('transformers', fromlist=['LlamaTokenizer']).LlamaTokenizer.from_pretrained(model_name, trust_remote_code=True),
        # 方法3: use_fast=True（最後の手段）
        lambda: AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=True),
    ]
    
    # 通常の方法（フォールバック付き）
    normal_methods = [
        # 方法1: use_fast=False
        lambda: AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False),
        # 方法2: LlamaTokenizer直接指定
        lambda: __import__('transformers', fromlist=['LlamaTokenizer']).LlamaTokenizer.from_pretrained(model_name, trust_remote_code=True),
        # 方法3: 代替モデル
        lambda: AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium"),
        # 方法4: GPT-2トークナイザー
        lambda: AutoTokenizer.from_pretrained("gpt2"),
    ]
    
    if force_swallow:
        methods = swallow_methods
        model_names = [model_name, model_name, model_name]
        logger.info("🦅 Swallowモデル強制モード: フォールバック無効")
    else:
        methods = normal_methods
        model_names = [model_name, model_name, "microsoft/DialoGPT-medium", "gpt2"]
    
    for i, (method, fallback_name) in enumerate(zip(methods, model_names)):
        try:
            tokenizer = method()
            logger.info("✅ トークナイザー読み込み成功 (方法{}): {}".format(i+1, fallback_name))
            return tokenizer, fallback_name
        except Exception as e:
            logger.warning("❌ トークナイザー読み込み失敗 (方法{}): {}".format(i+1, e))
            continue
    
    if force_swallow:
        raise RuntimeError("🚨 Swallowモデルのトークナイザー読み込みが全て失敗しました。モデルが正しくダウンロードされているか確認してください。")
    else:
        raise RuntimeError("全てのトークナイザー読み込み方法が失敗しました")

def setup_tokenizer_padding(tokenizer):
    """
    トークナイザーのパディング設定
    """
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    
    tokenizer.padding_side = "right"
    return tokenizer

# --- 回帰タスク用データコレーター ---
class RegressionDataCollator:
    """
    回帰タスク用のデータコレーター。
    数値ラベルを適切に処理する。
    """
    def __init__(self, tokenizer=None, padding=True, max_length=None, 
                 pad_to_multiple_of=None, return_tensors="pt"):
        self.tokenizer = tokenizer
        self.padding = padding
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.return_tensors = return_tensors

    def __call__(self, features):
        # input_idsとattention_maskをパディング
        batch = {}
        
        # input_idsの処理
        input_ids = [f["input_ids"] for f in features]
        if self.padding:
            max_len = max(len(ids) for ids in input_ids)
            input_ids = [ids + [self.tokenizer.pad_token_id] * (max_len - len(ids)) for ids in input_ids]
        
        # attention_maskの処理
        attention_mask = [f["attention_mask"] for f in features]
        if self.padding:
            attention_mask = [mask + [0] * (max_len - len(mask)) for mask in attention_mask]
        
        # ラベル（数値）の処理
        labels = [f["labels"] for f in features]
        
        batch["input_ids"] = torch.tensor(input_ids, dtype=torch.long)
        batch["attention_mask"] = torch.tensor(attention_mask, dtype=torch.long)
        batch["labels"] = torch.tensor(labels, dtype=torch.float)
        
        return batch

# --- カスタムTrainerクラス（MSE損失用） ---
class RegressionTrainer(Trainer):
    """
    回帰タスク用のカスタムTrainer。
    0～5点の評価スコア予測にMSE損失を使用。
    実験管理機能付き。
    """
    
    def __init__(self, experiment_tracker=None, *args, **kwargs):
        super(RegressionTrainer, self).__init__(*args, **kwargs)
        self.experiment_tracker = experiment_tracker
        self.step_count = 0
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        MSE損失を計算する。
        """
        labels = inputs.pop("labels")
        
        # モデルの出力を取得
        outputs = model(**inputs)
        logits = outputs.logits
        
        # 最後のトークンの隠れ状態を取得
        # logitsの形状: [batch_size, sequence_length, vocab_size]
        last_hidden_states = logits[:, -1, :]  # [batch_size, vocab_size]
        
        # 回帰用のヘッドを追加（vocab_sizeから1次元へ）
        if not hasattr(model, 'regression_head'):
            model.regression_head = nn.Linear(logits.size(-1), 1).to(logits.device)
            # 回帰ヘッドをモデルのモジュールとして登録（保存時に含まれるようにする）
            if hasattr(model, 'add_module'):
                model.add_module('regression_head', model.regression_head)
        
        # 回帰予測値を計算
        predictions = model.regression_head(last_hidden_states).squeeze(-1)  # [batch_size]
        
        # MSE損失を計算
        loss = F.mse_loss(predictions, labels.float())
        
        # 実験管理ツールにメトリクスを記録
        if self.experiment_tracker:
            try:
                metrics = {
                    "train_loss": loss.item(),
                    "mse_loss": loss.item(),
                    "step": self.step_count
                }
                self.experiment_tracker.log_metrics(metrics, step=self.step_count)
                self.step_count += 1
            except Exception as e:
                logger.warning("実験管理メトリクス記録エラー: {}".format(e))
        
        return (loss, {"logits": predictions}) if return_outputs else loss

# --- データ準備 ---
def create_regression_dataset_from_real_labels(original_dataset, max_samples, tokenizer):
    """
    実際の正解ラベルを使用して回帰タスク用のデータセットを作成する。
    MSE損失を使用するため、入力テキストと数値ラベルのペアを作成。
    """
    regression_data = []

    # 処理時間を短縮するために対象件数を制限
    if len(original_dataset) > max_samples:
        logger.info("{}件にサンプリングして処理します。".format(max_samples))
        dataset_to_process = original_dataset.shuffle(seed=42).select(range(max_samples))
    else:
        dataset_to_process = original_dataset

    for i, example in enumerate(tqdm(dataset_to_process, desc="回帰データ生成中（実際の正解ラベル使用）")):
        dialogue = example.get('dialogue', [])
        review_jp = example.get('review_by_client_jp', {})
        
        # 対話データの処理
        if not isinstance(dialogue, list):
            logger.warning("サンプル {}: 対話データが正しい形式ではないためスキップ".format(i))
            continue

        # 対話を「役割: 発言」のテキストリストに変換
        try:
            turn_list = create_turn_list(dialogue)
            if not turn_list:
                logger.warning("サンプル {}: turn_listが空のためスキップ".format(i))
                continue
            full_conversation_text = "\n".join(turn_list)
            # 対話テキストを短縮
            short_conversation = full_conversation_text if len(full_conversation_text) <= 800 else full_conversation_text[:800] + "..."
            logger.info("サンプル {}: 対話テキスト長 = {}".format(i, len(short_conversation)))
        except Exception as e:
            logger.error("サンプル {}: turn_list作成でエラー: {}".format(i, e))
            continue

        # 実際の評価スコアを取得
        if not review_jp:
            logger.warning("サンプル {}: 評価データが見つからないためスキップ".format(i))
            continue

        # 各評価項目について学習データを作成
        for item in EVALUATION_ITEMS:
            if item not in review_jp:
                logger.warning("サンプル {}: 評価項目 '{}' が見つからないためスキップ".format(i, item))
                continue
            
            # 実際のスコア（0-5点）を取得
            actual_score = review_jp[item]
            if not isinstance(actual_score, int) or actual_score < 0 or actual_score > 5:
                logger.warning("サンプル {}, 項目 '{}': 無効なスコア {} をスキップ".format(i, item, actual_score))
                continue
            
            # プロンプト（入力）の定義
            input_text = """### 指示
以下の対話について「{}」の満足度を0～5点で評価してください。

### 対話
{}

### 回答
""".format(item, short_conversation)
            
            # トークナイズ
            inputs = tokenizer(
                input_text,
                truncation=True,
                padding=False,
                max_length=256,  # M4最適化: メモリ削減のため大幅短縮
                return_tensors=None
            )
            
            # 回帰データを追加
            regression_data.append({
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
                "labels": float(actual_score)  # 数値ラベル
            })

    logger.info("最終的に生成された回帰データ数: {}".format(len(regression_data)))
    return Dataset.from_list(regression_data)

def get_regression_dataset(
    tokenizer,
    use_cache=True,
    cache_path="./regression_dataset_real_labels.jsonl",
    max_samples=100
):
    """
    キャッシュが存在すればそれを読み込み、なければ実際の正解ラベルから回帰データセットを生成する。
    """
    if use_cache and os.path.exists(cache_path):
        logger.info("キャッシュされた回帰データセット {} を読み込みます。".format(cache_path))
        return load_dataset("json", data_files=cache_path, split="train")

    logger.info("キャッシュが見つからないため、実際の正解ラベルから回帰データセットを新たに生成します。")
    train_ds, _, _ = load_and_split_dataset()
    
    # デバッグ: 最初のサンプルの構造を詳細に確認
    if len(train_ds) > 0:
        sample = train_ds[0]
        logger.info("=== サンプルデータの構造確認 ===")
        logger.info("サンプルのkeys: {}".format(list(sample.keys())))
        if 'review_by_client_jp' in sample:
            review = sample['review_by_client_jp']
            logger.info("review_by_client_jp の型: {}".format(type(review)))
            logger.info("利用可能な評価項目: {}".format([k for k in review.keys() if k in EVALUATION_ITEMS]))
    
    regression_dataset = create_regression_dataset_from_real_labels(train_ds, max_samples=max_samples, tokenizer=tokenizer)

    # 次回以降のためにキャッシュを保存
    regression_dataset.to_json(cache_path)
    logger.info("生成したデータセットを {} に保存しました。".format(cache_path))

    return regression_dataset

# --- メインの学習処理 ---
def main():
    # --- 0. メモリ監視開始 ---
    memory_guard.start_monitoring()
    
    # --- 1. 設定 ---
    # ★★★★★★★★★★ Swallow-7b-instruct-hf 専用設定 ★★★★★★★★★★
    # 絶対にSwallowモデルを使用（フォールバック無効）
    sft_model_name = "tokyotech-llm/Swallow-7b-instruct-hf"
    generator_model_name = "tokyotech-llm/Swallow-7b-instruct-hf"
    
    # Swallowモデル強制使用フラグ
    FORCE_SWALLOW_MODEL = True
    logger.info("🦅 Swallow-7b-instruct-hfモデルの強制使用が有効です")

    # 学習済みアダプタの保存先
    output_dir = "./swallow_emotion_reward_adapter"
    # データ生成をスキップしてキャッシュを使うか
    USE_CACHE = True  # テスト実行用にキャッシュを有効化（処理時間短縮）
    # M4最適化: サンプル数を大幅削減（メモリ効率最優先）
    MAX_SAMPLES_FOR_DATA_GENERATION = 10  # M4メモリ効率用に大幅削減（20→10）
    
    # 実験管理の設定
    EXPERIMENT_TRACKING_TOOL = "both"  # "tensorboard", "wandb", "both", "none"

    # --- 1.5. 実験管理の初期化 ---
    experiment_tracker = None
    try:
        if EXPERIMENT_TRACKING_AVAILABLE and EXPERIMENT_TRACKING_TOOL != "none":
            experiment_tracker = create_experiment_tracker(
                tracking_tool=EXPERIMENT_TRACKING_TOOL,
                project_name="emotion_reward_sft"
            )
            logger.info("実験管理ツール '{}' を初期化しました".format(EXPERIMENT_TRACKING_TOOL))
    except (NameError, Exception) as e:
        logger.warning("実験管理ツール初期化をスキップ: {}".format(e))
        experiment_tracker = None
    
    # --- 2. トークナイザーの準備 ---
    logger.info("🦅 Swallowトークナイザー '{}' を読み込みます...".format(sft_model_name))
    
    # Swallowモデル強制使用でトークナイザーを読み込み
    tokenizer, actual_model_name = load_tokenizer_with_fallback(sft_model_name, force_swallow=FORCE_SWALLOW_MODEL)
    tokenizer = setup_tokenizer_padding(tokenizer)
    
    # Swallow強制モードではモデル名変更を許可しない
    if FORCE_SWALLOW_MODEL and actual_model_name != sft_model_name:
        raise RuntimeError("🚨 Swallow強制モードでモデル名が変更されました: {} -> {}".format(sft_model_name, actual_model_name))
    elif actual_model_name != sft_model_name:
        logger.info("モデル名を '{}' から '{}' に変更しました".format(sft_model_name, actual_model_name))
        sft_model_name = actual_model_name
    
    logger.info("トークナイザー設定完了: pad_token='{}'".format(tokenizer.pad_token))

    # --- 3. 回帰用データセットの準備（実際の正解ラベル使用） ---
    logger.info("実際の正解ラベルを使用して回帰データセットを準備します...")
    regression_dataset = get_regression_dataset(
        tokenizer=tokenizer,
        use_cache=USE_CACHE,
        max_samples=MAX_SAMPLES_FOR_DATA_GENERATION
    )
    logger.info("回帰用データセットを準備完了。サンプル数: {}".format(len(regression_dataset)))
    
    # サンプルデータを確認
    if len(regression_dataset) > 0:
        logger.info("=== 回帰データセットサンプル ===")
        sample = regression_dataset[0]
        logger.info("入力トークン数: {}".format(len(sample['input_ids'])))
        logger.info("ラベル: {}".format(sample['labels']))

    # --- 4. 回帰対象モデルの準備 (CPU版) ---
    logger.info("🦅 Swallow回帰対象モデル '{}' をCPU環境用に読み込みます...".format(sft_model_name))

    # M4 MacBook Air最適化: メモリ制限のためCPUモードを使用
    device = "cpu"
    logger.info("🚀 M4最適化: 10コアCPU並列処理モード（メモリ効率重視）")
    logger.info("Device set to use {}".format(device))

    # Swallowモデルの読み込み（CPU用、再試行機能付き）
    logger.info("🦅 Swallowモデル '{}' を読み込みます...".format(sft_model_name))
    max_retries = 3
    retry_delay = 60  # 60秒待機
    
    try:
        model = None
        for attempt in range(max_retries):
            try:
                logger.info("モデル読み込み試行 {}/{}".format(attempt + 1, max_retries))
                # M4最適化: MPSでは量子化なし（Macでは非対応）
                if device == "mps":
                    # MPSでは量子化なしでメモリ効率重視
                    model = AutoModelForCausalLM.from_pretrained(
                        sft_model_name,
                        torch_dtype=torch.float32,
                        device_map=None,
                        trust_remote_code=True,
                        resume_download=True,
                        low_cpu_mem_usage=True,  # メモリ効率重視
                    )
                else:
                    # CPU/CUDAの場合（CPUでは量子化なし）
                    if device == "cpu":
                        # CPUモードでは量子化なし（メモリ制限があるが安定）
                        model = AutoModelForCausalLM.from_pretrained(
                            sft_model_name,
                            torch_dtype=torch.float32,
                            device_map=None,
                            trust_remote_code=True,
                            resume_download=True,
                            low_cpu_mem_usage=True,  # メモリ効率重視
                        )
                    else:
                        # CUDAの場合は8bit量子化
                        model = AutoModelForCausalLM.from_pretrained(
                            sft_model_name,
                            torch_dtype=torch.float16,
                            device_map=None,
                            trust_remote_code=True,
                            resume_download=True,
                            load_in_8bit=True,
                            llm_int8_enable_fp32_cpu_offload=True,
                        )
                logger.info("✅ Swallowモデル読み込み成功")
                break
            except Exception as retry_error:
                logger.warning("モデル読み込み試行 {}/{} 失敗: {}".format(attempt + 1, max_retries, retry_error))
                if attempt < max_retries - 1:
                    logger.info("{}秒後に再試行します...".format(retry_delay))
                    time.sleep(retry_delay)
                else:
                    # 最後の試行も失敗した場合、外側のexceptに進む
                    raise retry_error
        
        # Swallowモデル情報の出力
        logger.info("🦅 Swallowモデル情報:")
        logger.info("  モデル名: {}".format(sft_model_name))
        logger.info("  モデルタイプ: {}".format(type(model).__name__))
        if hasattr(model, 'config'):
            logger.info("  アーキテクチャ: {}".format(getattr(model.config, 'architectures', 'Unknown')))
            logger.info("  モデルタイプ: {}".format(getattr(model.config, 'model_type', 'Unknown')))
        
    except Exception as e:
        if FORCE_SWALLOW_MODEL:
            logger.error("🚨 Swallow強制モード: モデル読み込み失敗")
            logger.error("エラー詳細: {}".format(e))
            logger.error("解決方法:")
            logger.error("1. インターネット接続を確認してください")
            logger.error("2. Hugging Face Hubからモデルがダウンロード可能か確認してください")
            logger.error("3. 十分なディスク容量があるか確認してください（約14GB必要）")
            logger.error("4. transformersライブラリが最新版か確認してください")
            raise RuntimeError("🚨 Swallow強制モードでモデル読み込みに失敗しました")
        else:
            logger.error("モデル読み込み失敗: {}".format(e))
            logger.info("代替モデルを使用します...")
            # 代替モデルに切り替え
            if sft_model_name != "microsoft/DialoGPT-medium":
                sft_model_name = "microsoft/DialoGPT-medium"
                model = AutoModelForCausalLM.from_pretrained(
                    sft_model_name,
                    torch_dtype=torch.float32,
                    device_map=None,
                    trust_remote_code=True,
                )
                logger.info("代替モデル '{}' で読み込み成功".format(sft_model_name))
            else:
                raise e
    model = model.to(device)
    model.config.use_cache = False
    
    # M4最適化: グラディエントチェックポイントでメモリ大幅削減
    model.gradient_checkpointing_enable()
    logger.info("🚀 M4最適化: グラディエントチェックポイント有効化（メモリ50%削減）")
    
    # M4最適化: メモリ使用量最適化
    import gc
    gc.collect()  # ガベージコレクション実行
    
    # PyTorchメモリ管理設定
    if device == "cpu":
        # M4 CPU最適化設定
        torch.set_num_threads(10)  # M4の10コアを活用
        torch.set_num_interop_threads(4)  # 並列処理最適化
        logger.info("🚀 M4 CPU最適化: スレッド数=10, 並列スレッド数=4")
    elif device == "mps":
        # MPS設定（使用しないが念のため）
        torch.mps.set_per_process_memory_fraction(0.6)
        logger.info("🚀 M4 MPS: メモリ使用量を60%に設定")
    elif device == "cuda":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        logger.info("🔥 CUDA最適化設定")

    # モデルの構造を調べてターゲットモジュールを自動検出
    def find_target_modules(model):
        """モデルの線形層を自動検出してターゲットモジュールを決定"""
        target_modules = set()
        all_modules = {}
        
        # 全てのモジュールを調査
        for name, module in model.named_modules():
            module_type = type(module).__name__
            all_modules[name] = module_type
            
            if isinstance(module, torch.nn.Linear):
                # モジュール名の最後の部分を取得
                module_name = name.split('.')[-1]
                target_modules.add(module_name)
                logger.debug("線形層発見: {} ({}次元 -> {}次元)".format(
                    name, module.in_features, module.out_features))
        
        # デバッグ: 最初の数層のモジュール構造を表示
        logger.info("=== モデル構造の詳細 ===")
        layer_count = 0
        for name, module_type in all_modules.items():
            if layer_count < 20:  # 最初の20層のみ表示
                logger.info("  {}: {}".format(name, module_type))
                layer_count += 1
            elif layer_count == 20:
                logger.info("  ... (以下省略)")
                break
        
        # Swallowモデル（LLaMAベース）専用ターゲットモジュール
        if FORCE_SWALLOW_MODEL:
            # Swallow/LLaMAアーキテクチャ専用（優先順）
            common_targets = [
                # LLaMA/Swallow系（最優先）
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
                # 追加のTransformer系
                "self_attn", "mlp"
            ]
            logger.info("🦅 Swallow専用ターゲットモジュール検出モード")
        else:
            # 一般的なターゲットモジュール名のリスト（優先順）
            common_targets = [
                # Transformer系
                "q_proj", "k_proj", "v_proj", "o_proj",
                # LLaMA系
                "gate_proj", "up_proj", "down_proj",
                # GPT系
                "c_attn", "c_proj", "c_fc",
                # BERT系
                "query", "key", "value", "dense",
                # DialoGPT/GPT-2系
                "attn", "mlp",
                # その他
                "attention", "feed_forward", "linear", "fc"
            ]
        
        # 見つかったモジュールから適切なものを選択
        selected_targets = []
        for target in common_targets:
            if target in target_modules:
                selected_targets.append(target)
        
        # 最低限のターゲットが見つからない場合
        if len(selected_targets) < 1:
            # より柔軟なマッチングを試行
            flexible_targets = []
            for module_name in target_modules:
                if any(keyword in module_name.lower() for keyword in 
                      ['proj', 'attn', 'mlp', 'dense', 'linear', 'fc']):
                    flexible_targets.append(module_name)
            
            if flexible_targets:
                selected_targets = flexible_targets[:4]  # 最大4つまで
            else:
                # 最後の手段: 全ての線形層から最初の数個を選択
                selected_targets = list(target_modules)[:4]
        
        if FORCE_SWALLOW_MODEL:
            logger.info("🦅 Swallow検出されたターゲットモジュール: {}".format(selected_targets))
            logger.info("🦅 Swallow利用可能な全線形層: {}".format(sorted(target_modules)))
        else:
            logger.info("検出されたターゲットモジュール: {}".format(selected_targets))
            logger.info("利用可能な全線形層: {}".format(sorted(target_modules)))
        
        return selected_targets
    
    # ターゲットモジュールの自動検出
    target_modules = find_target_modules(model)
    
    if not target_modules:
        logger.error("適切なターゲットモジュールが見つかりませんでした")
        raise ValueError("LoRA用のターゲットモジュールが見つかりません")
    
    # M4最適化: LoRAの設定
    if device == "cpu":
        # M4 CPU最適化設定
        peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=32,  # CPUでも十分なrank
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]  # 主要モジュール
        )
        logger.info("🚀 M4 CPU最適化: LoRA設定（r=32, alpha=16）")
    elif device == "mps":
        # MPS軽量設定（使用しないが念のため）
        peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=16,  # より軽量
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
        )
        logger.info("🚀 M4 MPS: 軽量LoRA設定（r=16, alpha=16）")
    else:
        # 標準設定
        peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=32,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
        )
        logger.info("💻 標準LoRA設定（r=32, alpha=16）")
    
    # PEFTモデルの適用
    try:
        logger.info("PEFTモデルを適用中...")
        logger.info("LoRA設定: r={}, alpha={}, dropout={}, targets={}".format(
            peft_config.r, peft_config.lora_alpha, peft_config.lora_dropout, peft_config.target_modules))
        
        model = get_peft_model(model, peft_config)
        logger.info("✅ PEFTモデルの適用成功")
        
        # 学習可能パラメータ数の表示
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info("学習可能パラメータ: {:,} / {:,} ({:.2f}%)".format(
            trainable_params, total_params, 100 * trainable_params / total_params))
        
    except ValueError as e:
        logger.error("PEFTモデル適用失敗: {}".format(e))
        logger.info("フォールバック: LoRAなしで学習を続行します")
        
        # LoRAなしでの学習用に設定を調整
        logger.warning("⚠️ LoRAを使用せずに全パラメータを学習します（メモリ使用量が増加します）")
        
        # 学習率を下げる（全パラメータ学習の場合）
        logger.info("学習率を調整: 2e-4 -> 5e-5")
        
        # モデルはそのまま使用（LoRAなし）
        peft_config = None

    # --- 5. 回帰トレーニング設定（CPU環境用に調整） ---
    # M4最適化: 学習率の動的調整
    if peft_config is not None:
        if device == "cpu":
            # M4 CPU最適化: CPUでは安定した学習率
            learning_rate = 2e-4
            logger.info("🚀 M4 CPU最適化: 学習率 = {}".format(learning_rate))
        elif device == "mps":
            # MPS用学習率
            learning_rate = 3e-4
            logger.info("🚀 M4 MPS最適化: 学習率 = {}".format(learning_rate))
        else:
            learning_rate = 2e-4  # 標準LoRA学習率
            logger.info("LoRA使用: 学習率 = {}".format(learning_rate))
    else:
        learning_rate = 5e-5  # 全パラメータ学習時
        logger.info("全パラメータ学習: 学習率 = {}".format(learning_rate))
    
    # M4最適化: CPUモード用の並列処理設定
    if device == "cpu":
        # M4 10コアCPU最適化設定
        batch_size = 2  # CPUでは少し大きく
        accumulation_steps = 8  # 実効バッチサイズ16
        num_workers = 8  # M4の10コアを活用（8並列）
        pin_memory = False  # CPUではFalse
        fp16_enabled = False  # CPUではfp16無効
        logger.info("🚀 M4 CPU最適化: バッチサイズ={}, 並列数={}".format(batch_size, num_workers))
    elif device == "mps":
        # MPS設定（使用しないが念のため）
        batch_size = 1
        accumulation_steps = 16
        num_workers = 0
        pin_memory = False
        fp16_enabled = False
    else:
        # CUDA設定
        batch_size = 1
        accumulation_steps = 16
        num_workers = 0
        pin_memory = False
        fp16_enabled = True
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,  # M4最適化
        gradient_accumulation_steps=accumulation_steps,  # M4最適化
        optim="adamw_torch",
        save_steps=100,
        logging_steps=10,
        learning_rate=learning_rate,
        fp16=fp16_enabled,  # M4最適化
        max_grad_norm=0.3,
        num_train_epochs=1,
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
        dataloader_pin_memory=pin_memory,  # M4最適化
        dataloader_num_workers=num_workers,  # M4最適化: 並列処理
        remove_unused_columns=False,
        # M4最適化: メモリ削減設定
        max_steps=50,  # ステップ数制限でメモリ削減
        gradient_checkpointing=True,  # グラディエントチェックポイント
        dataloader_drop_last=True,  # 不完全バッチを削除
    )

    # ハイパーパラメータの記録
    if experiment_tracker:
        hyperparams = {
            "model_name": sft_model_name,
            "max_samples": MAX_SAMPLES_FOR_DATA_GENERATION,
            "use_cache": USE_CACHE,
            "batch_size": training_args.per_device_train_batch_size,
            "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
            "learning_rate": training_args.learning_rate,
            "num_epochs": training_args.num_train_epochs,
            "device": device,
            "dataset_size": len(regression_dataset),
            "use_lora": peft_config is not None
        }
        
        # LoRA関連パラメータ（LoRA使用時のみ）
        if peft_config is not None:
            hyperparams.update({
                "lora_r": peft_config.r,
                "lora_alpha": peft_config.lora_alpha,
                "lora_dropout": peft_config.lora_dropout,
                "lora_target_modules": peft_config.target_modules
            })
        else:
            hyperparams.update({
                "training_mode": "full_parameter_training",
                "lora_r": "N/A",
                "lora_alpha": "N/A",
                "lora_dropout": "N/A"
            })
        
        experiment_tracker.log_hyperparameters(hyperparams)

    # データコレーターの準備（回帰タスク用）
    data_collator = RegressionDataCollator(tokenizer=tokenizer)

    # --- 6. カスタムTrainerの初期化と学習開始（MSE損失） ---
    trainer = RegressionTrainer(
        experiment_tracker=experiment_tracker,
        model=model,
        args=training_args,
        train_dataset=regression_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,  # tokenizerの代わりにprocessing_classを使用
    )

    logger.info("🚀 MSE損失を使用した回帰トレーニングを開始します。")
    trainer.train()

    # --- 7. 学習済みモデルの保存 ---
    final_model_path = os.path.join(output_dir, "final_model")
    trainer.save_model(final_model_path)
    logger.info("✅ 回帰トレーニング完了。モデルは {} に保存されました。".format(final_model_path))
    
    # モデルアーティファクトの保存（W&B）
    if experiment_tracker:
        try:
            experiment_tracker.log_model_artifact(final_model_path)
            experiment_tracker.log_text(
                "回帰トレーニングが正常に完了しました。MSE損失を使用した感情報酬モデルの学習。",
                "training_summary"
            )
        except Exception as e:
            logger.error("実験管理アーティファクト保存エラー: {}".format(e))
        finally:
            # 実験管理ツールの終了処理
            experiment_tracker.finish()

if __name__ == "__main__":
    main()