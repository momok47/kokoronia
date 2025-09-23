# OpenAI GPT-4o mini Supervised Fine Tuning
import os
import json
import time
import psutil
import threading
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging
from pathlib import Path

from datasets import load_dataset
import openai
from openai import OpenAI
import tiktoken
from dotenv import load_dotenv
import random

try:
    from .progress_monitor import create_progress_monitor
except ImportError:
    # フォールバック: 進行状況表示なし
    def create_progress_monitor(*args, **kwargs):
        class DummyMonitor:
            def start_monitoring(self, *args, **kwargs): pass
            def stop_monitoring(self): pass
            def start_phase(self, *args, **kwargs): pass
            def update_phase(self, *args, **kwargs): pass
            def complete_phase(self, *args): pass
            def add_log(self, *args, **kwargs): pass
        return DummyMonitor()

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# デフォルト制限設定（OpenAI制限内で最大限活用）
DEFAULT_MAX_TOKENS_PER_SAMPLE = 100000  # OpenAI制限: 128K、推奨: 100K
DEFAULT_MAX_MESSAGES_PER_DIALOGUE = 2000  # 実用的な上限
DEFAULT_TOKEN_OVERFLOW_TOLERANCE = 1.1  # 10%の超過を許容（制限が大きいため厳格に）

# バッチ分割設定
DEFAULT_BATCH_SPLIT_SIZE = 500  # 1バッチあたりのサンプル数（ファイル分割用）
DEFAULT_MAX_BATCHES = 10  # 最大バッチ数

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

class SystemMonitor:
    """システムリソース監視クラス"""
    
    def __init__(self, interval: int = 30):
        self.interval = interval
        self.monitoring = False
        self.monitor_thread = None
        
    def start_monitoring(self):
        """監視開始"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        logger.info("システム監視を開始しました")
        
    def stop_monitoring(self):
        """監視停止"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        logger.info("システム監視を停止しました")
        
    def _monitor_loop(self):
        """監視ループ"""
        while self.monitoring:
            try:
                # CPU使用率
                cpu_percent = psutil.cpu_percent(interval=1)
                
                # メモリ使用量
                memory = psutil.virtual_memory()
                memory_percent = memory.percent
                memory_used_gb = memory.used / (1024**3)
                memory_total_gb = memory.total / (1024**3)
                
                # ディスク使用量
                disk = psutil.disk_usage('/')
                disk_percent = disk.percent
                
                logger.info(f"[監視] CPU: {cpu_percent:.1f}% | "
                          f"メモリ: {memory_percent:.1f}% ({memory_used_gb:.1f}/{memory_total_gb:.1f}GB) | "
                          f"ディスク: {disk_percent:.1f}%")
                
                # アラート条件
                if cpu_percent > 90:
                    logger.warning(f"⚠️ CPU使用率が高いです: {cpu_percent:.1f}%")
                if memory_percent > 90:
                    logger.warning(f"⚠️ メモリ使用率が高いです: {memory_percent:.1f}%")
                if disk_percent > 90:
                    logger.warning(f"⚠️ ディスク使用率が高いです: {disk_percent:.1f}%")
                    
            except Exception as e:
                logger.error(f"システム監視エラー: {e}")
                
            time.sleep(self.interval)

class OpenAISFT:
    """OpenAI Supervised Fine Tuning クラス"""
    
    def __init__(self, api_key: Optional[str] = None, show_progress: bool = True):
        """
        初期化
        
        Args:
            api_key: OpenAI APIキー（環境変数OPENAI_API_KEYからも取得可能）
            show_progress: 進行状況表示を使用するか
        """
        # プロジェクトルートの.envファイルを読み込み
        project_root = Path(__file__).parent.parent.parent.parent  # src/core/feedback から4つ上
        env_path = project_root / ".env"
        if env_path.exists():
            load_dotenv(env_path)
            logger.info(f".envファイルを読み込みました: {env_path}")
        else:
            logger.info(f".envファイルが見つかりません: {env_path}")
        
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API keyが設定されていません。.envファイルにOPENAI_API_KEYを設定するか、引数で指定してください。")
            
        self.client = OpenAI(api_key=self.api_key)
        self.tokenizer = tiktoken.encoding_for_model("gpt-4o-mini")
        self.monitor = SystemMonitor()
        
        # 進行状況表示の初期化
        self.show_progress = show_progress
        self.progress_monitor = create_progress_monitor() if show_progress else None
        
        # 出力ディレクトリの作成
        self.output_dir = Path("openai_sft_outputs")
        self.output_dir.mkdir(exist_ok=True)
        
    def _validate_and_clean_data(self, dataset, max_tokens_per_sample: int = None, max_messages_per_dialogue: int = None) -> tuple:
        """
        データを検証・クリーニングして制限に引っかかるデータを削除
        
        Args:
            dataset: 元のデータセット
            max_tokens_per_sample: サンプルあたりの最大トークン数
            max_messages_per_dialogue: 対話あたりの最大メッセージ数
            
        Returns:
            (cleaned_indices, removed_count, validation_stats): クリーニング済みデータのインデックス、削除されたサンプル数、検証統計
        """
        # デフォルト値の設定
        if max_tokens_per_sample is None:
            max_tokens_per_sample = DEFAULT_MAX_TOKENS_PER_SAMPLE
        if max_messages_per_dialogue is None:
            max_messages_per_dialogue = DEFAULT_MAX_MESSAGES_PER_DIALOGUE
            
        logger.info(f"データの検証・クリーニングを開始... (制限: トークン数={max_tokens_per_sample}, メッセージ数={max_messages_per_dialogue})")
        
        cleaned_indices = []
        removed_count = 0
        validation_stats = {
            'total_samples': len(dataset),
            'removed_samples': 0,
            'token_limit_exceeded': 0,
            'message_limit_exceeded': 0,
            'invalid_format': 0,
            'empty_dialogue': 0
        }
        
        for i, example in enumerate(dataset):
            try:
                dialogue = example.get('dialogue', [])
                
                # 空の対話をチェック
                if not dialogue:
                    validation_stats['empty_dialogue'] += 1
                    removed_count += 1
                    continue
                
                # メッセージ数の制限をチェック
                if len(dialogue) > max_messages_per_dialogue:
                    validation_stats['message_limit_exceeded'] += 1
                    removed_count += 1
                    continue
                
                # トークン数の制限をチェック（OpenAI制限内で最大限活用）
                total_tokens = 0
                for message in dialogue:
                    content = message.get('content', '')
                    if content:
                        tokens = len(self.tokenizer.encode(content))
                        total_tokens += tokens
                        # 個別メッセージが極端に長い場合は警告
                        if tokens > max_tokens_per_sample * 0.5:  # 50%を超える場合
                            logger.warning(f"長いメッセージ検出: {tokens} tokens")
                
                if total_tokens > max_tokens_per_sample:
                    # OpenAI制限内での厳格なチェック
                    if total_tokens > max_tokens_per_sample * DEFAULT_TOKEN_OVERFLOW_TOLERANCE:
                        validation_stats['token_limit_exceeded'] += 1
                        removed_count += 1
                        logger.warning(f"トークン数制限超過で削除: {total_tokens} > {max_tokens_per_sample}")
                        continue
                    else:
                        logger.warning(f"トークン数制限を軽微に超過: {total_tokens} > {max_tokens_per_sample}")
                
                # OpenAI形式に変換してテスト
                try:
                    messages = self._format_dialogue_for_openai(dialogue)
                    if not messages:
                        validation_stats['invalid_format'] += 1
                        removed_count += 1
                        continue
                    
                    # 最終チェック: メッセージが正しくフォーマットされているか
                    if not all('role' in msg and 'content' in msg for msg in messages):
                        validation_stats['invalid_format'] += 1
                        removed_count += 1
                        continue
                    
                    # 有効なデータのインデックスとして追加
                    cleaned_indices.append(i)
                    
                except Exception as e:
                    validation_stats['invalid_format'] += 1
                    removed_count += 1
                    continue
                
                # 進行状況表示
                if (i + 1) % 1000 == 0:
                    logger.info(f"検証済み: {i + 1}/{len(dataset)} (削除: {removed_count})")
                    
            except Exception as e:
                logger.error(f"サンプル {i} の検証でエラー: {e}")
                validation_stats['invalid_format'] += 1
                removed_count += 1
                continue
        
        validation_stats['removed_samples'] = removed_count
        validation_stats['cleaned_samples'] = len(cleaned_indices)
        
        logger.info("=== データ検証・クリーニング結果 ===")
        logger.info(f"元サンプル数: {validation_stats['total_samples']}")
        logger.info(f"クリーニング済み: {validation_stats['cleaned_samples']}")
        logger.info(f"削除されたサンプル: {validation_stats['removed_samples']}")
        logger.info(f"削除理由:")
        logger.info(f"  - トークン数超過: {validation_stats['token_limit_exceeded']}")
        logger.info(f"  - メッセージ数超過: {validation_stats['message_limit_exceeded']}")
        logger.info(f"  - 形式不正: {validation_stats['invalid_format']}")
        logger.info(f"  - 空の対話: {validation_stats['empty_dialogue']}")
        
        return cleaned_indices, removed_count, validation_stats

    def load_and_split_dataset(self, max_samples: Optional[int] = None, seed: int = 42, 
                              max_tokens_per_sample: int = None, max_messages_per_dialogue: int = None) -> tuple:
        """
        Kokoro Chatデータセットを読み込み、制限に引っかかるデータを削除してから8:1:1に分割する
        
        Args:
            max_samples: 最大サンプル数（デバッグ用）
            seed: 再現性のための乱数シード
            max_tokens_per_sample: サンプルあたりの最大トークン数
            max_messages_per_dialogue: 対話あたりの最大メッセージ数
            
        Returns:
            (train_data, test_data, valid_data): 分割されたデータセット
        """
        logger.info("KokoroChat データセットを読み込み中...")
        dataset = load_dataset("UEC-InabaLab/KokoroChat", split="train")
        
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
            logger.info(f"デバッグ用に{max_samples}サンプルに制限")
        
        logger.info(f"元データセットサイズ: {len(dataset)}")
        
        # データの検証・クリーニング
        cleaned_indices, removed_count, validation_stats = self._validate_and_clean_data(
            dataset, max_tokens_per_sample, max_messages_per_dialogue
        )
        
        if len(cleaned_indices) == 0:
            raise ValueError("クリーニング後に有効なデータが残りませんでした。制限を緩和してください。")
        
        # クリーニング済みデータのインデックスをシャッフル
        import random
        random.seed(seed)
        shuffled_indices = random.sample(cleaned_indices, len(cleaned_indices))
        
        # 8:1:1に分割
        total_size = len(shuffled_indices)
        train_size = int(total_size * 0.8)
        test_size = int(total_size * 0.1)
        valid_size = total_size - train_size - test_size  # 残りをvalidに
        
        # データセットを分割（Hugging Faceデータセット形式を保持）
        train_dataset = dataset.select(shuffled_indices[:train_size])
        test_dataset = dataset.select(shuffled_indices[train_size:train_size + test_size])
        valid_dataset = dataset.select(shuffled_indices[train_size + test_size:])
        
        logger.info("=== データ分割結果 ===")
        logger.info(f"クリーニング済みデータ: {total_size} サンプル")
        logger.info(f"トレーニングデータ: {len(train_dataset)} サンプル ({len(train_dataset)/total_size*100:.1f}%)")
        logger.info(f"テストデータ: {len(test_dataset)} サンプル ({len(test_dataset)/total_size*100:.1f}%)")
        logger.info(f"検証データ: {len(valid_dataset)} サンプル ({len(valid_dataset)/total_size*100:.1f}%)")
        
        return train_dataset, test_dataset, valid_dataset
    
    def prepare_dataset(self, dataset, dataset_type: str = "train", use_evaluation_prompts: bool = False) -> List[Dict[str, Any]]:
        """
        データセットをOpenAI形式に準備
        
        Args:
            dataset: 処理するデータセット
            dataset_type: データセットの種類（"train", "test", "valid"）
            use_evaluation_prompts: 評価用プロンプトを使用するか
            
        Returns:
            OpenAI形式のトレーニングデータ
        """
        logger.info(f"{dataset_type}データセットを準備中...")
        logger.info(f"データセットサイズ: {len(dataset)}")
        logger.info(f"評価プロンプトモード: {use_evaluation_prompts}")
        
        training_data = []
        skipped = 0
        removed_utterances = 0
        total_original_tokens = 0
        total_cleaned_tokens = 0
        
        for i, example in enumerate(dataset):
            try:
                dialogue = example.get('dialogue', [])
                if not dialogue:
                    skipped += 1
                    continue
                
                # 元のトークン数を計算
                original_tokens = sum(len(turn.get('utterance', '')) for turn in dialogue)
                total_original_tokens += original_tokens
                
                if use_evaluation_prompts:
                    # 評価用プロンプト形式でデータ生成
                    evaluation_data = self._create_evaluation_training_data(example)
                    training_data.extend(evaluation_data)
                else:
                    # 従来の対話形式でデータ生成
                    messages = self._format_dialogue_for_openai(dialogue)
                    if not messages:
                        skipped += 1
                        continue
                    
                    # 削除後のトークン数を計算
                    cleaned_tokens = sum(len(msg.get('content', '')) for msg in messages)
                    total_cleaned_tokens += cleaned_tokens
                    
                    # 削除されたかチェック
                    if len(messages) < len(dialogue):
                        removed_utterances += 1
                    
                    # 正解スコアとメタデータを保持
                    training_sample = {
                        "messages": messages
                    }
                    
                    # review_by_client_jpを保持（正解スコア用）
                    if 'review_by_client_jp' in example:
                        training_sample['review_by_client_jp'] = example['review_by_client_jp']
                    
                    # review_by_client_enを保持
                    if 'review_by_client_en' in example:
                        training_sample['review_by_client_en'] = example['review_by_client_en']
                    
                    # topicを保持
                    if 'topic' in example:
                        training_sample['topic'] = example['topic']
                    
                    # dialogueの元の構造も保持（必要に応じて）
                    if 'dialogue' in example:
                        training_sample['dialogue'] = example['dialogue']
                    
                    # その他の重要なメタデータを保持
                    for key in ['metadata', 'annotation', 'label', 'ground_truth']:
                        if key in example:
                            training_sample[key] = example[key]
                    
                    training_data.append(training_sample)
                
                if (i + 1) % 1000 == 0:
                    logger.info(f"処理済み: {i + 1}/{len(dataset)}")
                    
            except Exception as e:
                logger.error(f"サンプル {i} の処理でエラー: {e}")
                skipped += 1
                continue
        
        logger.info(f"{dataset_type}データ準備完了: {len(training_data)}サンプル (スキップ: {skipped})")
        
        # 正解スコアの保持状況を確認
        samples_with_scores = sum(1 for item in training_data if 'review_by_client_jp' in item)
        logger.info(f"正解スコア付きサンプル: {samples_with_scores}/{len(training_data)}")
        
        if samples_with_scores == 0:
            logger.warning("⚠️  正解スコアが保持されていません！")
            logger.warning("  データ変換の修正が必要です")
        else:
            logger.info(f"✅ 正解スコアが正常に保持されています")
        
        return training_data
    
    def _create_evaluation_training_data(self, example: Dict) -> List[Dict[str, Any]]:
        """
        1つのサンプルから評価用トレーニングデータを生成
        
        Args:
            example: Kokoro Chatの1サンプル
            
        Returns:
            評価用トレーニングデータのリスト
        """
        dialogue = example.get('dialogue', [])
        review_data = example.get('review_by_client_jp', {})
        
        # 対話をテキスト形式に変換
        dialogue_text = self._convert_dialogue_to_text(dialogue)
        
        training_samples = []
        
        # 各評価項目について学習データを作成
        for eval_item in EVALUATION_ITEMS:
            # 実際の評価データがある場合はそれを使用、ない場合はスキップ
            if eval_item in review_data:
                actual_score = review_data[eval_item]
                
                # プロンプトを作成
                prompt = self._create_evaluation_prompt(dialogue_text, eval_item)
                
                # 確率分布を生成（実際のスコアに基づく）
                probability_response = self._generate_probability_response(actual_score)
                
                # OpenAI形式のメッセージを作成
                messages = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": probability_response}
                ]
                
                training_samples.append({
                    "messages": messages
                })
        
        return training_samples
    
    def _generate_probability_response(self, actual_score: float) -> str:
        """
        実際のスコアに基づいて確率分布の応答を生成
        
        Args:
            actual_score: 実際の評価スコア（0-5）
            
        Returns:
            確率分布の応答文字列
        """
        # 実際のスコアを中心とした正規分布的な確率分布を生成
        probabilities = [0.0] * 6  # 0-5点の6個
        
        if 0 <= actual_score <= 5:
            # 実際のスコアに最も近い整数を中心に分布を作成
            center = round(actual_score)
            probabilities[center] = 60.0  # 中心に60%
            
            # 隣接する点に確率を分散
            if center > 0:
                probabilities[center - 1] = 20.0
            if center < 5:
                probabilities[center + 1] = 20.0
            
            # 残りの確率を他の点に分散
            remaining_prob = 100.0 - sum(probabilities)
            if remaining_prob > 0:
                for i in range(6):
                    if probabilities[i] == 0:
                        probabilities[i] = remaining_prob / (6 - sum(1 for p in probabilities if p > 0))
        else:
            # 不正な値の場合は均等分布
            probabilities = [100.0 / 6] * 6
        
        # 確率の合計を100%に正規化
        total = sum(probabilities)
        if total > 0:
            probabilities = [p / total * 100 for p in probabilities]
        
        # 応答文字列を生成
        response_lines = []
        for i, prob in enumerate(probabilities):
            response_lines.append(f"{i}点: {prob:.0f}%")
        
        return "\n".join(response_lines)
    
    def _remove_initial_client_utterance(self, dialogue: List[Dict]) -> List[Dict]:
        """
        最初のclient発言を条件付きで削除（挨拶系のみ）
        
        Args:
            dialogue: 元の対話データ
            
        Returns:
            削除後の対話データ
        """
        if len(dialogue) < 2:
            return dialogue
        
        first_turn = dialogue[0]
        second_turn = dialogue[1]
        
        # 最初がclientで、2番目がcounselorの場合のみ削除対象
        if (first_turn.get('role', '').lower() in ['client', 'クライアント', 'user'] and 
            second_turn.get('role', '').lower() in ['counselor', 'カウンセラー', 'therapist']):
            
            first_content = first_turn.get('utterance', '').lower()
            
            # 挨拶系のパターンを定義
            greeting_patterns = [
                'こんにちは。', 'こんばんは。', 'はじめまして。', 'よろしく。',
                'お世話になります。', '相談員です。', 'お願いします。',
                'よろしくお願いします。','よろしくお願いいたします。',
                'お世話になっております。',
                '申し訳ございません。', '失礼いたします。', 'お疲れ様です。',
                'おはようございます。', '夜分遅くに', '突然で。',
                'ご相談。', 'ご質問。', 'お聞きしたい。', '教えてください。'
            ]
            
            # 挨拶系の内容かチェック
            is_greeting = any(pattern in first_content for pattern in greeting_patterns)
            
            # 短い発言（15文字以下）で挨拶系の場合のみ削除
            if is_greeting and len(first_content) <= 15:
                logger.debug(f"最初のclient発言を削除: '{first_content}'")
                return dialogue[1:]  # 最初のターンを削除
        
        return dialogue

    def _format_dialogue_for_openai(self, dialogue: List[Dict]) -> List[Dict[str, str]]:
        """
        対話データをOpenAI形式に変換
        
        Args:
            dialogue: 元の対話データ
            
        Returns:
            OpenAI形式のメッセージリスト
        """
        # 最初のclient発言を条件付きで削除
        cleaned_dialogue = self._remove_initial_client_utterance(dialogue)
        
        messages = []
        
        for turn in cleaned_dialogue:
            if not isinstance(turn, dict):
                continue
                
            role = turn.get('role', '').lower()
            utterance = turn.get('utterance', '').strip()
            
            if not utterance:
                continue
            
            # 役割をOpenAI形式にマッピング
            if role in ['counselor', 'カウンセラー', 'therapist']:
                openai_role = "assistant"
            elif role in ['client', 'クライアント', 'user']:
                openai_role = "user"
            else:
                # 不明な役割はuserとして扱う
                openai_role = "user"
            
            messages.append({
                "role": openai_role,
                "content": utterance
            })
        
        # OpenAI Fine-tuning要件: 最後のメッセージは必ずassistantである必要がある
        if messages and messages[-1]["role"] == "user":
            # 最後がuserの場合、簡単なassistantの応答を追加
            messages.append({
                "role": "assistant",
                "content": "ありがとうございました。"
            })
        
        return messages
    
    def _create_evaluation_prompt(self, dialogue_text: str, evaluation_item: str) -> str:
        """
        評価用プロンプトを作成
        
        Args:
            dialogue_text: 対話テキスト
            evaluation_item: 評価項目
            
        Returns:
            評価用プロンプト
        """
        prompt_template = """### 指示
以下の対話について「{}」の満足度を相談者の視点で0～5点で評価し、各点数の確率を出力してください。

### 対話
{}

### 出力形式（数値のみ）
0点: XX%
1点: XX%
2点: XX%
3点: XX%
4点: XX%
5点: XX%"""
        
        return prompt_template.format(evaluation_item, dialogue_text)
    
    def _convert_dialogue_to_text(self, dialogue: List[Dict]) -> str:
        """
        対話データをテキスト形式に変換
        
        Args:
            dialogue: 対話データ
            
        Returns:
            テキスト形式の対話
        """
        dialogue_lines = []
        
        for turn in dialogue:
            if not isinstance(turn, dict):
                continue
                
            role = turn.get('role', '').strip()
            utterance = turn.get('utterance', '').strip()
            
            if not utterance:
                continue
            
            # 役割を日本語表記に統一
            if role.lower() in ['counselor', 'カウンセラー', 'therapist']:
                role_jp = "カウンセラー"
            elif role.lower() in ['client', 'クライアント', 'user']:
                role_jp = "相談者"
            else:
                role_jp = "相談者"  # デフォルト
            
            dialogue_lines.append(f"{role_jp}: {utterance}")
        
        return "\n".join(dialogue_lines)
    
    def _serialize_for_json(self, obj):
        """JSONシリアライゼーション用のオブジェクト変換"""
        import datetime
        
        if isinstance(obj, datetime.datetime):
            return obj.isoformat()
        elif isinstance(obj, datetime.date):
            return obj.isoformat()
        elif isinstance(obj, datetime.time):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {k: self._serialize_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._serialize_for_json(item) for item in obj]
        else:
            return obj

    def save_training_data(self, training_data: List[Dict], filename: str = None, dataset_type: str = "train") -> str:
        """
        トレーニングデータをJSONLファイルとして保存（正解スコア付き）
        
        Args:
            training_data: トレーニングデータ
            filename: ファイル名（指定しない場合は自動生成）
            dataset_type: データセットの種類（"train", "test", "valid"）
            
        Returns:
            保存されたファイルのパス
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{dataset_type}_data_{timestamp}.jsonl"
        
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            for item in training_data:
                # JSONシリアライゼーション用にオブジェクトを変換
                serialized_item = self._serialize_for_json(item)
                json.dump(serialized_item, f, ensure_ascii=False)
                f.write('\n')
        
        logger.info(f"{dataset_type}データを保存: {filepath}")
        
        # 保存されたデータの構造を確認
        if training_data:
            first_item = training_data[0]
            keys = list(first_item.keys())
            logger.info(f"保存されたデータの構造: {keys}")
            
            if 'review_by_client_jp' in keys:
                logger.info("✅ 正解スコア（review_by_client_jp）が正常に保存されています")
            else:
                logger.warning("⚠️  正解スコアが保存されていません")
        
        return str(filepath)
    
    def save_all_datasets(self, train_data: List[Dict], test_data: List[Dict], valid_data: List[Dict], 
                         timestamp: str = None) -> Dict[str, str]:
        """
        全てのデータセット（train/test/valid）を保存
        
        Args:
            train_data: トレーニングデータ
            test_data: テストデータ
            valid_data: 検証データ
            timestamp: タイムスタンプ（指定しない場合は自動生成）
            
        Returns:
            保存されたファイルパスの辞書
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        filepaths = {}
        
        # 各データセットを保存
        datasets = [
            ("train", train_data),
            ("test", test_data),
            ("valid", valid_data)
        ]
        
        for dataset_type, data in datasets:
            filename = f"{dataset_type}_data_{timestamp}.jsonl"
            filepath = self.save_training_data(data, filename, dataset_type)
            filepaths[dataset_type] = filepath
        
        logger.info("=== 全データセット保存完了 ===")
        for dataset_type, filepath in filepaths.items():
            logger.info(f"{dataset_type}: {filepath}")
        
        return filepaths
    
    def upload_training_file(self, filepath: str) -> str:
        """
        トレーニングファイルをOpenAIにアップロード
        
        Args:
            filepath: トレーニングファイルのパス
            
        Returns:
            ファイルID
        """
        logger.info(f"ファイルをOpenAIにアップロード中: {filepath}")
        
        with open(filepath, 'rb') as f:
            response = self.client.files.create(
                file=f,
                purpose='fine-tune'
            )
        
        file_id = response.id
        logger.info(f"ファイルアップロード完了: {file_id}")
        return file_id
    
    def create_fine_tune_job(self, training_file_id: str, epochs: int = 10) -> str:
        """
        ファインチューニングジョブを作成
        
        Args:
            training_file_id: トレーニングファイルID
            epochs: エポック数
            
        Returns:
            ジョブID
        """
        logger.info(f"ファインチューニングジョブを作成中... (epochs: {epochs})")
        
        response = self.client.fine_tuning.jobs.create(
            training_file=training_file_id,
            model="gpt-4o-mini-2024-07-18",
            hyperparameters={
                "n_epochs": epochs
            },
            learning_rate_multiplier=0.1  # 学習率を0.1倍に設定
        )
        
        job_id = response.id
        logger.info(f"ファインチューニングジョブ作成完了: {job_id}")
        return job_id
    
    def monitor_fine_tune_job(self, job_id: str) -> Dict[str, Any]:
        """
        ファインチューニングジョブを監視
        
        Args:
            job_id: ジョブID
            
        Returns:
            最終的なジョブ情報
        """
        logger.info(f"ファインチューニングジョブを監視中: {job_id}")
        
        # システム監視開始
        self.monitor.start_monitoring()
        
        try:
            while True:
                job = self.client.fine_tuning.jobs.retrieve(job_id)
                status = job.status
                
                logger.info(f"ジョブステータス: {status}")
                
                if status in ['succeeded', 'failed', 'cancelled']:
                    break
                
                # 進捗情報があれば表示
                if hasattr(job, 'trained_tokens') and job.trained_tokens:
                    logger.info(f"学習済みトークン数: {job.trained_tokens:,}")
                
                time.sleep(60)  # 1分間隔で確認
                
        finally:
            # システム監視停止
            self.monitor.stop_monitoring()
        
        if job.status == 'succeeded':
            logger.info(f"ファインチューニング完了! モデルID: {job.fine_tuned_model}")
        else:
            logger.error(f"ファインチューニング失敗: {job.status}")
            # エラーの詳細情報を取得・表示
            if hasattr(job, 'error') and job.error:
                logger.error(f"エラー詳細: {job.error}")
            if hasattr(job, 'result_files') and job.result_files:
                logger.info(f"結果ファイル: {job.result_files}")
            # ジョブの全情報をデバッグ用に出力
            logger.info(f"ジョブ詳細情報: {job}")
            
        return job
    
    def monitor_fine_tune_job_with_progress(self, job_id: str) -> Dict[str, Any]:
        """
        進行状況表示付きファインチューニングジョブ監視
        
        Args:
            job_id: ジョブID
            
        Returns:
            最終的なジョブ情報
        """
        logger.info(f"ファインチューニングジョブを監視中: {job_id}")
        
        # システム監視開始
        self.monitor.start_monitoring()
        
        try:
            last_progress = 0
            while True:
                job = self.client.fine_tuning.jobs.retrieve(job_id)
                status = job.status
                
                logger.info(f"ジョブステータス: {status}")
                
                if status in ['succeeded', 'failed', 'cancelled']:
                    if self.progress_monitor:
                        self.progress_monitor.complete_phase("fine_tuning")
                    break
                
                # 進捗情報があれば表示・更新
                if hasattr(job, 'trained_tokens') and job.trained_tokens:
                    logger.info(f"学習済みトークン数: {job.trained_tokens:,}")
                    
                    # 進行状況を推定（大まかな計算）
                    if self.progress_monitor:
                        # エポック数から推定進行状況を計算
                        estimated_total_tokens = 11618410 * 10  # 概算値
                        progress = min((job.trained_tokens / estimated_total_tokens) * 100, 99)
                        
                        if progress > last_progress + 5:  # 5%刻みで更新
                            self.progress_monitor.update_phase(
                                "fine_tuning", 
                                int(progress),
                                f"学習中... ({job.trained_tokens:,} tokens)"
                            )
                            last_progress = progress
                
                time.sleep(60)  # 1分間隔で確認
                
        finally:
            # システム監視停止
            self.monitor.stop_monitoring()
        
        if job.status == 'succeeded':
            logger.info(f"ファインチューニング完了! モデルID: {job.fine_tuned_model}")
        else:
            logger.error(f"ファインチューニング失敗: {job.status}")
            
        return job
    
    def evaluate_model(self, model_id: str, test_messages: List[Dict], dataset_type: str = "test", 
                      max_test_cases: int = 10) -> Dict[str, Any]:
        """
        ファインチューニングしたモデルを評価
        
        Args:
            model_id: モデルID
            test_messages: テストメッセージ
            dataset_type: データセットの種類（"test", "valid"）
            max_test_cases: 最大テストケース数
            
        Returns:
            評価結果
        """
        logger.info(f"モデル評価中 ({dataset_type}データセット): {model_id}")
        
        results = {
            "model_id": model_id,
            "dataset_type": dataset_type,
            "total_test_cases": len(test_messages),
            "evaluated_cases": min(len(test_messages), max_test_cases),
            "responses": []
        }
        
        test_cases_to_run = min(len(test_messages), max_test_cases)
        
        for i, messages in enumerate(test_messages[:test_cases_to_run]):
            try:
                response = self.client.chat.completions.create(
                    model=model_id,
                    messages=messages,
                    max_tokens=150,
                    temperature=0.7
                )
                
                result = {
                    "test_case": i + 1,
                    "input": messages,
                    "output": response.choices[0].message.content,
                    "usage": response.usage.dict() if response.usage else None
                }
                results["responses"].append(result)
                
                logger.info(f"{dataset_type}テストケース {i + 1}/{test_cases_to_run} 完了")
                
            except Exception as e:
                logger.error(f"{dataset_type}テストケース {i + 1} でエラー: {e}")
                results["responses"].append({
                    "test_case": i + 1,
                    "error": str(e)
                })
        
        return results
    
    def evaluate_model_comprehensive(self, model_id: str, test_data: List[Dict], valid_data: List[Dict], 
                                   max_test_cases: int = 10) -> Dict[str, Any]:
        """
        テストデータと検証データの両方でモデルを包括的に評価
        
        Args:
            model_id: モデルID
            test_data: テストデータ
            valid_data: 検証データ
            max_test_cases: 各データセットでの最大テストケース数
            
        Returns:
            包括的な評価結果
        """
        logger.info("=== 包括的モデル評価開始 ===")
        
        # テストデータでの評価
        test_messages = [item["messages"] for item in test_data]
        test_results = self.evaluate_model(model_id, test_messages, "test", max_test_cases)
        
        # 検証データでの評価
        valid_messages = [item["messages"] for item in valid_data]
        valid_results = self.evaluate_model(model_id, valid_messages, "valid", max_test_cases)
        
        # 結果を統合
        comprehensive_results = {
            "model_id": model_id,
            "evaluation_timestamp": datetime.now().isoformat(),
            "test_evaluation": test_results,
            "valid_evaluation": valid_results,
            "summary": {
                "test_cases_evaluated": test_results["evaluated_cases"],
                "valid_cases_evaluated": valid_results["evaluated_cases"],
                "total_cases_evaluated": test_results["evaluated_cases"] + valid_results["evaluated_cases"]
            }
        }
        
        logger.info("=== 包括的モデル評価完了 ===")
        return comprehensive_results
    
    def save_results(self, results: Dict[str, Any], filename: str = None) -> str:
        """
        評価結果を保存
        
        Args:
            results: 評価結果
            filename: ファイル名
            
        Returns:
            保存されたファイルのパス
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"evaluation_results_{timestamp}.json"
        
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"評価結果を保存: {filepath}")
        return str(filepath)
    
    def save_model_info(self, model_id: str, training_params: Dict[str, Any], 
                       evaluation_results: Dict[str, Any] = None) -> str:
        """
        モデル情報を保存（後で使用するため）
        
        Args:
            model_id: ファインチューニング済みモデルID
            training_params: 学習パラメータ
            evaluation_results: 評価結果（オプション）
            
        Returns:
            保存されたファイルのパス
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        model_info = {
            "model_id": model_id,
            "timestamp": timestamp,
            "base_model": "gpt-4o-mini-2024-07-18",
            "training_params": training_params,
            "evaluation_results": evaluation_results,
            "usage_instructions": {
                "description": "このモデルをOpenAI APIで使用する方法",
                "example_code": f'''
from openai import OpenAI
client = OpenAI(api_key="your-api-key")

response = client.chat.completions.create(
    model="{model_id}",
    messages=[
        {{"role": "user", "content": "相談内容をここに入力"}}
    ],
    max_tokens=150,
    temperature=0.7
)
print(response.choices[0].message.content)
'''
            }
        }
        
        filename = f"model_info_{model_id.replace(':', '_')}_{timestamp}.json"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, ensure_ascii=False, indent=2)
        
        logger.info(f"モデル情報を保存: {filepath}")
        logger.info(f"モデルID: {model_id}")
        logger.info("このモデルはOpenAI APIを通じて使用できます")
        
        return str(filepath)
    
    def load_model_from_file(self, model_info_file: str) -> Dict[str, Any]:
        """
        保存されたモデル情報ファイルから情報を読み込み
        
        Args:
            model_info_file: モデル情報ファイルのパス
            
        Returns:
            モデル情報
        """
        with open(model_info_file, 'r', encoding='utf-8') as f:
            model_info = json.load(f)
        
        logger.info(f"モデル情報を読み込み: {model_info['model_id']}")
        return model_info
    
    def list_saved_models(self) -> List[Dict[str, Any]]:
        """
        保存されている全モデル情報を一覧表示
        
        Returns:
            モデル情報のリスト
        """
        model_files = list(self.output_dir.glob("model_info_*.json"))
        models = []
        
        for file_path in model_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    model_info = json.load(f)
                    models.append({
                        "file_path": str(file_path),
                        "model_id": model_info.get("model_id"),
                        "timestamp": model_info.get("timestamp"),
                        "training_params": model_info.get("training_params", {})
                    })
            except Exception as e:
                logger.error(f"モデルファイル読み込みエラー {file_path}: {e}")
        
        # タイムスタンプでソート（新しい順）
        models.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        
        logger.info(f"保存されているモデル数: {len(models)}")
        for model in models:
            logger.info(f"  - {model['model_id']} ({model['timestamp']})")
        
        return models
    
    def use_saved_model(self, model_id: str, messages: List[Dict[str, str]], 
                       max_tokens: int = 150, temperature: float = 0.7) -> str:
        """
        保存されたモデルを使用して推論実行
        
        Args:
            model_id: モデルID
            messages: 入力メッセージ
            max_tokens: 最大トークン数
            temperature: 生成温度
            
        Returns:
            モデルの応答
        """
        try:
            response = self.client.chat.completions.create(
                model=model_id,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"モデル使用エラー: {e}")
            raise
    
    def run_full_pipeline(self, epochs: int = 10, max_samples: Optional[int] = None, 
                         seed: int = 42, max_test_cases: int = 10, use_evaluation_prompts: bool = False,
                         max_tokens_per_sample: int = 4000, max_messages_per_dialogue: int = 50):
        """
        8:1:1データ分割対応のフルパイプラインを実行（制限チェック付き）
        
        Args:
            epochs: エポック数
            max_samples: 最大サンプル数（デバッグ用）
            seed: データ分割の再現性のためのシード
            max_test_cases: 評価時の最大テストケース数
            use_evaluation_prompts: 評価用プロンプト形式を使用するか
            max_tokens_per_sample: サンプルあたりの最大トークン数
            max_messages_per_dialogue: 対話あたりの最大メッセージ数
        """
        logger.info("=== OpenAI GPT-4o mini Supervised Fine Tuning 開始 (8:1:1データ分割・制限チェック付き) ===")
        logger.info(f"制限設定: 最大トークン数={max_tokens_per_sample}, 最大メッセージ数={max_messages_per_dialogue}")
        
        # 進行状況監視開始
        if self.progress_monitor:
            self.progress_monitor.start_monitoring(total_phases=6)
        
        try:
            # 1. データセットの読み込みと分割（制限チェック付き）
            if self.progress_monitor:
                self.progress_monitor.start_phase("data_loading", "データセットの読み込み・分割・クリーニング", total=100)
            
            train_dataset, test_dataset, valid_dataset = self.load_and_split_dataset(
                max_samples=max_samples, seed=seed,
                max_tokens_per_sample=max_tokens_per_sample,
                max_messages_per_dialogue=max_messages_per_dialogue
            )
            
            if self.progress_monitor:
                self.progress_monitor.complete_phase("data_loading")
            
            # 2. 各データセットをOpenAI形式に変換
            if self.progress_monitor:
                self.progress_monitor.start_phase("data_preparation", "データ変換・準備", total=100)
            
            train_data = self.prepare_dataset(train_dataset, "train", use_evaluation_prompts)
            if self.progress_monitor:
                self.progress_monitor.update_phase("data_preparation", 33, "学習データ変換完了")
                
            test_data = self.prepare_dataset(test_dataset, "test", use_evaluation_prompts)
            if self.progress_monitor:
                self.progress_monitor.update_phase("data_preparation", 66, "テストデータ変換完了")
                
            valid_data = self.prepare_dataset(valid_dataset, "valid", use_evaluation_prompts)
            if self.progress_monitor:
                self.progress_monitor.update_phase("data_preparation", 100, "検証データ変換完了")
                self.progress_monitor.complete_phase("data_preparation")
            
            # 3. 全データセットを保存
            filepaths = self.save_all_datasets(train_data, test_data, valid_data)
            
            # 4. トレーニングファイルをアップロード
            if self.progress_monitor:
                self.progress_monitor.start_phase("file_upload", "ファイルアップロード", total=100)
            
            file_id = self.upload_training_file(filepaths["train"])
            
            if self.progress_monitor:
                self.progress_monitor.complete_phase("file_upload")
            
            # 5. ファインチューニングジョブ作成
            if self.progress_monitor:
                self.progress_monitor.start_phase("fine_tuning", "ファインチューニング実行", total=100)
            
            job_id = self.create_fine_tune_job(file_id, epochs=epochs)
            
            # 6. ジョブ監視
            job_result = self.monitor_fine_tune_job_with_progress(job_id)
            
            if job_result.status == 'succeeded':
                model_id = job_result.fine_tuned_model
                
                # 7. 包括的モデル評価（テストデータと検証データの両方）
                if self.progress_monitor:
                    self.progress_monitor.start_phase("evaluation", "モデル評価", total=100)
                
                evaluation_results = self.evaluate_model_comprehensive(
                    model_id, test_data, valid_data, max_test_cases
                )
                
                if self.progress_monitor:
                    self.progress_monitor.complete_phase("evaluation")
                
                # 8. 結果保存
                if self.progress_monitor:
                    self.progress_monitor.start_phase("results_saving", "結果保存", total=100)
                
                self.save_results(evaluation_results, 
                                f"comprehensive_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
                
                # 9. モデル情報保存（後で使用するため）
                training_params = {
                    "epochs": epochs,
                    "max_samples": max_samples,
                    "seed": seed,
                    "use_evaluation_prompts": use_evaluation_prompts,
                    "data_splits": {
                        "train": len(train_data),
                        "test": len(test_data),
                        "valid": len(valid_data)
                    }
                }
                model_info_file = self.save_model_info(model_id, training_params, evaluation_results)
                
                if self.progress_monitor:
                    self.progress_monitor.complete_phase("results_saving")
                
                logger.info("=== ファインチューニング完了 ===")
                logger.info(f"モデルID: {model_id}")
                logger.info(f"モデル情報ファイル: {model_info_file}")
                logger.info(f"データ分割: Train={len(train_data)}, Test={len(test_data)}, Valid={len(valid_data)}")
                logger.info(f"評価ケース: Test={evaluation_results['test_evaluation']['evaluated_cases']}, "
                          f"Valid={evaluation_results['valid_evaluation']['evaluated_cases']}")
                
                return {
                    "model_id": model_id,
                    "model_info_file": model_info_file,
                    "job_result": job_result,
                    "evaluation_results": evaluation_results,
                    "data_splits": {
                        "train": len(train_data),
                        "test": len(test_data),
                        "valid": len(valid_data)
                    },
                    "filepaths": filepaths
                }
                
            else:
                logger.error("ファインチューニングが失敗しました")
                return {
                    "job_result": job_result,
                    "data_splits": {
                        "train": len(train_data),
                        "test": len(test_data),
                        "valid": len(valid_data)
                    },
                    "filepaths": filepaths
                }
                
        except Exception as e:
            logger.error(f"パイプライン実行中にエラー: {e}")
            if self.progress_monitor:
                self.progress_monitor.add_log(f"エラー発生: {e}", "error")
            raise
        finally:
            # 進行状況監視停止
            if self.progress_monitor:
                self.progress_monitor.stop_monitoring()

    def split_dataset_into_batches(self, dataset, batch_split_size: int = None, epochs: int = 20) -> List[Dict[str, Any]]:
        """
        データセットをバッチに分割
        
        Args:
            dataset: 分割するデータセット
            batch_split_size: 1バッチあたりのサンプル数（ファイル分割用）
            epochs: 各バッチのエポック数
            
        Returns:
            バッチ情報のリスト
        """
        if batch_split_size is None:
            batch_split_size = DEFAULT_BATCH_SPLIT_SIZE
            
        batches = []
        total_samples = len(dataset)
        
        logger.info(f"データセットをバッチに分割中... (総サンプル数: {total_samples}, バッチ分割サイズ: {batch_split_size})")
        
        for i in range(0, total_samples, batch_split_size):
            end_idx = min(i + batch_split_size, total_samples)
            batch_dataset = dataset.select(range(i, end_idx))
            
            batch_info = {
                'batch_id': len(batches) + 1,
                'start_idx': i,
                'end_idx': end_idx,
                'data': batch_dataset,
                'samples': len(batch_dataset),
                'epochs': epochs,
                'estimated_tokens': len(batch_dataset) * epochs * 6000,  # 概算
                'estimated_size_mb': (len(batch_dataset) * epochs * 6000 * 4) / (1024 * 1024)  # 概算ファイルサイズ
            }
            
            batches.append(batch_info)
            
            logger.info(f"バッチ {batch_info['batch_id']}: サンプル {i+1}-{end_idx} "
                       f"({batch_info['samples']}サンプル, 推定サイズ: {batch_info['estimated_size_mb']:.1f}MB)")
        
        logger.info(f"バッチ分割完了: {len(batches)}バッチ")
        return batches
    
    def prepare_batch_training_data(self, batch: Dict[str, Any], dataset_type: str = "train", 
                                  use_evaluation_prompts: bool = False) -> List[Dict[str, Any]]:
        """
        1つのバッチのトレーニングデータを準備
        
        Args:
            batch: バッチ情報
            dataset_type: データセットの種類
            use_evaluation_prompts: 評価用プロンプトを使用するか
            
        Returns:
            トレーニングデータ
        """
        logger.info(f"バッチ {batch['batch_id']} のトレーニングデータを準備中... "
                   f"({batch['samples']}サンプル)")
        
        return self.prepare_dataset(batch['data'], dataset_type, use_evaluation_prompts)
    
    def save_batch_training_data(self, training_data: List[Dict], batch_id: int, 
                                timestamp: str = None) -> str:
        """
        バッチのトレーニングデータを保存
        
        Args:
            training_data: トレーニングデータ
            batch_id: バッチID
            timestamp: タイムスタンプ
            
        Returns:
            保存されたファイルのパス
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        filename = f"batch_{batch_id:02d}_train_data_{timestamp}.jsonl"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            for item in training_data:
                json.dump(item, f, ensure_ascii=False)
                f.write('\n')
        
        logger.info(f"バッチ {batch_id} のトレーニングデータを保存: {filepath}")
        return str(filepath)
    
    def upload_batch_file(self, filepath: str, batch_id: int) -> str:
        """
        バッチファイルをOpenAIにアップロード
        
        Args:
            filepath: ファイルパス
            batch_id: バッチID
            
        Returns:
            ファイルID
        """
        logger.info(f"バッチ {batch_id} のファイルをOpenAIにアップロード中: {filepath}")
        
        with open(filepath, 'rb') as f:
            response = self.client.files.create(
                file=f,
                purpose='fine-tune'
            )
        
        file_id = response.id
        logger.info(f"バッチ {batch_id} のファイルアップロード完了: {file_id}")
        return file_id
    
    def create_batch_fine_tune_job(self, training_file_id: str, batch_id: int, 
                                  epochs: int = 20) -> str:
        """
        バッチのファインチューニングジョブを作成
        
        Args:
            training_file_id: トレーニングファイルID
            batch_id: バッチID
            epochs: エポック数
            
        Returns:
            ジョブID
        """
        logger.info(f"バッチ {batch_id} のファインチューニングジョブを作成中... "
                   f"(epochs: {epochs})")
        
        # デバッグ用: リクエストパラメータをログ出力
        request_params = {
            "training_file": training_file_id,
            "model": "gpt-4o-mini-2024-07-18"
        }
        logger.info(f"リクエストパラメータ: {request_params}")
        
        try:
            response = self.client.fine_tuning.jobs.create(
                training_file=training_file_id,
                model="gpt-4o-mini-2024-07-18",
                hyperparameters={
                    "n_epochs": "auto",
                    "learning_rate_multiplier": 0.1  # 学習率を0.1倍に設定
                }
            )
            
            job_id = response.id
            logger.info(f"バッチ {batch_id} のファインチューニングジョブ作成完了: {job_id}")
            return job_id
            
        except Exception as e:
            logger.error(f"OpenAI API呼び出しエラー: {type(e).__name__}: {str(e)}")
            if hasattr(e, 'response') and hasattr(e.response, 'json'):
                try:
                    error_detail = e.response.json()
                    logger.error(f"OpenAI API エラーレスポンス: {error_detail}")
                except:
                    logger.error(f"OpenAI API エラーレスポンス: {e.response.text}")
            raise
    
    def run_batch_fine_tuning(self, batches: List[Dict[str, Any]], epochs: int = 20, 
                             use_evaluation_prompts: bool = False) -> List[Dict[str, Any]]:
        """
        バッチ分割でファインチューニングを実行
        
        Args:
            batches: バッチ情報のリスト
            epochs: 各バッチのエポック数
            use_evaluation_prompts: 評価用プロンプトを使用するか
            
        Returns:
            各バッチの結果
        """
        logger.info(f"=== バッチ分割ファインチューニング開始 ===")
        logger.info(f"バッチ数: {len(batches)}")
        logger.info(f"各バッチ: {epochs}エポック")
        
        batch_results = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for batch in batches:
            try:
                logger.info(f"=== バッチ {batch['batch_id']} 処理開始 ===")
                
                # 1. トレーニングデータ準備
                training_data = self.prepare_batch_training_data(
                    batch, "train", use_evaluation_prompts
                )
                
                # 2. ファイル保存
                filepath = self.save_batch_training_data(training_data, batch['batch_id'], timestamp)
                
                # 3. ファイルアップロード
                file_id = self.upload_batch_file(filepath, batch['batch_id'])
                
                # 4. ファインチューニングジョブ作成
                job_id = self.create_batch_fine_tune_job(
                    file_id, batch['batch_id'], epochs
                )
                
                # 5. 結果を記録
                batch_result = {
                    'batch_id': batch['batch_id'],
                    'samples': batch['samples'],
                    'training_data_count': len(training_data),
                    'filepath': filepath,
                    'file_id': file_id,
                    'job_id': job_id,
                    'status': 'created',
                    'timestamp': timestamp
                }
                
                batch_results.append(batch_result)
                
                logger.info(f"=== バッチ {batch['batch_id']} 処理完了 ===")
                
            except Exception as e:
                logger.error(f"バッチ {batch['batch_id']} でエラー: {e}")
                logger.error(f"エラーの詳細: {type(e).__name__}: {str(e)}")
                if hasattr(e, '__dict__'):
                    logger.error(f"エラー属性: {e.__dict__}")
                batch_result = {
                    'batch_id': batch['batch_id'],
                    'samples': batch['samples'],
                    'status': 'failed',
                    'error': str(e),
                    'timestamp': timestamp
                }
                batch_results.append(batch_result)
        
        logger.info(f"=== バッチ分割ファインチューニング完了 ===")
        logger.info(f"成功: {len([r for r in batch_results if r['status'] == 'created'])}バッチ")
        logger.info(f"失敗: {len([r for r in batch_results if r['status'] == 'failed'])}バッチ")
        
        return batch_results
    
    def monitor_batch_jobs(self, batch_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        全バッチのジョブを監視
        
        Args:
            batch_results: バッチ結果のリスト
            
        Returns:
            更新されたバッチ結果
        """
        logger.info("=== 全バッチジョブの監視開始 ===")
        
        # システム監視開始
        self.monitor.start_monitoring()
        
        try:
            for batch_result in batch_results:
                if batch_result['status'] == 'created':
                    job_id = batch_result['job_id']
                    logger.info(f"バッチ {batch_result['batch_id']} のジョブ {job_id} を監視中...")
                    
                    # ジョブの監視
                    job_result = self.monitor_fine_tune_job(job_id)
                    
                    # 結果を更新
                    batch_result['final_status'] = job_result.status
                    batch_result['final_model_id'] = getattr(job_result, 'fine_tuned_model', None)
                    batch_result['completion_time'] = datetime.now().isoformat()
                    
                    if job_result.status == 'succeeded':
                        logger.info(f"バッチ {batch_result['batch_id']} 完了! モデルID: {batch_result['final_model_id']}")
                    else:
                        logger.error(f"バッチ {batch_result['batch_id']} 失敗: {job_result.status}")
                        
        finally:
            # システム監視停止
            self.monitor.stop_monitoring()
        
        return batch_results

    def run_batch_pipeline(self, epochs: int = 20, max_samples: Optional[int] = None, 
                          seed: int = 42, max_test_cases: int = 10, use_evaluation_prompts: bool = False,
                          max_tokens_per_sample: int = None, max_messages_per_dialogue: int = None,
                          batch_split_size: int = None):
        """
        バッチ分割対応のフルパイプラインを実行
        
        Args:
            epochs: エポック数
            max_samples: 最大サンプル数
            seed: データ分割の再現性のためのシード
            max_test_cases: 評価時の最大テストケース数
            use_evaluation_prompts: 評価用プロンプト形式を使用するか
            max_tokens_per_sample: サンプルあたりの最大トークン数
            max_messages_per_dialogue: 対話あたりの最大メッセージ数
            batch_split_size: バッチ分割サイズ
        """
        logger.info("=== OpenAI GPT-4o mini バッチ分割ファインチューニング開始 ===")
        
        # 進行状況監視開始
        if self.progress_monitor:
            self.progress_monitor.start_monitoring(total_phases=7)
        
        try:
            # 1. データセットの読み込みと分割
            if self.progress_monitor:
                self.progress_monitor.start_phase("data_loading", "データセットの読み込み・分割・クリーニング", total=100)
            
            train_dataset, test_dataset, valid_dataset = self.load_and_split_dataset(
                max_samples=max_samples, seed=seed,
                max_tokens_per_sample=max_tokens_per_sample,
                max_messages_per_dialogue=max_messages_per_dialogue
            )
            
            if self.progress_monitor:
                self.progress_monitor.complete_phase("data_loading")
            
            # 2. トレーニングデータをバッチに分割
            if self.progress_monitor:
                self.progress_monitor.start_phase("batch_splitting", "トレーニングデータのバッチ分割", total=100)
            
            if batch_split_size is None:
                batch_split_size = DEFAULT_BATCH_SPLIT_SIZE
            
            batches = self.split_dataset_into_batches(train_dataset, batch_split_size, epochs)
            
            if self.progress_monitor:
                self.progress_monitor.complete_phase("batch_splitting")
            
            # 3. 各バッチでファインチューニング実行
            if self.progress_monitor:
                self.progress_monitor.start_phase("batch_processing", "バッチ処理・ファインチューニング", total=100)
            
            batch_results = self.run_batch_fine_tuning(
                batches, epochs, use_evaluation_prompts
            )
            
            if self.progress_monitor:
                self.progress_monitor.complete_phase("batch_processing")
            
            # 4. バッチジョブの監視
            if self.progress_monitor:
                self.progress_monitor.start_phase("job_monitoring", "バッチジョブの監視", total=100)
            
            final_batch_results = self.monitor_batch_jobs(batch_results)
            
            if self.progress_monitor:
                self.progress_monitor.complete_phase("job_monitoring")
            
            # 5. 結果の集計
            if self.progress_monitor:
                self.progress_monitor.start_phase("results_aggregation", "結果の集計", total=100)
            
            successful_batches = [r for r in final_batch_results if r.get('final_status') == 'succeeded']
            failed_batches = [r for r in final_batch_results if r.get('final_status') != 'succeeded']
            
            total_samples_processed = sum(r['samples'] for r in final_batch_results)
            successful_samples = sum(r['samples'] for r in successful_batches)
            
            # 6. 結果保存
            results_summary = {
                "pipeline_type": "batch_fine_tuning",
                "timestamp": datetime.now().isoformat(),
                "total_batches": len(batches),
                "successful_batches": len(successful_batches),
                "failed_batches": len(failed_batches),
                "total_samples": total_samples_processed,
                "successful_samples": successful_samples,
                "success_rate": successful_samples / total_samples_processed if total_samples_processed > 0 else 0,
                "batch_results": final_batch_results,
                "training_params": {
                    "epochs": epochs,
                    "max_samples": max_samples,
                    "seed": seed,
                    "use_evaluation_prompts": use_evaluation_prompts,
                    "batch_split_size": batch_split_size,
                    "max_tokens_per_sample": max_tokens_per_sample,
                    "max_messages_per_dialogue": max_messages_per_dialogue
                },
                "data_splits": {
                    "train": len(train_dataset),
                    "test": len(test_dataset),
                    "valid": len(valid_dataset)
                }
            }
            
            # 結果を保存
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = self.output_dir / f"batch_fine_tuning_results_{timestamp}.json"
            
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results_summary, f, ensure_ascii=False, indent=2)
            
            if self.progress_monitor:
                self.progress_monitor.complete_phase("results_aggregation")
            
            # 7. 完了
            if self.progress_monitor:
                self.progress_monitor.start_phase("completion", "完了", total=100)
            
            logger.info("=== バッチ分割ファインチューニング完了 ===")
            logger.info(f"総バッチ数: {len(batches)}")
            logger.info(f"成功バッチ: {len(successful_batches)}")
            logger.info(f"失敗バッチ: {len(failed_batches)}")
            logger.info(f"総サンプル数: {total_samples_processed}")
            logger.info(f"成功サンプル数: {successful_samples}")
            logger.info(f"成功率: {results_summary['success_rate']*100:.1f}%")
            logger.info(f"結果ファイル: {results_file}")
            
            if successful_batches:
                logger.info("=== 成功したバッチのモデルID ===")
                for batch in successful_batches:
                    logger.info(f"バッチ {batch['batch_id']}: {batch['final_model_id']}")
            
            if self.progress_monitor:
                self.progress_monitor.complete_phase("completion")
            
            return results_summary
            
        except Exception as e:
            logger.error(f"バッチパイプライン実行中にエラー: {e}")
            if self.progress_monitor:
                self.progress_monitor.add_log(f"エラー発生: {e}", "error")
            raise
        finally:
            # 進行状況監視停止
            if self.progress_monitor:
                self.progress_monitor.stop_monitoring()

def main():
    """メイン関数"""
    try:
        # SFTインスタンス作成（.envファイルから自動的にAPIキーを読み込み）
        sft = OpenAISFT()
        
        # 中規模設定で実行（クォータ制限内で最大限）
        logger.info("=== 中規模設定でバッチ分割ファインチューニングを実行します ===")
        sft.run_batch_pipeline(
            epochs=70,                       # 70エポック（前回成功した設定）
            max_samples=1500,                # 1,500サンプル（前回成功した設定）
            batch_split_size=300,            # 300サンプル/バッチ（前回成功した設定）
            max_tokens_per_sample=12000,     # 12Kトークン（前回成功した設定）
            max_messages_per_dialogue=250,   # 250メッセージ（前回成功した設定）
            seed=42                          # 元のランダムシード（再現性確保）
        )
        
    except ValueError as e:
        logger.error(f"設定エラー: {e}")
        logger.error("プロジェクトルートに.envファイルを作成し、OPENAI_API_KEY=your-api-key-here を設定してください")
    except Exception as e:
        logger.error(f"実行エラー: {e}")
        raise

if __name__ == "__main__":
    main()
