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

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    
    def __init__(self, api_key: Optional[str] = None):
        """
        初期化
        
        Args:
            api_key: OpenAI APIキー（環境変数OPENAI_API_KEYからも取得可能）
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
        
        # 出力ディレクトリの作成
        self.output_dir = Path("openai_sft_outputs")
        self.output_dir.mkdir(exist_ok=True)
        
    def prepare_dataset(self, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Kokoro Chatデータセットを準備
        
        Args:
            max_samples: 最大サンプル数（デバッグ用）
            
        Returns:
            OpenAI形式のトレーニングデータ
        """
        logger.info("KokoroChat データセットを読み込み中...")
        dataset = load_dataset("UEC-InabaLab/KokoroChat", split="train")
        
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
            logger.info(f"デバッグ用に{max_samples}サンプルに制限")
        
        logger.info(f"データセットサイズ: {len(dataset)}")
        
        training_data = []
        skipped = 0
        
        for i, example in enumerate(dataset):
            try:
                dialogue = example.get('dialogue', [])
                if not dialogue:
                    skipped += 1
                    continue
                
                # 対話を整形
                messages = self._format_dialogue_for_openai(dialogue)
                if not messages:
                    skipped += 1
                    continue
                
                training_data.append({
                    "messages": messages
                })
                
                if (i + 1) % 1000 == 0:
                    logger.info(f"処理済み: {i + 1}/{len(dataset)}")
                    
            except Exception as e:
                logger.error(f"サンプル {i} の処理でエラー: {e}")
                skipped += 1
                continue
        
        logger.info(f"トレーニングデータ準備完了: {len(training_data)}サンプル (スキップ: {skipped})")
        return training_data
    
    def _format_dialogue_for_openai(self, dialogue: List[Dict]) -> List[Dict[str, str]]:
        """
        対話データをOpenAI形式に変換
        
        Args:
            dialogue: 元の対話データ
            
        Returns:
            OpenAI形式のメッセージリスト
        """
        messages = []
        
        for turn in dialogue:
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
        
        return messages
    
    def save_training_data(self, training_data: List[Dict], filename: str = None) -> str:
        """
        トレーニングデータをJSONLファイルとして保存
        
        Args:
            training_data: トレーニングデータ
            filename: ファイル名（指定しない場合は自動生成）
            
        Returns:
            保存されたファイルのパス
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"training_data_{timestamp}.jsonl"
        
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            for item in training_data:
                json.dump(item, f, ensure_ascii=False)
                f.write('\n')
        
        logger.info(f"トレーニングデータを保存: {filepath}")
        return str(filepath)
    
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
    
    def create_fine_tune_job(self, training_file_id: str, epochs: int = 10, batch_size: int = 16) -> str:
        """
        ファインチューニングジョブを作成
        
        Args:
            training_file_id: トレーニングファイルID
            epochs: エポック数
            batch_size: バッチサイズ
            
        Returns:
            ジョブID
        """
        logger.info(f"ファインチューニングジョブを作成中... (epochs: {epochs}, batch_size: {batch_size})")
        
        response = self.client.fine_tuning.jobs.create(
            training_file=training_file_id,
            model="gpt-4o-mini-2024-07-18",
            hyperparameters={
                "n_epochs": epochs,
                "batch_size": batch_size
            }
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
            
        return job
    
    def evaluate_model(self, model_id: str, test_messages: List[Dict]) -> Dict[str, Any]:
        """
        ファインチューニングしたモデルを評価
        
        Args:
            model_id: モデルID
            test_messages: テストメッセージ
            
        Returns:
            評価結果
        """
        logger.info(f"モデル評価中: {model_id}")
        
        results = {
            "model_id": model_id,
            "test_cases": len(test_messages),
            "responses": []
        }
        
        for i, messages in enumerate(test_messages[:10]):  # 最初の10ケースのみテスト
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
                
                logger.info(f"テストケース {i + 1}/10 完了")
                
            except Exception as e:
                logger.error(f"テストケース {i + 1} でエラー: {e}")
                results["responses"].append({
                    "test_case": i + 1,
                    "error": str(e)
                })
        
        return results
    
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
    
    def run_full_pipeline(self, epochs: int = 10, batch_size: int = 16, max_samples: Optional[int] = None):
        """
        フルパイプラインを実行
        
        Args:
            epochs: エポック数
            batch_size: バッチサイズ
            max_samples: 最大サンプル数（デバッグ用）
        """
        logger.info("=== OpenAI GPT-4o mini Supervised Fine Tuning 開始 ===")
        
        try:
            # 1. データ準備
            training_data = self.prepare_dataset(max_samples=max_samples)
            
            # 2. トレーニングデータ保存
            training_file = self.save_training_data(training_data)
            
            # 3. ファイルアップロード
            file_id = self.upload_training_file(training_file)
            
            # 4. ファインチューニングジョブ作成
            job_id = self.create_fine_tune_job(file_id, epochs=epochs, batch_size=batch_size)
            
            # 5. ジョブ監視
            job_result = self.monitor_fine_tune_job(job_id)
            
            if job_result.status == 'succeeded':
                model_id = job_result.fine_tuned_model
                
                # 6. モデル評価
                test_messages = training_data[:20]  # 最初の20サンプルをテスト用に使用
                test_messages = [item["messages"] for item in test_messages]
                evaluation_results = self.evaluate_model(model_id, test_messages)
                
                # 7. 結果保存
                self.save_results(evaluation_results)
                
                logger.info("=== ファインチューニング完了 ===")
                logger.info(f"モデルID: {model_id}")
                
            else:
                logger.error("ファインチューニングが失敗しました")
                
        except Exception as e:
            logger.error(f"パイプライン実行中にエラー: {e}")
            raise

def main():
    """メイン関数"""
    try:
        # SFTインスタンス作成（.envファイルから自動的にAPIキーを読み込み）
        sft = OpenAISFT()
        
        # フルパイプライン実行
        # デバッグ用に100サンプルに制限（実際の運用では max_samples=None にする）
        sft.run_full_pipeline(
            epochs=10,
            batch_size=16,
            max_samples=100  # デバッグ用、実際の運用では None
        )
    except ValueError as e:
        logger.error(f"設定エラー: {e}")
        logger.error("プロジェクトルートに.envファイルを作成し、OPENAI_API_KEY=your-api-key-here を設定してください")
    except Exception as e:
        logger.error(f"実行エラー: {e}")
        raise

if __name__ == "__main__":
    main()
