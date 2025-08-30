# -*- coding: utf-8 -*-
# experiment_tracker.py - 実験管理ツールの統合

import os
import logging
from datetime import datetime
from experiment_config import ExperimentConfig

logger = logging.getLogger(__name__)

class ExperimentTracker:
    """TensorBoardとW&Bを統合した実験管理クラス"""
    
    def __init__(self, config=None):
        self.config = config or ExperimentConfig()
        self.tensorboard_writer = None
        self.wandb_run = None
        self.step_count = 0
        
        # 実験名の生成
        if not self.config.experiment_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.config.experiment_name = "emotion_sft_{}".format(timestamp)
        
        self._setup_tracking()
    
    def _setup_tracking(self):
        """実験管理ツールのセットアップ"""
        tracking_tool = self.config.tracking_tool.lower()
        
        if tracking_tool in ["tensorboard", "both"]:
            self._setup_tensorboard()
        
        if tracking_tool in ["wandb", "both"]:
            self._setup_wandb()
        
        if tracking_tool == "none":
            logger.info("実験管理ツールは使用しません")
    
    def _setup_tensorboard(self):
        """TensorBoardのセットアップ"""
        try:
            from torch.utils.tensorboard import SummaryWriter
            
            # ログディレクトリの作成
            log_dir = os.path.join(self.config.tensorboard_log_dir, self.config.experiment_name)
            os.makedirs(log_dir, exist_ok=True)
            
            self.tensorboard_writer = SummaryWriter(log_dir)
            logger.info("TensorBoard writer initialized: {}".format(log_dir))
            
        except ImportError:
            logger.error("TensorBoardが利用できません。pip install tensorboard を実行してください。")
            self.tensorboard_writer = None
        except Exception as e:
            logger.error("TensorBoard初期化エラー: {}".format(e))
            self.tensorboard_writer = None
    
    def _setup_wandb(self):
        """W&Bのセットアップ"""
        try:
            import wandb
            
            # W&Bの初期化
            self.wandb_run = wandb.init(
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                name=self.config.experiment_name,
                tags=self.config.wandb_tags,
                config=self.config.get_config_dict()
            )
            logger.info("W&B run initialized: {}".format(self.config.experiment_name))
            
        except ImportError:
            logger.error("W&Bが利用できません。pip install wandb を実行してください。")
            self.wandb_run = None
        except Exception as e:
            logger.error("W&B初期化エラー: {}".format(e))
            self.wandb_run = None
    
    def log_hyperparameters(self, hyperparams):
        """ハイパーパラメータの記録"""
        if not self.config.track_hyperparameters:
            return
        
        if self.tensorboard_writer:
            try:
                # TensorBoard用にハイパーパラメータを適切な型に変換
                tb_hyperparams = {}
                for key, value in hyperparams.items():
                    if isinstance(value, (int, float, str, bool)):
                        tb_hyperparams[key] = value
                    elif isinstance(value, list):
                        # リストは文字列に変換
                        tb_hyperparams[key] = str(value)
                    elif value is None:
                        tb_hyperparams[key] = "None"
                    else:
                        # その他の複雑な型は文字列に変換
                        tb_hyperparams[key] = str(value)
                
                # TensorBoardにハイパーパラメータを記録
                self.tensorboard_writer.add_hparams(
                    tb_hyperparams, 
                    {"hp_metric": 0}  # ダミーメトリック
                )
                logger.info("ハイパーパラメータをTensorBoardに記録しました")
            except Exception as e:
                logger.error("TensorBoardハイパーパラメータ記録エラー: {}".format(e))
                # フォールバック: テキストとして記録
                try:
                    param_text = "\n".join(["{}: {}".format(k, v) for k, v in hyperparams.items()])
                    self.tensorboard_writer.add_text("hyperparameters", param_text)
                    logger.info("ハイパーパラメータをTensorBoardにテキストとして記録しました")
                except Exception as e2:
                    logger.error("TensorBoardテキスト記録も失敗: {}".format(e2))
        
        if self.wandb_run:
            try:
                self.wandb_run.config.update(hyperparams)
                logger.info("ハイパーパラメータをW&Bに記録しました")
            except Exception as e:
                logger.error("W&Bハイパーパラメータ記録エラー: {}".format(e))
    
    def log_metrics(self, metrics, step=None):
        """メトリクスの記録"""
        if step is None:
            step = self.step_count
            self.step_count += 1
        
        # フィルタリング: 設定されたメトリクスのみ記録
        filtered_metrics = {}
        for key, value in metrics.items():
            if key in self.config.metrics_to_track or "loss" in key.lower():
                filtered_metrics[key] = value
        
        if self.tensorboard_writer:
            try:
                for key, value in filtered_metrics.items():
                    self.tensorboard_writer.add_scalar(key, value, step)
                self.tensorboard_writer.flush()
            except Exception as e:
                logger.error("TensorBoardメトリクス記録エラー: {}".format(e))
        
        if self.wandb_run:
            try:
                log_dict = filtered_metrics.copy()
                log_dict["step"] = step
                self.wandb_run.log(log_dict)
            except Exception as e:
                logger.error("W&Bメトリクス記録エラー: {}".format(e))
    
    def log_model_artifact(self, model_path, artifact_name=None):
        """モデルアーティファクトの保存（W&B）"""
        if not self.config.save_model_artifacts or not self.wandb_run:
            return
        
        try:
            import wandb
            
            if artifact_name is None:
                artifact_name = "model_{}".format(self.config.experiment_name)
            
            # アーティファクトの作成
            artifact = wandb.Artifact(
                artifact_name, 
                type="model",
                description="Emotion reward SFT model"
            )
            artifact.add_dir(model_path)
            self.wandb_run.log_artifact(artifact)
            
            logger.info("モデルアーティファクトをW&Bに保存しました: {}".format(artifact_name))
            
        except Exception as e:
            logger.error("W&Bアーティファクト保存エラー: {}".format(e))
    
    def log_text(self, text, title="log"):
        """テキストログの記録"""
        if self.wandb_run:
            try:
                self.wandb_run.log({"{}".format(title): text})
            except Exception as e:
                logger.error("W&Bテキストログエラー: {}".format(e))
    
    def finish(self):
        """実験の終了処理"""
        if self.tensorboard_writer:
            try:
                self.tensorboard_writer.close()
                logger.info("TensorBoard writer closed")
            except Exception as e:
                logger.error("TensorBoard終了エラー: {}".format(e))
        
        if self.wandb_run:
            try:
                self.wandb_run.finish()
                logger.info("W&B run finished")
            except Exception as e:
                logger.error("W&B終了エラー: {}".format(e))
    
    def __enter__(self):
        """コンテキストマネージャーのエントリ"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """コンテキストマネージャーの終了"""
        self.finish()

# 使用例とヘルパー関数
def create_experiment_tracker(tracking_tool="both", project_name="emotion_reward_sft"):
    """実験トラッカーの簡単な作成"""
    config = ExperimentConfig()
    config.tracking_tool = tracking_tool
    config.project_name = project_name
    return ExperimentTracker(config)
