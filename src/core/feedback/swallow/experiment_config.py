# -*- coding: utf-8 -*-
# experiment_config.py - 実験管理の設定

import os
import logging

logger = logging.getLogger(__name__)

class ExperimentConfig:
    """実験管理の設定クラス"""
    
    def __init__(self):
        # 実験管理ツールの選択 ("tensorboard", "wandb", "both", "none")
        self.tracking_tool = "both"
        
        # プロジェクト名
        self.project_name = "emotion_reward_sft"
        
        # 実験名（自動生成される場合はNone）
        self.experiment_name = None
        
        # TensorBoard設定
        self.tensorboard_log_dir = "./logs_tensorboard"
        
        # W&B設定
        self.wandb_project = "emotion-reward-sft"
        self.wandb_entity = None  # チーム名（個人の場合はNone）
        self.wandb_tags = ["emotion", "reward", "sft", "japanese"]
        
        # 保存する指標
        self.metrics_to_track = [
            "train_loss",
            "learning_rate", 
            "epoch",
            "step",
            "mse_loss",
            "gradient_norm"
        ]
        
        # ハイパーパラメータの記録
        self.track_hyperparameters = True
        
        # モデルの保存
        self.save_model_artifacts = True
        
        # 環境変数からの設定読み込み
        self._load_from_env()
    
    def _load_from_env(self):
        """環境変数から設定を読み込む"""
        self.tracking_tool = os.getenv("EXPERIMENT_TRACKING_TOOL", self.tracking_tool)
        self.wandb_project = os.getenv("WANDB_PROJECT", self.wandb_project)
        self.wandb_entity = os.getenv("WANDB_ENTITY", self.wandb_entity)
        
        # W&B APIキーの確認
        if self.tracking_tool in ["wandb", "both"]:
            wandb_api_key = os.getenv("WANDB_API_KEY")
            if not wandb_api_key:
                logger.warning("WANDB_API_KEY環境変数が設定されていません。W&Bを使用する場合は設定してください。")
    
    def get_config_dict(self):
        """設定を辞書形式で取得"""
        return {
            "tracking_tool": self.tracking_tool,
            "project_name": self.project_name,
            "experiment_name": self.experiment_name,
            "tensorboard_log_dir": self.tensorboard_log_dir,
            "wandb_project": self.wandb_project,
            "wandb_entity": self.wandb_entity,
            "wandb_tags": self.wandb_tags,
            "metrics_to_track": self.metrics_to_track,
            "track_hyperparameters": self.track_hyperparameters,
            "save_model_artifacts": self.save_model_artifacts
        }

# デフォルト設定のインスタンス
default_config = ExperimentConfig()
