#!/usr/bin/env python3
"""
進行状況監視・表示クラス

macOSのDockアイコンとターミナルの両方で進行状況を表示します。
"""

import os
import sys
import time
import threading
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import logging

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

try:
    from rich.console import Console
    from rich.progress import Progress, TaskID, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
    from rich.live import Live
    from rich.panel import Panel
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

logger = logging.getLogger(__name__)

class ProgressMonitor:
    """
    進行状況監視・表示クラス
    
    macOSのDockアイコンとターミナルの両方で進行状況を表示します。
    """
    
    def __init__(self, use_rich: bool = True):
        """
        初期化
        
        Args:
            use_rich: Rich ライブラリを使用するか
        """
        self.use_rich = use_rich and RICH_AVAILABLE
        self.console = Console() if self.use_rich else None
        self.progress = None
        self.live = None
        self.tasks: Dict[str, TaskID] = {}
        self.start_time = None
        self.current_phase = ""
        self.total_phases = 0
        self.completed_phases = 0
        
        # macOS Dock進行状況表示用
        self.dock_progress = 0.0
        self.dock_update_thread = None
        self.dock_running = False
        
        # 進行状況データ
        self.phase_info = {
            "data_loading": {"name": "データ読み込み", "weight": 5},
            "data_preparation": {"name": "データ変換・分割", "weight": 15},
            "file_upload": {"name": "ファイルアップロード", "weight": 10},
            "fine_tuning": {"name": "ファインチューニング", "weight": 60},
            "evaluation": {"name": "モデル評価", "weight": 8},
            "results_saving": {"name": "結果保存", "weight": 2}
        }
        
    def start_monitoring(self, total_phases: int = 6):
        """
        監視開始
        
        Args:
            total_phases: 総フェーズ数
        """
        self.start_time = datetime.now()
        self.total_phases = total_phases
        self.completed_phases = 0
        
        if self.use_rich:
            self.progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=self.console
            )
            self.live = Live(self._create_progress_panel(), refresh_per_second=2)
            self.live.start()
        
        # Dock進行状況表示開始
        self._start_dock_progress()
        
        logger.info("進行状況監視を開始しました")
    
    def stop_monitoring(self):
        """監視停止"""
        if self.live:
            self.live.stop()
            self.live = None
        
        if self.progress:
            self.progress = None
        
        self._stop_dock_progress()
        
        logger.info("進行状況監視を停止しました")
    
    def start_phase(self, phase_name: str, description: str = None, total: int = 100):
        """
        フェーズ開始
        
        Args:
            phase_name: フェーズ名
            description: 表示用説明
            total: 総ステップ数
        """
        self.current_phase = phase_name
        
        if description is None:
            description = self.phase_info.get(phase_name, {}).get("name", phase_name)
        
        if self.use_rich and self.progress:
            task_id = self.progress.add_task(description, total=total)
            self.tasks[phase_name] = task_id
            
            if self.live:
                self.live.update(self._create_progress_panel())
        else:
            logger.info(f"=== {description} 開始 ===")
    
    def update_phase(self, phase_name: str, completed: int, description: str = None):
        """
        フェーズ進行状況更新
        
        Args:
            phase_name: フェーズ名
            completed: 完了ステップ数
            description: 更新する説明
        """
        if self.use_rich and self.progress and phase_name in self.tasks:
            task_id = self.tasks[phase_name]
            update_kwargs = {"completed": completed}
            if description:
                update_kwargs["description"] = description
            self.progress.update(task_id, **update_kwargs)
            
            if self.live:
                self.live.update(self._create_progress_panel())
        else:
            if description:
                logger.info(f"[{phase_name}] {description} ({completed})")
        
        # 全体進行状況を更新
        self._update_overall_progress()
    
    def complete_phase(self, phase_name: str):
        """
        フェーズ完了
        
        Args:
            phase_name: フェーズ名
        """
        if self.use_rich and self.progress and phase_name in self.tasks:
            task_id = self.tasks[phase_name]
            self.progress.update(task_id, completed=100)
        
        self.completed_phases += 1
        self._update_overall_progress()
        
        phase_info = self.phase_info.get(phase_name, {})
        logger.info(f"✅ {phase_info.get('name', phase_name)} 完了")
    
    def add_log(self, message: str, level: str = "info"):
        """
        ログメッセージ追加
        
        Args:
            message: ログメッセージ
            level: ログレベル
        """
        if level == "error":
            logger.error(message)
        elif level == "warning":
            logger.warning(message)
        else:
            logger.info(message)
    
    def _create_progress_panel(self):
        """進行状況パネル作成"""
        if not self.use_rich or not self.progress:
            return ""
        
        # 経過時間計算
        elapsed = datetime.now() - self.start_time if self.start_time else timedelta(0)
        elapsed_str = str(elapsed).split('.')[0]  # ミリ秒を除去
        
        # 全体進行状況
        overall_progress = (self.completed_phases / self.total_phases * 100) if self.total_phases > 0 else 0
        
        # パネル作成
        title = Text("🚀 OpenAI SFT 実行状況", style="bold blue")
        
        content = [
            Text(f"経過時間: {elapsed_str}", style="cyan"),
            Text(f"全体進行: {overall_progress:.1f}% ({self.completed_phases}/{self.total_phases} フェーズ完了)", style="green"),
            Text(f"現在のフェーズ: {self.phase_info.get(self.current_phase, {}).get('name', self.current_phase)}", style="yellow"),
            Text(""),
            self.progress
        ]
        
        return Panel.fit(
            "\n".join([str(item) for item in content]),
            title=title,
            border_style="blue"
        )
    
    def _update_overall_progress(self):
        """全体進行状況を更新"""
        # フェーズ重み付き進行状況計算
        total_weight = sum(info["weight"] for info in self.phase_info.values())
        weighted_progress = 0.0
        
        for phase_name, info in self.phase_info.items():
            if phase_name in self.tasks and self.use_rich and self.progress:
                task = self.progress.tasks[self.tasks[phase_name]]
                phase_progress = (task.completed / task.total) if task.total > 0 else 0
                weighted_progress += (phase_progress * info["weight"]) / total_weight
        
        self.dock_progress = min(weighted_progress, 1.0)
    
    def _start_dock_progress(self):
        """macOS Dock進行状況表示開始"""
        if sys.platform != "darwin":  # macOSでない場合はスキップ
            return
        
        self.dock_running = True
        self.dock_update_thread = threading.Thread(target=self._dock_progress_loop, daemon=True)
        self.dock_update_thread.start()
    
    def _stop_dock_progress(self):
        """macOS Dock進行状況表示停止"""
        self.dock_running = False
        if self.dock_update_thread:
            self.dock_update_thread.join(timeout=1.0)
        
        # 進行状況をリセット
        if sys.platform == "darwin":
            try:
                os.system("defaults delete com.apple.dock progress 2>/dev/null")
                os.system("killall Dock 2>/dev/null")
            except:
                pass
    
    def _dock_progress_loop(self):
        """Dock進行状況更新ループ"""
        while self.dock_running:
            try:
                if sys.platform == "darwin":
                    # macOSのDockアイコンに進行状況を表示
                    progress_value = int(self.dock_progress * 100)
                    os.system(f"defaults write com.apple.dock progress -int {progress_value} 2>/dev/null")
                    
                    # Dockを更新（頻繁すぎると重くなるので10秒間隔）
                    if progress_value % 10 == 0:  # 10%刻みでのみDock更新
                        os.system("killall Dock 2>/dev/null")
                
                time.sleep(5)  # 5秒間隔で更新
            except Exception as e:
                logger.debug(f"Dock進行状況更新エラー: {e}")
                break

class SimpleProgressMonitor:
    """
    シンプルな進行状況監視クラス（依存ライブラリなし）
    """
    
    def __init__(self):
        self.start_time = None
        self.current_phase = ""
        self.completed_phases = 0
        self.total_phases = 0
    
    def start_monitoring(self, total_phases: int = 6):
        """監視開始"""
        self.start_time = datetime.now()
        self.total_phases = total_phases
        self.completed_phases = 0
        logger.info("🚀 SFT実行開始")
    
    def stop_monitoring(self):
        """監視停止"""
        if self.start_time:
            elapsed = datetime.now() - self.start_time
            logger.info(f"🎉 SFT実行完了 (総時間: {elapsed})")
    
    def start_phase(self, phase_name: str, description: str = None, total: int = 100):
        """フェーズ開始"""
        self.current_phase = phase_name
        logger.info(f"=== {description or phase_name} 開始 ===")
    
    def update_phase(self, phase_name: str, completed: int, description: str = None):
        """フェーズ進行状況更新"""
        if description:
            logger.info(f"[{phase_name}] {description}")
    
    def complete_phase(self, phase_name: str):
        """フェーズ完了"""
        self.completed_phases += 1
        progress = (self.completed_phases / self.total_phases * 100) if self.total_phases > 0 else 0
        logger.info(f"✅ {phase_name} 完了 (全体: {progress:.1f}%)")
    
    def add_log(self, message: str, level: str = "info"):
        """ログメッセージ追加"""
        if level == "error":
            logger.error(message)
        elif level == "warning":
            logger.warning(message)
        else:
            logger.info(message)

def create_progress_monitor(use_rich: bool = None) -> ProgressMonitor:
    """
    進行状況監視インスタンス作成
    
    Args:
        use_rich: Rich ライブラリを使用するか（Noneの場合は自動判定）
        
    Returns:
        進行状況監視インスタンス
    """
    if use_rich is None:
        use_rich = RICH_AVAILABLE
    
    if use_rich and RICH_AVAILABLE:
        return ProgressMonitor(use_rich=True)
    else:
        return SimpleProgressMonitor()
