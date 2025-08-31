#!/usr/bin/env python3
"""
失敗したファインチューニングジョブの詳細を分析し、問題を回避するスクリプト
"""

import os
import json
import logging
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from datetime import datetime
from typing import List, Dict, Any, Tuple

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_json_format(data: List[Dict[str, Any]]) -> Tuple[bool, List[int], List[str]]:
    """
    JSONデータの形式を検証
    
    Args:
        data: 検証するデータリスト
        
    Returns:
        (is_valid, invalid_indices, error_messages)
    """
    is_valid = True
    invalid_indices = []
    error_messages = []
    
    for i, item in enumerate(data):
        try:
            # 必須フィールドの存在確認
            if not isinstance(item, dict):
                raise ValueError(f"行 {i+1}: 辞書形式ではありません")
            
            if 'messages' not in item:
                raise ValueError(f"行 {i+1}: 'messages'フィールドがありません")
            
            if not isinstance(item['messages'], list):
                raise ValueError(f"行 {i+1}: 'messages'がリスト形式ではありません")
            
            # メッセージの形式確認
            for j, message in enumerate(item['messages']):
                if not isinstance(message, dict):
                    raise ValueError(f"行 {i+1}, メッセージ {j+1}: 辞書形式ではありません")
                
                if 'role' not in message or 'content' not in message:
                    raise ValueError(f"行 {i+1}, メッセージ {j+1}: 'role'または'content'フィールドがありません")
                
                if not isinstance(message['role'], str) or not isinstance(message['content'], str):
                    raise ValueError(f"行 {i+1}, メッセージ {j+1}: 'role'または'content'が文字列ではありません")
            
            # JSONとしてシリアライズ可能かテスト
            json.dumps(item, ensure_ascii=False)
            
        except Exception as e:
            is_valid = False
            invalid_indices.append(i)
            error_messages.append(str(e))
    
    return is_valid, invalid_indices, error_messages

def fix_json_data(data: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    """
    破損したJSONデータを修復
    
    Args:
        data: 修復するデータリスト
        
    Returns:
        (修復されたデータ, 修復された行数)
    """
    fixed_data = []
    fixed_count = 0
    
    for i, item in enumerate(data):
        try:
            # 基本的な形式チェック
            if not isinstance(item, dict):
                logger.warning(f"行 {i+1}: 辞書形式ではないためスキップ")
                continue
            
            # 必須フィールドの補完
            if 'messages' not in item:
                logger.warning(f"行 {i+1}: 'messages'フィールドを空のリストで補完")
                item['messages'] = []
            
            # メッセージの修復
            if isinstance(item['messages'], list):
                fixed_messages = []
                for j, message in enumerate(item['messages']):
                    if isinstance(message, dict) and 'role' in message and 'content' in message:
                        if isinstance(message['role'], str) and isinstance(message['content'], str):
                            fixed_messages.append(message)
                        else:
                            # 型を修正
                            fixed_message = {
                                'role': str(message.get('role', 'user')),
                                'content': str(message.get('content', ''))
                            }
                            fixed_messages.append(fixed_message)
                            fixed_count += 1
                    else:
                        # 無効なメッセージをスキップ
                        logger.warning(f"行 {i+1}, メッセージ {j+1}: 無効なメッセージをスキップ")
                
                item['messages'] = fixed_messages
            else:
                item['messages'] = []
                fixed_count += 1
            
            # JSONとしてシリアライズ可能かテスト
            json.dumps(item, ensure_ascii=False)
            fixed_data.append(item)
            
        except Exception as e:
            logger.error(f"行 {i+1}: 修復不可能 - {e}")
            continue
    
    return fixed_data, fixed_count

def create_safe_training_file(input_file: Path, output_file: Path) -> Tuple[bool, int, int]:
    """
    安全なトレーニングファイルを作成
    
    Args:
        input_file: 入力ファイルパス
        output_file: 出力ファイルパス
        
    Returns:
        (成功フラグ, 総行数, 有効行数)
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # JSONデータを解析
        data = []
        for line in lines:
            try:
                item = json.loads(line.strip())
                data.append(item)
            except json.JSONDecodeError as e:
                logger.warning(f"JSON解析エラー: {e}")
                continue
        
        if not data:
            logger.error("有効なデータが見つかりません")
            return False, 0, 0
        
        # データ形式を検証
        is_valid, invalid_indices, error_messages = validate_json_format(data)
        
        if not is_valid:
            logger.warning(f"データ形式に問題があります: {len(invalid_indices)}行")
            for idx, error in zip(invalid_indices, error_messages):
                logger.warning(f"  行 {idx+1}: {error}")
            
            # データを修復
            logger.info("データの修復を試行中...")
            fixed_data, fixed_count = fix_json_data(data)
            
            if not fixed_data:
                logger.error("データの修復に失敗しました")
                return False, len(data), 0
            
            logger.info(f"修復完了: {fixed_count}行を修復")
            data = fixed_data
        
        # 安全なファイルを作成
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in data:
                json.dump(item, f, ensure_ascii=False)
                f.write('\n')
        
        logger.info(f"安全なトレーニングファイルを作成: {output_file}")
        return True, len(lines), len(data)
        
    except Exception as e:
        logger.error(f"ファイル作成エラー: {e}")
        return False, 0, 0

def create_openai_sft_safe_file(input_file: Path, output_file: Path) -> Tuple[bool, str]:
    """
    OpenAI SFT用の安全なトレーニングファイルを作成
    
    Args:
        input_file: 入力ファイルパス
        output_file: 出力ファイルパス
        
    Returns:
        (成功フラグ, エラーメッセージ)
    """
    try:
        # 安全なファイルを作成
        success, total_lines, valid_lines = create_safe_training_file(input_file, output_file)
        
        if not success:
            return False, "安全なファイルの作成に失敗しました"
        
        # ファイルサイズの確認（OpenAI制限: 100MB以下）
        file_size_mb = output_file.stat().st_size / (1024 * 1024)
        if file_size_mb > 100:
            return False, f"ファイルサイズが大きすぎます: {file_size_mb:.2f}MB (制限: 100MB)"
        
        # 行数の確認（最小1行）
        if valid_lines < 1:
            return False, "有効なデータが1行もありません"
        
        logger.info(f"OpenAI SFT用の安全なファイルを作成: {output_file}")
        logger.info(f"  ファイルサイズ: {file_size_mb:.2f}MB")
        logger.info(f"  有効行数: {valid_lines}")
        
        return True, ""
        
    except Exception as e:
        error_msg = f"ファイル作成エラー: {e}"
        logger.error(error_msg)
        return False, error_msg

def validate_openai_sft_format(file_path: Path) -> Tuple[bool, List[str]]:
    """
    OpenAI SFT形式の検証
    
    Args:
        file_path: 検証するファイルパス
        
    Returns:
        (有効フラグ, エラーメッセージリスト)
    """
    errors = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        if not lines:
            errors.append("ファイルが空です")
            return False, errors
        
        # 各行を検証
        for i, line in enumerate(lines):
            try:
                data = json.loads(line.strip())
                
                # 必須フィールドの確認
                if not isinstance(data, dict):
                    errors.append(f"行 {i+1}: 辞書形式ではありません")
                    continue
                
                if 'messages' not in data:
                    errors.append(f"行 {i+1}: 'messages'フィールドがありません")
                    continue
                
                if not isinstance(data['messages'], list):
                    errors.append(f"行 {i+1}: 'messages'がリスト形式ではありません")
                    continue
                
                if len(data['messages']) == 0:
                    errors.append(f"行 {i+1}: 'messages'が空です")
                    continue
                
                # メッセージの形式確認
                for j, message in enumerate(data['messages']):
                    if not isinstance(message, dict):
                        errors.append(f"行 {i+1}, メッセージ {j+1}: 辞書形式ではありません")
                        continue
                    
                    if 'role' not in message:
                        errors.append(f"行 {i+1}, メッセージ {j+1}: 'role'フィールドがありません")
                        continue
                    
                    if 'content' not in message:
                        errors.append(f"行 {i+1}, メッセージ {j+1}: 'content'フィールドがありません")
                        continue
                    
                    if not isinstance(message['role'], str):
                        errors.append(f"行 {i+1}, メッセージ {j+1}: 'role'が文字列ではありません")
                        continue
                    
                    if not isinstance(message['content'], str):
                        errors.append(f"行 {i+1}, メッセージ {j+1}: 'content'が文字列ではありません")
                        continue
                    
                    if message['role'] not in ['system', 'user', 'assistant']:
                        errors.append(f"行 {i+1}, メッセージ {j+1}: 無効な'role'値: {message['role']}")
                        continue
                    
                    if len(message['content'].strip()) == 0:
                        errors.append(f"行 {i+1}, メッセージ {j+1}: 'content'が空です")
                        continue
                
            except json.JSONDecodeError as e:
                errors.append(f"行 {i+1}: JSON解析エラー - {e}")
                continue
        
        return len(errors) == 0, errors
        
    except Exception as e:
        errors.append(f"ファイル読み込みエラー: {e}")
        return False, errors

def get_failed_job_info(job_id: str, api_key: str):
    """
    失敗したファインチューニングジョブの詳細情報を取得
    
    Args:
        job_id: ファインチューニングジョブID
        api_key: OpenAI APIキー
        
    Returns:
        ジョブ情報
    """
    client = OpenAI(api_key=api_key)
    
    try:
        job = client.fine_tuning.jobs.retrieve(job_id)
        logger.info(f"ジョブ情報を取得: {job_id}")
        return job
    except Exception as e:
        logger.error(f"ジョブ情報の取得に失敗: {e}")
        return None

def analyze_failure_cause(job_info):
    """
    失敗の原因を分析
    
    Args:
        job_info: ジョブ情報
    """
    print(f"\n=== 失敗原因の分析 ===")
    
    if hasattr(job_info, 'error') and job_info.error:
        print(f"エラー詳細: {job_info.error}")
    
    if hasattr(job_info, 'status') and job_info.status == 'failed':
        print(f"ステータス: {job_info.status}")
        
        # 失敗の一般的な原因を分析
        if hasattr(job_info, 'training_file'):
            print(f"トレーニングファイル: {job_info.training_file}")
        
        if hasattr(job_info, 'validation_file'):
            print(f"検証ファイル: {job_info.validation_file}")
        
        if hasattr(job_info, 'hyperparameters'):
            print(f"ハイパーパラメータ: {job_info.hyperparameters}")
        
        if hasattr(job_info, 'created_at'):
            created_time = datetime.fromtimestamp(job_info.created_at)
            print(f"作成日時: {created_time}")
        
        if hasattr(job_info, 'finished_at') and job_info.finished_at:
            finished_time = datetime.fromtimestamp(job_info.finished_at)
            print(f"完了日時: {finished_time}")
            duration = finished_time - created_time
            print(f"実行時間: {duration}")
        else:
            print("完了日時: 未完了")
    
    # ファイルサイズの分析
    analyze_file_sizes()
    
    # トレーニングファイルの内容を確認
    analyze_training_file_content()

def analyze_file_sizes():
    """
    トレーニングファイルのサイズを分析
    """
    print(f"\n=== ファイルサイズの分析 ===")
    
    output_dir = Path("openai_sft_outputs")
    if not output_dir.exists():
        print("出力ディレクトリが見つかりません")
        return
    
    # ファイルサイズを時系列で表示
    files = []
    for file_path in output_dir.glob("*.jsonl"):
        if file_path.name.startswith(("train_", "test_", "valid_")):
            stat = file_path.stat()
            created_time = datetime.fromtimestamp(stat.st_mtime)
            size_mb = stat.st_size / (1024 * 1024)
            files.append((created_time, file_path.name, size_mb))
    
    # 時系列順にソート
    files.sort(key=lambda x: x[0])
    
    print("ファイルサイズの推移:")
    for created_time, filename, size_mb in files:
        print(f"  {created_time.strftime('%H:%M:%S')} - {filename}: {size_mb:.2f} MB")

def analyze_training_file_content():
    """
    トレーニングファイルの内容を分析
    """
    print(f"\n=== トレーニングファイルの内容分析 ===")
    
    output_dir = Path("openai_sft_outputs")
    if not output_dir.exists():
        print("出力ディレクトリが見つかりません")
        return
    
    # 最新のトレーニングファイルを確認
    train_files = list(output_dir.glob("train_*.jsonl"))
    if not train_files:
        print("トレーニングファイルが見つかりません")
        return
    
    # 最新のファイルを取得
    latest_train_file = max(train_files, key=lambda x: x.stat().st_mtime)
    print(f"最新のトレーニングファイル: {latest_train_file.name}")
    
    try:
        with open(latest_train_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        print(f"総行数: {len(lines)}")
        
        # 最初の数行を確認
        print("\n最初の5行の内容:")
        for i, line in enumerate(lines[:5]):
            try:
                data = json.loads(line.strip())
                print(f"行 {i+1}: 有効なJSON ✓")
                if 'messages' in data:
                    print(f"  メッセージ数: {len(data['messages'])}")
            except json.JSONDecodeError as e:
                print(f"行 {i+1}: 無効なJSON ✗ - {e}")
                print(f"  内容: {line.strip()[:100]}...")
        
        # 無効な行を検出
        invalid_lines = []
        for i, line in enumerate(lines):
            try:
                json.loads(line.strip())
            except json.JSONDecodeError:
                invalid_lines.append(i + 1)
        
        if invalid_lines:
            print(f"\n⚠️  無効なJSON行: {len(invalid_lines)}行")
            print(f"  行番号: {invalid_lines[:10]}{'...' if len(invalid_lines) > 10 else ''}")
        else:
            print("\n✅ すべての行が有効なJSON形式です")
            
    except Exception as e:
        print(f"ファイル読み込みエラー: {e}")

def get_openai_sft_integration_code():
    """
    OpenAI SFTクラスに統合するためのコード例を表示
    """
    print("\n" + "="*60)
    print("🔧 OpenAI SFTクラス統合用コード")
    print("="*60)
    
    integration_code = '''
# OpenAI SFTクラスに以下のメソッドを追加してください

def create_safe_training_file(self, input_file: str, output_file: str = None) -> Tuple[bool, str]:
    """
    安全なトレーニングファイルを作成
    
    Args:
        input_file: 入力ファイルパス
        output_file: 出力ファイルパス（Noneの場合は自動生成）
        
    Returns:
        (成功フラグ, エラーメッセージ)
    """
    if output_file is None:
        input_path = Path(input_file)
        output_file = str(input_path.parent / f"safe_{input_path.name}")
    
    success, error_msg = create_openai_sft_safe_file(Path(input_file), Path(output_file))
    return success, error_msg

def validate_training_file(self, file_path: str) -> Tuple[bool, List[str]]:
    """
    トレーニングファイルの形式を検証
    
    Args:
        file_path: 検証するファイルパス
        
    Returns:
        (有効フラグ, エラーメッセージリスト)
    """
    return validate_openai_sft_format(Path(file_path))

def run_safe_fine_tuning(self, training_file: str, **kwargs) -> str:
    """
    安全なファインチューニングを実行
    
    Args:
        training_file: トレーニングファイルパス
        **kwargs: その他のパラメータ
        
    Returns:
        ジョブID
    """
    # ファイル形式を検証
    is_valid, errors = self.validate_training_file(training_file)
    if not is_valid:
        raise ValueError(f"トレーニングファイルの形式が無効です: {errors[:3]}")
    
    # 安全なファイルを作成
    safe_file = str(Path(training_file).parent / f"safe_{Path(training_file).name}")
    success, error_msg = self.create_safe_training_file(training_file, safe_file)
    
    if not success:
        raise ValueError(f"安全なファイルの作成に失敗: {error_msg}")
    
    # 安全なファイルを使用してファインチューニングを実行
    logger.info(f"安全なファイルを使用してファインチューニングを実行: {safe_file}")
    return self.create_fine_tune_job(safe_file, **kwargs)
'''
    
    print(integration_code)
    print("="*60)

def main():
    # プロジェクトルートの.envファイルを読み込み
    project_root = Path(__file__).parent.parent.parent.parent
    env_path = project_root / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        logger.info(f".envファイルを読み込みました: {env_path}")
    
    # APIキーの確認
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OpenAI APIキーが設定されていません")
        return
    
    # 失敗したジョブID
    failed_job_id = "ftjob-N6KpXmRlivMYE0sMCKNJqxVz"
    
    print("=== 失敗したファインチューニングジョブの分析 ===")
    print(f"ジョブID: {failed_job_id}")
    
    try:
        # ジョブ情報を取得
        job_info = get_failed_job_info(failed_job_id, api_key)
        if not job_info:
            return
        
        # 失敗原因を分析
        analyze_failure_cause(job_info)
        
        # 成功したジョブとの比較
        print(f"\n=== 成功したジョブとの比較 ===")
        print("成功したジョブ: ftjob-3FLRyCkixK8eFBaxzPNELbTl")
        print("失敗したジョブ: ftjob-N6KpXmRlivMYE0sMCKNJqxVz")
        
        # ファイルサイズの違いを強調
        print(f"\n💡 推測される失敗原因:")
        print("1. トレーニングデータが大きすぎる（7.3MB vs 0.8MB）")
        print("2. データ形式の問題（invalid_file_format）")
        print("3. OpenAIの制限に引っかかった")
        
        # 安全なトレーニングファイルの作成
        print(f"\n=== 安全なトレーニングファイルの作成 ===")
        output_dir = Path("openai_sft_outputs")
        if output_dir.exists():
            train_files = list(output_dir.glob("train_*.jsonl"))
            if train_files:
                latest_train_file = max(train_files, key=lambda x: x.stat().st_mtime)
                safe_file = output_dir / f"safe_{latest_train_file.name}"
                
                print(f"元ファイル: {latest_train_file.name}")
                print(f"安全ファイル: {safe_file.name}")
                
                success, total_lines, valid_lines = create_safe_training_file(latest_train_file, safe_file)
                
                if success:
                    print(f"✅ 安全なファイル作成成功")
                    print(f"  総行数: {total_lines}")
                    print(f"  有効行数: {valid_lines}")
                    print(f"  ファイルサイズ: {safe_file.stat().st_size / (1024 * 1024):.2f} MB")
                    
                    # ファイル内容の検証
                    print(f"\n=== 安全ファイルの検証 ===")
                    with open(safe_file, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    
                    data = []
                    for line in lines:
                        try:
                            item = json.loads(line.strip())
                            data.append(item)
                        except json.JSONDecodeError:
                            continue
                    
                    is_valid, invalid_indices, error_messages = validate_json_format(data)
                    
                    if is_valid:
                        print("✅ すべての行が有効なJSON形式です")
                    else:
                        print(f"⚠️  {len(invalid_indices)}行に問題があります")
                        for idx, error in zip(invalid_indices[:5], error_messages[:5]):
                            print(f"  行 {idx+1}: {error}")
                    
                    # OpenAI SFT形式の検証
                    print(f"\n=== OpenAI SFT形式の検証 ===")
                    sft_valid, sft_errors = validate_openai_sft_format(safe_file)
                    
                    if sft_valid:
                        print("✅ OpenAI SFT形式として有効です")
                    else:
                        print(f"⚠️  OpenAI SFT形式に問題があります: {len(sft_errors)}件")
                        for error in sft_errors[:5]:
                            print(f"  {error}")
                    
                    # 推奨事項の表示
                    print(f"\n💡 推奨事項:")
                    if sft_valid:
                        print("1. この安全ファイルを使用してSFTを実行してください")
                        print("2. 元ファイルは問題があるため使用しないでください")
                        print("3. 今後のデータ生成時は、このスクリプトで事前検証を行ってください")
                    else:
                        print("1. データ形式の問題を修正してから再実行してください")
                        print("2. メッセージのroleとcontentフィールドを確認してください")
                        print("3. 空のメッセージや無効な形式を除去してください")
                else:
                    print("❌ 安全なファイルの作成に失敗しました")
        
        # OpenAI SFTクラス統合用コードの表示
        get_openai_sft_integration_code()
        
    except Exception as e:
        logger.error(f"エラーが発生しました: {e}")

if __name__ == "__main__":
    main()
