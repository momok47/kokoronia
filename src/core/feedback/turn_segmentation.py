import json
import os
import sys
from typing import Dict, List, Tuple

def combine_utterances(turns: List[Dict]) -> str:
    """同じroleの複数発話を結合"""
    if not turns:
        return ""
    
    utterances = [turn['utterance'] for turn in turns]
    return " ".join(utterances)

def segment_turns(dialogue: List[Dict]) -> Tuple[List[List[Dict]], List[List[Dict]], int]:
    """
    会話をターンごとに分割する
    
    Args:
        dialogue: 会話データのリスト [{'role': 'counselor', 'utterance': '...'}, ...]
    
    Returns:
        counselor_turns: counselorの連続発話グループのリスト
        client_turns: clientの連続発話グループのリスト
        max_turns: 作成される論理ターン数
    """
    # 同じroleの連続発話をまとめる
    counselor_turns = []
    client_turns = []
    
    # 同じroleの連続発話をまとめる
    current_role = None
    current_turns = []
    
    for turn in dialogue:
        role = turn['role']
        
        if current_role is None:
            # 最初の発話
            current_role = role
            current_turns = [turn]
        elif role == current_role:
            # 同じroleの連続発話
            current_turns.append(turn)
        else:
            # roleが変わった場合、前のターンを処理
            if current_role == 'counselor':
                counselor_turns.append(current_turns)
            elif current_role == 'client':
                client_turns.append(current_turns)
            
            # 新しいroleを開始
            current_role = role
            current_turns = [turn]
    
    # 最後のターンを処理
    if current_turns:
        if current_role == 'counselor':
            counselor_turns.append(current_turns)
        elif current_role == 'client':
            client_turns.append(current_turns)
    
    max_turns = max(len(counselor_turns), len(client_turns))
    
    return counselor_turns, client_turns, max_turns

def create_turn_text(counselor_turns: List[List[Dict]], client_turns: List[List[Dict]], max_turns: int) -> str:
    """
    ターンテキストを作成する
    
    Args:
        counselor_turns: counselorの連続発話グループのリスト
        client_turns: clientの連続発話グループのリスト
        max_turns: 作成される論理ターン数
    
    Returns:
        ターンテキスト
    """
    conversation_text = ""
    
    for i in range(max_turns):
        conversation_text += f"--- ターン {i+1} ---\n"
        
        # counselorの連続発話グループを表示（存在する場合）
        if i < len(counselor_turns):
            counselor_group = counselor_turns[i]
            combined_counselor = combine_utterances(counselor_group)
            conversation_text += f"counselor: {combined_counselor}\n"
        
        # clientの連続発話グループを表示（存在する場合）
        if i < len(client_turns):
            client_group = client_turns[i]
            combined_client = combine_utterances(client_group)
            conversation_text += f"client: {combined_client}\n"
        
        conversation_text += "\n"
    
    return conversation_text

def create_turn_list(counselor_turns: List[List[Dict]], client_turns: List[List[Dict]], max_turns: int) -> List[List[Dict]]:
    """
    ターンごとの要素リストを作成する
    
    Args:
        counselor_turns: counselorの連続発話グループのリスト
        client_turns: clientの連続発話グループのリスト
        max_turns: 作成される論理ターン数
    
    Returns:
        ターンごとの要素リスト [ターン1, ターン2, ...]
    """
    turn_list = []
    
    for i in range(max_turns):
        current_turn = []
        
        # counselorの連続発話グループを追加（存在する場合）
        if i < len(counselor_turns):
            counselor_group = counselor_turns[i]
            combined_counselor = combine_utterances(counselor_group)
            current_turn.append({'role': 'counselor', 'utterance': combined_counselor})
        
        # clientの連続発話グループを追加（存在する場合）
        if i < len(client_turns):
            client_group = client_turns[i]
            combined_client = combine_utterances(client_group)
            current_turn.append({'role': 'client', 'utterance': combined_client})
        
        if current_turn:
            turn_list.append(current_turn)
    
    return turn_list

def analyze_turn_splitting_from_json(json_file_path: str) -> Dict:
    """
    JSONファイルからターン分割の分析結果を取得
    
    Args:
        json_file_path: JSONファイルのパス
    
    Returns:
        分析結果の辞書
    """
    # JSONファイルを読み込み
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    dialogue = data['dialogue']
    
    # ターン分割を実行
    counselor_turns, client_turns, max_turns = segment_turns(dialogue)
    
    # ターンテキストを作成
    conversation_text = create_turn_text(counselor_turns, client_turns, max_turns)
    
    # ターンリストを作成
    turn_list = create_turn_list(counselor_turns, client_turns, max_turns)
    
    # 結果を辞書で返す
    result = {
        'counselor_turns': counselor_turns,
        'client_turns': client_turns,
        'max_turns': max_turns,
        'conversation_text': conversation_text,
        'turn_list': turn_list,
        'counselor_groups_count': len(counselor_turns),
        'client_groups_count': len(client_turns)
    }
    
    return result

def display_turn_analysis(json_file_path: str):
    """
    JSONファイルからターン分割の分析結果を表示
    
    Args:
        json_file_path: JSONファイルのパス
    """
    result = analyze_turn_splitting_from_json(json_file_path)
    # print(result['conversation_text'])
    print(result['turn_list'])

if __name__ == "__main__":
    # コマンドライン引数からJSONファイルパスを取得
    if len(sys.argv) > 1:
        json_file_path = sys.argv[1]
    else:
        json_file_path = "test.json"  # デフォルト
    
    display_turn_analysis(json_file_path)