# -*- coding: utf-8 -*-
# turn_segmentation.py

import json
try:
    from typing import Dict, List, Tuple
except ImportError:
    # 古いPythonバージョン用のフォールバック
    Dict = dict
    List = list
    Tuple = tuple

def combine_utterances(turns):
    """同じroleの連続する複数発話を一つの文字列に結合する。"""
    if not turns:
        return ""
    return " ".join(turn['utterance'] for turn in turns)

def segment_turns(dialogue):
    """
    対話を役割(role)ごとに連続した発話のグループに分割する。
    """
    if not dialogue:
        return [], []

    counselor_groups = []
    client_groups = []
    
    current_role = None
    current_group = []
    
    for turn in dialogue:
        role = turn.get('role')
        if not role:
            continue

        if role != current_role:
            # 役割が変わった時、それまでのグループを保存
            if current_group:
                if current_role == 'counselor':
                    counselor_groups.append(current_group)
                elif current_role == 'client':
                    client_groups.append(current_group)
            
            # 新しいグループを開始
            current_role = role
            current_group = [turn]
        else:
            # 同じ役割が続く場合、現在のグループに追加
            current_group.append(turn)
    
    # 最後のグループを保存
    if current_group:
        if current_role == 'counselor':
            counselor_groups.append(current_group)
        elif current_role == 'client':
            client_groups.append(current_group)
            
    return counselor_groups, client_groups

def create_turn_list(dialogue):
    """
    対話リストを「役割: 発言」の形式の文字列リストに変換する。
    連続する同じ役割の発言は一つにまとめる。
    """
    if not dialogue:
        return []

    # 対話データの各要素の構造を確認し、標準化
    normalized_dialogue = []
    for turn in dialogue:
        if not isinstance(turn, dict):
            continue
            
        # 様々なキー名に対応
        role = None
        utterance = None
        
        # role の取得
        if 'role' in turn:
            role = turn['role']
        elif 'speaker' in turn:
            role = turn['speaker']
        elif 'who' in turn:
            role = turn['who']
            
        # utterance の取得
        if 'utterance' in turn:
            utterance = turn['utterance']
        elif 'text' in turn:
            utterance = turn['text']
        elif 'content' in turn:
            utterance = turn['content']
        elif 'message' in turn:
            utterance = turn['message']
            
        if role and utterance:
            normalized_dialogue.append({'role': role, 'utterance': utterance})

    if not normalized_dialogue:
        return []

    turn_text_list = []
    
    counselor_groups, client_groups = segment_turns(normalized_dialogue)
    
    # どちらの役割が先に発言したかを判定
    first_role = normalized_dialogue[0].get('role') if normalized_dialogue else None
    
    max_len = max(len(counselor_groups), len(client_groups))
    
    for i in range(max_len):
        # 最初の発言者の役割に基づいて交互に追加
        if first_role == 'counselor':
            if i < len(counselor_groups):
                turn_text_list.append("counselor: {}".format(combine_utterances(counselor_groups[i])))
            if i < len(client_groups):
                turn_text_list.append("client: {}".format(combine_utterances(client_groups[i])))
        else: # clientが先の場合
            if i < len(client_groups):
                turn_text_list.append("client: {}".format(combine_utterances(client_groups[i])))
            if i < len(counselor_groups):
                turn_text_list.append("counselor: {}".format(combine_utterances(counselor_groups[i])))

    return turn_text_list

if __name__ == '__main__':
    # このスクリプトを直接実行した際のテスト用動作
    sample_dialogue = [
        {'role': 'client', 'utterance': 'こんにちは。'},
        {'role': 'counselor', 'utterance': 'こんにちは、どうされましたか？'},
        {'role': 'counselor', 'utterance': '何でも話してくださいね。'},
        {'role': 'client', 'utterance': '実は、最近仕事のことで悩んでいて...'},
    ]
    
    turn_list = create_turn_list(sample_dialogue)
    print("--- ターン分割結果 ---")
    for turn in turn_list:
        print(turn)