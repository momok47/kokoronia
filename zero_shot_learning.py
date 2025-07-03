import pandas as pd
import MeCab
from transformers import pipeline
import torch

class InterestsExtraction:
    """
    テキストデータから関心度を抽出するクラス
    
    Args:
        conversation_data (str or list): 会話データ（文字列または辞書のリスト）
        model_name (str): 使用するBERTモデル名
        topic_labels (list): トピックラベルのリスト
        mecab_dic_path (str, optional): MeCab辞書のパス
    """
    def __init__(self, conversation_data, model_name, topic_labels, mecab_dic_path=None):
        self.conversation_data = conversation_data
        self.model_name = model_name
        self.topic_labels = topic_labels

        self.classifier = pipeline(
            "zero-shot-classification",
            model=self.model_name,
            tokenizer=self.model_name,
            device=0 if torch.cuda.is_available() else -1
        )

        if mecab_dic_path:
            self.tagger = MeCab.Tagger(f'-d {mecab_dic_path} -r /dev/null')
        else:
            self.tagger = MeCab.Tagger()

        self.conv_text_list = []
        self.pred_score = []
        self.tagged_line = []
        self.calc_amount_of_speech = []

    # display_speaker_label 引数を追加
    def main_function(self, display_speaker_label=None):
        self.prepare_conversation_data()
        self.pred_label()
        self.morphological_analysis()
        self.calculate_morph_ratio()
        self.score_of_interest(display_speaker_label)

    def prepare_conversation_data(self):
        # jsonからのデータがfull_textの場合
        if isinstance(self.conversation_data, str):
            self.conv_text_list.append(f"話者/{ self.conversation_data}")
        # jsonからのデータがconversationの場合
        elif isinstance(self.conversation_data, list):
            for utterance in self.conversation_data:
                speaker = utterance.get("speaker", "不明な話者")
                message = utterance.get("text", "")
                self.conv_text_list.append(f"{speaker}/{message}")
        else:
            raise ValueError("conversation_dataは文字列またはリストである必要があります")
        
        return self.conv_text_list

    def pred_label(self):
        all_scores_per_text = []

        for line in self.conv_text_list:
            # 最初のスラッシュで分割して話者とメッセージを区別
            parts = line.split("/", 1)
            if len(parts) < 2: # スラッシュがない場合のハンドリング
                speaker, message = "不明な話者", line 
            else:
                speaker, message = parts[0], parts[1]
            
            result = self.classifier(
                message,
                candidate_labels=self.topic_labels,
                hypothesis_template="このテキストは{}に関する会話である"
            )

            label_to_score = {label: score for label, score in zip(result['labels'], result['scores'])}
            all_scores_per_text.append(label_to_score)

        label_avg_scores = {label: [] for label in self.topic_labels}
        for scores_dict in all_scores_per_text:
            for label, score in scores_dict.items():
                if label in label_avg_scores:
                    label_avg_scores[label].append(score)

        self.pred_score = []
        for label, scores_list in label_avg_scores.items():
            if scores_list:
                avg_score = sum(scores_list) / len(scores_list)
                self.pred_score.append([label, [avg_score]])

        self.pred_score.sort(key=lambda x: x[1][0], reverse=True)
        
        if self.pred_score:
            max_item = self.pred_score[0]
            max_item_of_value = round(max_item[1][0] * 100, 2)
            for i in range(len(self.pred_score)):
                print(f"{i+1}: {self.pred_score[i]}")
            # print("done: pred_label")
            return max_item[0], max_item_of_value
        else:
            print("警告: pred_scoreが空です。トピック予測が行われませんでした。") 
            return None, 0.0

    def morphological_analysis(self):
        for line in self.conv_text_list:
            parts = line.split("/", 1)
            if len(parts) < 2:
                speaker, message = "不明な話者", line
            else:
                speaker, message = parts[0], parts[1]

            tagged = self.tagger.parse(message)
            self.tagged_line.append(f"{speaker}/{tagged}")
        # print("done: morphological_analysis")
        return self.tagged_line

    def calculate_morph_ratio(self):
        ok_hinshi = ["名詞", "動詞", "形容動詞", "形容詞"]
        speaker_morph_count = {}
        speaker_match_count = {}
        
        temp_total_morph_count = 0
        temp_total_match_count = 0

        for line in self.tagged_line:
            parts = line.split("/", 1)
            if len(parts) < 2:
                speaker, tagged = "不明な話者", line
            else:
                speaker, tagged = parts[0], parts[1]

            morphs = [m for m in tagged.strip().split("\n") if m and m != "EOS"]
            num_morph_in_line = len(morphs)

            temp_total_morph_count += num_morph_in_line
            for morph in morphs:
                if any(hinshi in morph for hinshi in ok_hinshi):
                    temp_total_match_count += 1
        
        for line in self.tagged_line:
            parts = line.split("/", 1)
            if len(parts) < 2:
                speaker, tagged = "不明な話者", line
            else:
                speaker, tagged = parts[0], parts[1]

            morphs = [m for m in tagged.strip().split("\n") if m and m != "EOS"]
            
            if speaker not in speaker_morph_count:
                speaker_morph_count[speaker] = 0
                speaker_match_count[speaker] = 0

            num_morph_current_line = len(morphs)
            speaker_morph_count[speaker] += num_morph_current_line

            match_count_current_line = 0
            for morph in morphs:
                if any(hinshi in morph for hinshi in ok_hinshi):
                    match_count_current_line += 1
            speaker_match_count[speaker] += match_count_current_line

        self.calc_amount_of_speech = []
        for speaker in speaker_morph_count:
            ratio = speaker_match_count[speaker] / temp_total_match_count if temp_total_match_count > 0 else 0
            self.calc_amount_of_speech.append(f"{speaker}:{ratio:.2f}")

        # print("done: calculate_morph_ratio")
        return self.calc_amount_of_speech
    
    def score_of_interest(self, display_speaker_label=None):
        # すでに計算済みのpred_scoreを使用
        if not self.pred_score:
            print("トピック予測がスキップされました。関心度を計算できません。")
            return
            
        max_item = self.pred_score[0]
        label = max_item[0]
        pred_score_value = round(max_item[1][0] * 100, 2)
            
        am_of_speech = self.calc_amount_of_speech
        print("＝＝＝＝＝")
        if display_speaker_label:
            print(f"'{display_speaker_label}'さんの会話ログのトピック: {label}")
        else:
            print(f"会話ログのトピック: {label}")

        for i in am_of_speech:
            talker, score_speech_str = i.split(":")
            score_speech = float(score_speech_str)
            
            calc_score = pred_score_value * score_speech
            print(f"{talker}の関心度: {calc_score:.2f}%")
        # print("done: score_of_interest")