# Zero Shot Learning Module
# 軽量版 - 機械学習とゼロショット学習機能

import openai
import MeCab
from transformers import pipeline
import torch

class ZeroShotLearning:
    def __init__(self, model_name="facebook/bart-large-mnli", mecab_dic_path=None):
        self.classifier = pipeline(
            'zero-shot-classification',
            model=model_name,
            tokenizer=model_name,
            device=0 if torch.cuda.is_available() else -1
        )
        
        if mecab_dic_path:
            self.tagger = MeCab.Tagger(f'-d {mecab_dic_path} -r /dev/null')
        else:
            self.tagger = MeCab.Tagger()
        
    def classify_text(self, text, candidate_labels):
        '''テキストをゼロショット分類'''
        result = self.classifier(text, candidate_labels)
        return result
        
    def extract_insights(self, conversation_data, topic_labels, display_speaker_label=None):
        # 会話データの準備
        conversations = self._prepare_conversation_data(conversation_data)
        
        # 各発話を分類
        all_results = []
        speaker_texts = {}
        
        for speaker, text in conversations:
            if speaker not in speaker_texts:
                speaker_texts[speaker] = []
            speaker_texts[speaker].append(text)
            
            # 個別の発話を分類
            result = self.classifier(
                text,
                candidate_labels=topic_labels,
                hypothesis_template="このテキストは{}に関する会話である"
            )
            all_results.append((speaker, result))
        
        # 結果の集計
        insights = self._aggregate_results(all_results, topic_labels, display_speaker_label)
        return insights
    
    def _prepare_conversation_data(self, conversation_data):
        '''会話データの準備'''
        conversations = []
        
        if isinstance(conversation_data, str):
            conversations.append(("話者", conversation_data))
        elif isinstance(conversation_data, list):
            for utterance in conversation_data:
                speaker = utterance.get("speaker", "不明な話者")
                message = utterance.get("text", "")
                conversations.append((speaker, message))
        else:
            raise ValueError("conversation_dataは文字列またはリストである必要があります")
        
        return conversations
    

    
    def _aggregate_results(self, all_results, topic_labels, display_speaker_label):
        '''結果の集計'''
        # トピック別の平均スコア計算
        topic_scores = {label: [] for label in topic_labels}
        
        for speaker, result in all_results:
            for label, score in zip(result['labels'], result['scores']):
                if label in topic_scores:
                    topic_scores[label].append(score)
        
        # 各トピックの平均スコアを計算
        avg_topic_scores = {}
        for label, scores in topic_scores.items():
            if scores:
                avg_topic_scores[label] = sum(scores) / len(scores)
        
        # 最も高いスコアのトピックを取得
        if avg_topic_scores:
            best_topic = max(avg_topic_scores, key=avg_topic_scores.get)
            best_score = avg_topic_scores[best_topic]
        else:
            best_topic = "不明"
            best_score = 0.0
        
        """
        # ラベルごとの予測値を表示
        sorted_topic_scores = sorted(avg_topic_scores.items(), key=lambda x: x[1], reverse=True)
        for i, (label, score) in enumerate(sorted_topic_scores):
            print(f"{i+1}: [{label}, [{score:.4f}]]")
        """
        
        return {
            "best_topic": best_topic,
            "best_score": best_score,
            "topic_scores": avg_topic_scores
        }