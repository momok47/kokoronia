# Zero Shot Learning Module
# 軽量版 - 機械学習とゼロショット学習機能

import openai
import MeCab
from transformers import pipeline
import torch

class ZeroShotLearning:
    def __init__(self, model_name="facebook/bart-large-mnli", unidic_path=None):
        # モデルタイプの判定
        if "japanese" in model_name or "tohoku" in model_name:
            self.model_type = "japanese"
        elif "mDeBERTa" in model_name or "xlm" in model_name or "multilingual" in model_name:
            self.model_type = "multilingual"
        else:
            self.model_type = "english"
        
        # パイプラインの初期化
        self.classifier = pipeline(
            'zero-shot-classification',
            model=model_name,
            tokenizer=model_name,
            device=0 if torch.cuda.is_available() else -1
        )
        
        # 後方互換性のために
        self.is_japanese_model = (self.model_type == "japanese")
        
        # MeCabの初期化（必要な場合のみ）
        self.tagger = None
        if unidic_path:
            try:
                self.tagger = MeCab.Tagger(f'-d {unidic_path} -r /dev/null')
            except RuntimeError:
                print("警告: UniDic辞書パスが見つかりません。MeCab機能は無効化されます。")
        elif unidic_path is None:
            # unidic_path=Noneの場合、MeCabを初期化しない
            pass
        else:
            try:
                self.tagger = MeCab.Tagger()
            except RuntimeError:
                print("警告: MeCabの初期化に失敗しました。MeCab機能は無効化されます。")
        
    def classify_text(self, text, candidate_labels):
        # モデルタイプに応じてhypothesis_templateを選択
        if self.model_type == "japanese":
            # 日本語モデルの場合
            result = self.classifier(text, candidate_labels)
        elif self.model_type == "multilingual":
            # 多言語モデルの場合は日本語テンプレートを使用
            result = self.classifier(
                text, 
                candidate_labels,
                hypothesis_template="このテキストは{}について述べています"
            )
        else:
            # 英語モデルの場合は英語テンプレートを使用
            result = self.classifier(
                text, 
                candidate_labels,
                hypothesis_template="This text is about {}"
            )
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
            if self.model_type == "japanese":
                result = self.classifier(
                    text,
                    candidate_labels=topic_labels,
                    hypothesis_template="このテキストは{}に関する会話である"
                )
            elif self.model_type == "multilingual":
                # 多言語モデルの場合は日本語テンプレートを使用
                result = self.classifier(
                    text,
                    candidate_labels=topic_labels,
                    hypothesis_template="このテキストは{}に関する会話である"
                )
            else:
                # 英語モデルの場合は英語テンプレートを使用
                result = self.classifier(
                    text,
                    candidate_labels=topic_labels,
                    hypothesis_template="This text is about {}"
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
        
        # ラベルごとの予測値を表示（デバッグ用）
        sorted_topic_scores = sorted(avg_topic_scores.items(), key=lambda x: x[1], reverse=True)
        model_type_display = {"japanese": "日本語", "multilingual": "多言語", "english": "英語"}
        print(f"=== スコア分布 (モデル: {model_type_display.get(self.model_type, self.model_type)}) ===")
        for i, (label, score) in enumerate(sorted_topic_scores[:5]):  # トップ5のみ表示
            print(f"{i+1}: {label} -> {score:.4f}")
        
        return {
            "best_topic": best_topic,
            "best_score": best_score,
            "topic_scores": avg_topic_scores
        }