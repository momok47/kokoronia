import os

from openai import OpenAI

from .models import UserTopicScore


DEFAULT_NO_API_KEY_SUGGESTION = "次回は、お互いが最近関心を持っているテーマについて、印象に残った出来事を1つずつ共有してみましょう。"
DEFAULT_NO_DATA_SUGGESTION = "次回は、最近気になっているニュースや出来事を1つずつ持ち寄って話してみましょう。"
DEFAULT_ERROR_SUGGESTION = "次回は、二人が最近気になっているテーマを1つ選んで深掘りしてみましょう。"
DEFAULT_EMPTY_OUTPUT_SUGGESTION = "次回は、二人が共通して興味を持つテーマについて最近の体験を交えて話してみましょう。"


def get_user_topic_scores_text(account_id: str) -> str:
    """score > 0 の全トピックスコアをプロンプト用に整形する。"""
    rows = list(
        UserTopicScore.objects.filter(
            user__account_id=account_id,
            score__gt=0,
        )
        .order_by("-score", "-updated_at")
        .values("topic_label", "score")
    )
    if not rows:
        return ""
    return "\n".join(f"- {row['topic_label']}: {row['score']:.3f}" for row in rows)


def generate_next_topic_sentence(speaker_tag_a: str, speaker_tag_b: str) -> str:
    """2ユーザーの興味関心スコアから次回話題を1文で生成する。"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return DEFAULT_NO_API_KEY_SUGGESTION

    profile_a = get_user_topic_scores_text(speaker_tag_a)
    profile_b = get_user_topic_scores_text(speaker_tag_b)
    if not profile_a and not profile_b:
        return DEFAULT_NO_DATA_SUGGESTION

    prompt = f"""
あなたは会話支援アシスタントです。
次の2人の興味関心スコア（トピックラベルとスコア）を分析し、次回の会話に最適な**具体的な話題を1つ提案**してください。

出力要件:
- 返答は日本語の1文のみで、ユーザーにそのまま提示できる自然な語りかけの口調にすること。
- 箇条書き・前置き・理由説明・引用符などは一切禁止。
- 共通の興味がない場合やデータがない場合は、誰もが話しやすい一般的な話題（天気、最近のニュースなど）を提案すること。

<出力例>
最近見た映画について話してみませんか？

ユーザーA({speaker_tag_a})の興味関心:
{profile_a or "データなし"}

ユーザーB({speaker_tag_b})の興味関心:
{profile_b or "データなし"}
""".strip()

    try:
        client = OpenAI(api_key=api_key)
        response = client.responses.create(
            model="gpt-4o-mini",
            input=prompt,
            temperature=0.5,
        )
        suggestion = (response.output_text or "").strip()
        if not suggestion:
            return DEFAULT_EMPTY_OUTPUT_SUGGESTION
        return " ".join(suggestion.splitlines())[:180]
    except Exception:
        return DEFAULT_ERROR_SUGGESTION
