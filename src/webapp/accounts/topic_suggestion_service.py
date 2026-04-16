import os

from openai import OpenAI

from .models import UserTopicScore


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
        raise ValueError("OPENAI_API_KEY is not set")

    profile_a = get_user_topic_scores_text(speaker_tag_a)
    profile_b = get_user_topic_scores_text(speaker_tag_b)
    if not profile_a and not profile_b:
        raise ValueError("No topic score data found for both users")

    prompt = f"""
あなたは会話支援アシスタントです。
次の2人の興味関心スコア（トピックラベルとスコア）を分析し、次回の会話に最適な**個性的で具体的な話題を1つ提案**してください。

出力要件:
- 返答は日本語の1文のみで、ユーザーにそのまま提示できる自然な語りかけの口調にすること。
- 箇条書き・前置き・理由説明・引用符などは一切禁止。
- 共通の興味がない場合やデータがない場合は、誰もが話しやすい一般的な話題（天気、最近のニュースなど）を提案すること。

<出力例>
映画に出てきた食べてみたいキャンプ飯は？

ユーザーA({speaker_tag_a})の興味関心:
{profile_a or "データなし"}

ユーザーB({speaker_tag_b})の興味関心:
{profile_b or "データなし"}
""".strip()

    try:
        client = OpenAI(api_key=api_key)
        response = client.responses.create(
            model="gpt-5.4-nano",
            input=prompt,
            temperature=0.5,
        )
        suggestion = (response.output_text or "").strip()
        if not suggestion:
            raise ValueError("OpenAI returned empty output")
        return " ".join(suggestion.splitlines())[:180]
    except Exception as exc:
        raise RuntimeError("Failed to generate next topic sentence") from exc
