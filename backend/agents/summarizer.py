"""
summary.py — Tóm tắt history để AI không bị "lú" khi session dài.

Thiết kế: rolling summary + sliding window gần nhất.
- Bảng `history` trong DB CHỈ dùng để đọc/audit đầy đủ (không đổi cách lưu,
  không xoá gì cả).
- AI KHÔNG đọc toàn bộ history dài dần theo turn — chỉ đọc:
  1) 1 đoạn SUMMARY cô đọng (đã gộp mọi turn cũ hơn ngưỡng), và
  2) Các turn GẦN NHẤT còn nguyên văn (chưa gộp vào summary).
- Summary CHỈ được cập nhật bằng LLM khi số turn chưa gộp vượt ngưỡng
  (SUMMARIZE_TRIGGER_TURNS) — không tốn thêm lần gọi model mỗi turn, chỉ
  thỉnh thoảng khi cần "dọn dẹp" cửa sổ ngữ cảnh.
"""

import ollama

SUMMARY_MODEL = "qwen3:14b"
SUMMARY_OPTIONS = {"num_ctx": 4096, "num_predict": 400}

# Khi số turn CHƯA gộp vào summary vượt ngưỡng này -> trigger tóm tắt. Đây
# cũng chính là kích thước tối đa của "cửa sổ nguyên văn" gửi cho DM mỗi
# turn (vì mọi turn > summarized_up_to_turn đều được gửi verbatim).
SUMMARIZE_TRIGGER_TURNS = 6


def _build_summary_prompt() -> str:
    return """/no_think
You maintain a running summary of an ongoing D&D solo campaign for narrative
continuity. Output ONLY the updated summary text — no JSON, no markdown, no preamble,
no headers.

Merge the PREVIOUS SUMMARY with the NEW EVENTS into one updated summary. Keep it
tight — 100-180 words. Preserve: current location, key NPCs met and their
disposition/relationship, major decisions and consequences, active threats or quests,
important items/allies gained or lost. Drop minor flavor/dialogue detail. Write in
plain English, third person, present tense, as compact factual notes (not prose)."""


def needs_summarization(current_turn: int, summarized_up_to_turn: int) -> bool:
    return (current_turn - (summarized_up_to_turn or 0)) > SUMMARIZE_TRIGGER_TURNS


def update_summary(previous_summary: str, new_events_text: str, model: str = None, options: dict = None) -> str:
    """Gộp summary cũ + các turn mới (verbatim) thành summary mới, gọn lại
    bằng 1 lần gọi LLM riêng. Nếu lỗi (model không phản hồi được, JSON hỏng...),
    fallback nối thô thay vì mất thông tin — không đẹp nhưng an toàn, và tự
    chặn phình vô hạn bằng cách cắt bớt ký tự nếu fallback lặp lại nhiều lần."""
    model = model or SUMMARY_MODEL
    options = options or SUMMARY_OPTIONS

    previous_summary = (previous_summary or "").strip()
    payload = (
        f"PREVIOUS SUMMARY:\n{previous_summary or '(none yet — this is the first summary)'}"
        f"\n\nNEW EVENTS:\n{new_events_text}"
    )

    try:
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": _build_summary_prompt()},
                {"role": "user", "content": payload},
            ],
            options=options,
            think=False,
        )
        new_summary = response["message"]["content"].strip()
        return new_summary or previous_summary
    except Exception as e:
        print(f"[DEBUG] update_summary lỗi ({e}) -> fallback nối thô, giữ thông tin nhưng không nén gọn")
        combined = f"{previous_summary}\n{new_events_text}".strip()
        return combined[-1500:]  # chặn phình vô hạn nếu fallback lặp lại nhiều lần liên tiếp


def format_summary_context(summary: str) -> str:
    if not summary:
        return ""
    return f"""## STORY SO FAR (condensed summary of earlier events — stay consistent with this)
{summary}

Do not contradict anything stated above. Earlier established facts always take priority
over inventing something new this turn."""


def format_events_for_summarization(history_rows) -> str:
    """history_rows: list các row {role, content} (đã lấy từ DB) cần gộp vào
    summary. Format thành text đơn giản, dễ đọc cho model tóm tắt."""
    lines = []
    for row in history_rows:
        role = "Player" if row["role"] == "user" else "Story"
        lines.append(f"[{role}] {row['content']}")
    return "\n".join(lines)