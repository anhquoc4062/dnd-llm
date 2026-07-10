"""
translation.py — Dịch output của DM (đang được gen bằng tiếng Anh) sang tiếng
Việt trước khi trả về người chơi.

Có 2 cơ chế dịch khác nhau, dùng cho 2 loại nội dung khác nhau:

1. translate_result() — dịch bằng LLM (1 lần gọi ollama riêng). Dùng cho nội
   dung TỰ DO (story text, choices[].text) — không thể tra bảng vì đây là văn
   xuôi do model tự sinh, không nằm trong character sheet.

2. localize_choices() / localize_reason_name() — dịch bằng TRA BẢNG (không
   tốn LLM call). Dùng cho choices[].reason.name — trường này LUÔN là tên
   item/skill/trait có sẵn trong character sheet (model bắt buộc chỉ được
   cite tên có thật), nên tra ngược EN->VI từ chính character sheet rẻ hơn
   và chính xác hơn nhiều so với để LLM dịch tự do (LLM dịch tự do có thể
   dịch sai/lệch tên so với tên VI gốc người chơi đã đặt lúc tạo nhân vật).

QUAN TRỌNG VỀ THỨ TỰ GỌI (xem hướng dẫn tích hợp cuối file):
- translate_result() phải được gọi SAU khi đã INSERT story (tiếng Anh gốc)
  vào bảng history, vì history được feed ngược lại cho DM model ở các turn
  sau — nếu lưu bản tiếng Việt vào history, DM (đang được prompt để nghĩ và
  viết bằng tiếng Anh) sẽ bị lẫn ngôn ngữ trong context, dễ gây trôi văn
  phong/lặp lỗi dịch ngược.
- translate_result() nên được gọi TRƯỚC khi lưu last_result xuống DB (cột
  dùng cho /start_game resume), để lần resume sau trả thẳng bản đã dịch,
  không phải dịch lại tốn thêm 1 lần gọi LLM.
"""

import json

import ollama

TRANSLATION_MODEL = "qwen3:14b"  # có thể trỏ sang model nhỏ/nhanh hơn riêng cho dịch
TRANSLATION_OPTIONS = {"num_ctx": 4096, "num_predict": 1000}


def _build_translation_prompt() -> str:
    return """/no_think
You are a professional English-to-Vietnamese literary translator for a dark-fantasy
D&D narrative game. Output ONLY valid JSON, same shape as the input, no markdown
fences, no commentary outside the JSON.

STRICT RULES:
- Translate every string fully and naturally into Vietnamese — keep the dark, grim
  narrative tone of the original. Do not soften, summarize, shorten, or add anything
  not present in the original text.
- KEEP UNCHANGED (do not translate): proper nouns — names of people/NPCs, monsters,
  and locations (e.g. "Waterdeep", "Mangy Wolf", "Kesa Two-Blades" stay exactly as-is,
  in English, even mid-sentence).
- Preserve the exact number of items in the "choices" list, in the same order.
- If the input "story" is empty, output "story": "".
"""


def translate_result(result: dict, model: str = None, options: dict = None) -> dict:
    """Dịch result['story'] + result['choices'][i]['text'] từ tiếng Anh sang
    tiếng Việt bằng 1 lần gọi LLM riêng. CHỈ trích xuất đúng 2 phần này để
    dịch (không đụng vào mechanics/keys/roll_type...) — vừa giảm token, vừa
    tránh model dịch làm hỏng cấu trúc JSON mechanics.

    Sửa result IN-PLACE và trả về luôn để tiện dùng dạng:
        result = translation.translate_result(result)

    Nếu dịch lỗi (model trả JSON hỏng, hoặc lệch số lượng choices), trả về
    result với text GIỮ NGUYÊN TIẾNG ANH thay vì crash — an toàn hơn là để
    cả response lỗi."""
    model = model or TRANSLATION_MODEL
    options = options or TRANSLATION_OPTIONS

    story = result.get("story", "") or ""
    choices = result.get("choices") or []
    choice_texts = [c.get("text", "") if isinstance(c, dict) else str(c) for c in choices]

    if not story and not choice_texts:
        return result  # không có gì để dịch

    payload = {"story": story, "choices": choice_texts}

    try:
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": _build_translation_prompt()},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            format="json",
            options=options,
            think=False,
        )
        raw = response["message"]["content"].replace("```json", "").replace("```", "").strip()
        translated = json.loads(raw)
    except (json.JSONDecodeError, KeyError, TypeError):
        print("[DEBUG] translate_result: dịch lỗi, giữ nguyên bản tiếng Anh")
        return result

    translated_story = translated.get("story")
    translated_choice_texts = translated.get("choices")

    if isinstance(translated_story, str) and translated_story.strip():
        result["story"] = translated_story

    if (
        isinstance(translated_choice_texts, list)
        and len(translated_choice_texts) == len(choices)
    ):
        for ch, new_text in zip(choices, translated_choice_texts):
            if isinstance(ch, dict) and isinstance(new_text, str) and new_text.strip():
                ch["text"] = new_text
    elif translated_choice_texts is not None:
        print(f"[DEBUG] translate_result: số choices dịch ra ({len(translated_choice_texts) if isinstance(translated_choice_texts, list) else '?'}) "
            f"lệch với gốc ({len(choices)}) -> giữ nguyên choices tiếng Anh")

    return result


# ---------------------------------------------------------------------------
# Tra bảng EN -> VI cho reason.name (không tốn LLM call, chính xác theo sheet)
# ---------------------------------------------------------------------------

def localize_reason_name(reason_type: str, name: str, char_dict: dict) -> str:
    """Model chỉ biết tên EN của item/skill/trait — map ngược lại tên tiếng
    Việt (do người chơi đặt lúc tạo nhân vật) để hiển thị. Nếu không tìm
    thấy match (hoặc reason_type là "attribute", vốn giữ nguyên STR/DEX/...),
    trả lại chính tên gốc không đổi."""
    name_lower = (name or "").strip().lower()
    if reason_type == "strength":
        for t in char_dict.get("strengths", []):
            if (t.get("en") or t.get("name") or "").strip().lower() == name_lower:
                return t.get("name") or name
    elif reason_type == "weakness":
        for t in char_dict.get("weaknesses", []):
            if (t.get("en") or t.get("name") or "").strip().lower() == name_lower:
                return t.get("name") or name
    elif reason_type == "skill":
        for s in char_dict.get("skills", []):
            if (s.get("en") or "").strip().lower() == name_lower:
                return s.get("vi") or name
    elif reason_type == "item":
        for i in char_dict.get("items", []):
            if (i.get("en") or "").strip().lower() == name_lower:
                return i.get("vi") or name
    return name  # attribute (STR/DEX...) giữ nguyên, hoặc không tìm thấy match


def localize_choices(choices, char_dict: dict):
    """Duyệt qua list choices trả về từ model, dịch ngược reason.name của
    từng choice (nếu có) sang tiếng Việt theo sheet. Sửa in-place và trả về
    luôn list đó cho tiện dùng dạng `result["choices"] = localize_choices(...)`.

    Độc lập với translate_result() ở trên — có thể gọi trước hoặc sau đều
    được vì chúng đụng vào 2 trường khác nhau (reason.name vs text)."""
    for ch in choices or []:
        if not isinstance(ch, dict):
            continue  # phòng model trả sai schema (vd list string thay vì object)
        reason = ch.get("reason")
        if reason and reason.get("name"):
            reason["name"] = localize_reason_name(reason.get("type"), reason.get("name"), char_dict)
    return choices