"""
text_utils.py — Tiện ích xử lý text dùng chung giữa nhiều agent (tách riêng để
tránh vòng lặp import, vd dungeon_master.py <-> context_writer.py đều cần
strip_cjk nhưng không được phép import lẫn nhau).
"""

import re

_CJK_RE = re.compile(r"[㐀-䶿一-鿿豈-﫿]+")


def strip_cjk(text: str) -> str:
    """Hậu kỳ: qwen3 thỉnh thoảng lẫn 1-2 ký tự Hán vào giữa câu tiếng Việt
    (vd "găm vào鞘") dù prompt đã cấm rõ ràng — prompt-only không chặn được
    100% hành vi vốn có của model. Chỉ cắt bỏ ĐÚNG cụm ký tự CJK, giữ nguyên
    phần còn lại của câu thay vì vứt cả câu."""
    if not text or not _CJK_RE.search(text):
        return text
    cleaned = _CJK_RE.sub("", text)
    print(f"[DEBUG] strip_cjk: loại ký tự Hán lẫn trong text: {text!r} -> {cleaned!r}")
    return cleaned


def strip_cjk_deep(value):
    """Áp dụng strip_cjk đệ quy lên MỌI string trong 1 cấu trúc dict/list —
    dùng ngay tại điểm parse JSON của model (1 chỗ duy nhất mỗi module) để
    chặn cho toàn bộ field cùng lúc mà không phải sửa từng nơi tiêu thụ."""
    if isinstance(value, str):
        return strip_cjk(value)
    if isinstance(value, list):
        return [strip_cjk_deep(v) for v in value]
    if isinstance(value, dict):
        return {k: strip_cjk_deep(v) for k, v in value.items()}
    return value
