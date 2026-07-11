"""
context_writer.py — "Tiền kỳ" cho description (tiếng Việt, hiện cho người
chơi) + visual_prompt (tiếng Anh, cho image gen) của entity/location MỚI xuất
hiện trong 1 lượt /chat. Trước đây DM (model 14B, ngân sách token/lượt hạn
chế) phải tự viết cả 2 field này mỗi khi giới thiệu 1 NPC/quái/nơi chốn MỚI —
kể cả khi Campaign Bible/Milestone đã soạn sẵn appearance+visual_prompt cho
đúng cái đó rồi (roster), khiến DM tốn token bịa lại 1 thứ đã có sẵn, góp phần
gây tràn num_predict ở những lượt vừa có entity mới vừa có location mới.

resolve_visual() tách hẳn việc này khỏi lượt kể chuyện chính, ưu tiên theo
thứ tự:
1. TRA BẢNG (miễn phí, tức thời): tên khớp 1 entry đã có appearance+
   visual_prompt sẵn trong Milestone hiện tại (npcs/possible_encounters/
   location — mới nhất, khả năng khớp cao nhất vì DM được nhắc ưu tiên dùng
   roster) hoặc trong Campaign Bible (main_antagonist/key_npcs/
   major_monsters — major_locations của Bible KHÔNG có visual_prompt sẵn nên
   luôn rơi xuống bước 2).
2. LLM FALLBACK (1 lời gọi RIÊNG, model NHỎ qwen3:4b, tách khỏi lượt DM 14B):
   chỉ khi bước 1 không tìm thấy — nghĩa là DM đã bịa ra 1 cái tên hoàn toàn
   mới ngoài roster. Dùng đúng đoạn story text của lượt này làm ngữ cảnh.
"""

import json

import ollama

from . import campaign, text_utils

RESOLVER_MODEL = "qwen3:4b"  # nhỏ hơn hẳn model DM (14B) — việc này chỉ là
# "mô tả 1 câu + vài từ khoá thị giác", không cần suy luận sâu, dùng model to
# vừa phí VRAM/thời gian vừa tranh GPU với model DM đang chạy song song.
RESOLVER_OPTIONS = {"num_ctx": 2048, "num_predict": 220, "temperature": 0.8}

_KIND_LABEL = {"npc": "an NPC", "monster": "a monster/creature", "location": "a location"}


def _match(a, b) -> bool:
    a = (a or "").strip().lower()
    b = (b or "").strip().lower()
    return bool(a) and a == b


def _lookup(kind: str, name: str, bible: dict | None, current_milestone: dict | None) -> dict | None:
    """Trả {"description", "visual_prompt"} nếu tìm thấy entry khớp tên VÀ có
    đủ cả 2 field (thiếu 1 trong 2 vẫn coi là miss, rơi xuống LLM fallback
    thay vì trả về nửa vời)."""
    if kind == "location":
        loc = (current_milestone or {}).get("location") or {}
        if _match(loc.get("name"), name) and loc.get("description") and loc.get("visual_prompt"):
            return {"description": loc["description"], "visual_prompt": loc["visual_prompt"]}
        return None  # Bible.world.major_locations chưa có visual_prompt sẵn -> luôn miss ở đây

    # kind == "npc" hoặc "monster": gộp chung nguồn tra cứu vì main_antagonist
    # có thể được DM đưa vào entity dưới type="monster" hoặc "npc" tuỳ tình huống,
    # và milestone NPCs đôi khi được dùng làm đối thủ tạm thời.
    candidates = []
    if current_milestone:
        candidates.extend(current_milestone.get("npcs") or [])
        candidates.extend(current_milestone.get("possible_encounters") or [])
    if bible:
        b = campaign._normalize_bible(bible)
        candidates.append(b["characters"]["main_antagonist"])
        candidates.extend(b["characters"]["key_npcs"])
        candidates.extend(b["content"]["major_monsters"])

    for c in candidates:
        if not isinstance(c, dict):
            continue
        desc = c.get("appearance") or c.get("desc") or ""
        vp = c.get("visual_prompt") or ""
        if _match(c.get("name"), name) and desc and vp:
            return {"description": desc, "visual_prompt": vp}
    return None


def _llm_resolve(kind: str, name: str, scene_snippet: str) -> dict:
    kind_label = _KIND_LABEL.get(kind, "a thing")
    prompt = f"""/no_think
A D&D dark-fantasy DM just introduced {kind_label} named "{name}" in this scene (may be in
Vietnamese):
\"\"\"{(scene_snippet or "")[:600]}\"\"\"

Output ONLY this JSON, no markdown:
{{"description": "<Vietnamese, ONE vivid concise sentence, shown to the player>",
"visual_prompt": "<English, short comma-separated visual traits (species/build/clothing/
features for a character, or architecture/terrain/mood/lighting for a location) — for an
image generator, NOT shown to the player>"}}"""

    try:
        response = ollama.chat(
            model=RESOLVER_MODEL,
            messages=[{"role": "user", "content": prompt}],
            format="json",
            options=RESOLVER_OPTIONS,
            think=False,
        )
        raw = response["message"]["content"].replace("```json", "").replace("```", "").strip()
        parsed = json.loads(raw)
        desc = str(parsed.get("description") or "").strip()
        vp = str(parsed.get("visual_prompt") or "").strip()
    except Exception as e:
        print(f"[DEBUG] context_writer._llm_resolve lỗi ({kind}='{name}'): {e} -> dùng fallback rỗng")
        desc, vp = "", ""

    return {
        "description": text_utils.strip_cjk(desc) or f"{name} vừa xuất hiện.",
        "visual_prompt": vp or name,
    }


def resolve_visual(kind: str, name: str, bible: dict | None, current_milestone: dict | None, scene_snippet: str) -> dict:
    """kind: "npc" | "monster" | "location". Trả {"description", "visual_prompt"}
    — description LUÔN đã qua strip_cjk (đường tra bảng đọc thẳng từ Bible/
    Milestone, vốn cũng đã được sanitize từ lúc sinh — xem campaign.py/
    milestone.py — nhưng strip lại ở đây cho chắc, rẻ nếu không có gì để cắt)."""
    hit = _lookup(kind, name, bible, current_milestone)
    if hit:
        return {"description": text_utils.strip_cjk(hit["description"]), "visual_prompt": hit["visual_prompt"]}
    return _llm_resolve(kind, name, scene_snippet)
