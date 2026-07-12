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
2. LLM FALLBACK (chỉ khi bước 1 không tìm thấy — DM đã bịa ra 1 cái tên hoàn
   toàn mới ngoài roster). Dùng đúng đoạn story text của lượt này làm ngữ cảnh.

LƯU Ý VỀ MODEL: bước 2 (và _translate_to_vi ở bước 1) dùng LẠI ĐÚNG model của
DM (qwen3:14b, xem config.py) thay vì 1 model nhỏ riêng (từng dùng qwen3:4b).
Lý do: đo thực tế trên card 12GB — qwen3:14b một mình đã chiếm ~10GB, chỉ còn
~2GB cho mọi thứ khác; nạp thêm 1 model KHÁC (4B, ~2.5-3GB) trong lúc 14B vẫn
đang cần dùng lại (turn kế tiếp) khiến Ollama phải ĐÁ 14B ra khỏi VRAM để
nhường chỗ, rồi turn sau lại phải load lại 14B từ đầu — đo được có lần mất
tới 240s chỉ để load lại. Cùng 1 model thì Ollama không cần swap gì cả, dù
mỗi lời gọi "phụ" này tốn thêm chút compute so với dùng model 4B (chấp nhận
được, vẫn rẻ hơn nhiều so với chi phí swap)."""

import json

from config import MODEL as _DM_MODEL, OPTIONS as _DM_OPTIONS
import ollama

from . import campaign, text_utils

# RESOLVER_MODEL = "qwen3:14b"
RESOLVER_MODEL = "qwen3:8b"  # dùng lại ĐÚNG model DM đang chạy, xem lý do ở trên
# num_ctx khớp với OPTIONS của DM (config.py) — tránh Ollama phải cấp phát lại
# KV cache khác kích thước cho "cùng 1 model" (dù chưa chắc gây reload, khớp
# cho chắc, chi phí bằng 0 vì không đổi behaviour gì khác).
RESOLVER_OPTIONS = {"num_ctx": 8196, "num_predict": 320, "temperature": 0.8}  # nâng từ 220 -> 320 vì visual_prompt giờ chi tiết hơn (nhiều cụm mô tả theo _KIND_GUIDANCE)

_KIND_LABEL = {
    "npc": "an NPC (a person)",
    "monster": "a monster/creature (NOT a normal human — an inhuman beast, aberration, undead or demon)",
    "location": "a location/environment",
    "object": "an interactive object (a thing, NOT a creature — e.g. a door, chest, lever, altar)",
}


def _match(a, b) -> bool:
    a = (a or "").strip().lower()
    b = (b or "").strip().lower()
    return bool(a) and a == b


def _lookup(kind: str, name: str, bible: dict | None, current_milestone: dict | None) -> dict | None:
    """Trả {"description", "visual_prompt"} nếu tìm thấy entry khớp tên VÀ có
    đủ cả 2 field (thiếu 1 trong 2 vẫn coi là miss, rơi xuống LLM fallback
    thay vì trả về nửa vời)."""
    if kind == "location":
        # milestone.location (chính) + sub_locations đều có thể là nơi DM đưa vào
        candidates = []
        loc = (current_milestone or {}).get("location") or {}
        if isinstance(loc, dict):
            candidates.append(loc)
        candidates.extend(l for l in ((current_milestone or {}).get("sub_locations") or []) if isinstance(l, dict))
        for l in candidates:
            if _match(l.get("name"), name) and l.get("description") and l.get("visual_prompt"):
                return {"description": l["description"], "visual_prompt": l["visual_prompt"]}
        return None  # Bible.world.major_locations chưa có visual_prompt sẵn -> luôn miss ở đây

    if kind == "object":
        for o in ((current_milestone or {}).get("interactive_objects") or []):
            if isinstance(o, dict) and _match(o.get("name"), name) and o.get("description") and o.get("visual_prompt"):
                return {"description": o["description"], "visual_prompt": o["visual_prompt"]}
        return None

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


# Hướng dẫn visual_prompt CHI TIẾT theo từng loại (thay vì 1 dòng chung chung
# "species/build/clothing/features") — imagegen.py đã tự nối thêm STYLE_SUFFIX
# (dark fantasy concept art, grimdark, painterly, dramatic lighting...) vào
# MỌI ảnh, nên ở đây chỉ cần tập trung đặc điểm RIÊNG của chủ thể, không lặp
# lại từ khoá style chung đó.
# visual_prompt phải CỰC CHI TIẾT về HÌNH DẠNG (đây là toàn bộ thông tin
# image-gen có để vẽ — mơ hồ thì SD tự bịa, quái ra giống người). 8-12 cụm mô
# tả cụ thể, phân cách bằng dấu phẩy, KHÔNG câu hoàn chỉnh. Style chung (dark
# fantasy, grimdark, painterly...) + block riêng theo loại đã được imagegen.py
# tự nối, nên ở đây CHỈ tả đặc điểm riêng của chủ thể, không lặp style.
_KIND_GUIDANCE = {
    "monster": """This is a MONSTER — render it as clearly INHUMAN. visual_prompt (English,
comma-separated fragments, 8-12 concrete details) MUST pin down its physical FORM so an image
generator cannot default to a normal human:
- species/creature class and overall silhouette (quadruped, serpentine, hulking, skeletal, amorphous)
- body size and build relative to a human (towering, hunched, emaciated, bloated)
- skin/scale/chitin/fur/rotting-flesh texture AND specific colours
- head/face structure — explicitly state if it is eyeless, faceless, many-eyed, fanged, beaked,
  featureless (never leave the face ambiguous, or it will be drawn human)
- distinctive anatomy: number/shape of limbs, claws, horns, tail, wings, exposed bone, tumors, tendrils
- a threat/aggression cue in its pose (lunging, jaws splayed, claws raised, predatory crouch)
If it is genuinely humanoid (cultist, ghost, undead knight), still specify what makes it uncanny
(gaunt, corpse-grey, hollow sockets, unnatural proportions).""",
    "npc": """This is a PERSON. visual_prompt (English, comma-separated fragments, 8-12 concrete
details) must fully describe their APPEARANCE:
- age, gender presentation, build/height and posture
- face shape + hair (colour, length, style) + skin tone
- clothing/armour/gear that signals their role, described concretely (torn travel cloak, rune-
  etched plate, blood-stained apron), plus a carried item/weapon if any
- 2-3 distinguishing features (facial scar, missing eye, brand, tattoo, unusual eye colour, jewellery)
- an expression/mood fitting the scene (wary, hostile, grieving, smug)""",
    "location": """This is an ENVIRONMENT. visual_prompt (English, comma-separated fragments,
8-12 concrete details) must establish the PLACE:
- architecture or terrain type and scale (crumbling keep interior, drowned swamp, cavern, narrow alley)
- foreground + background layering to give depth (what's near vs. the vista beyond)
- lighting source and atmosphere (guttering torchlight, cold moonshafts, fog, drifting embers, dust motes)
- materials and wear (moss-slick stone, rusted iron, splintered wood, bone-littered floor)
- 2-3 concrete landmark details unique to THIS place (a toppled idol, a collapsed bridge, a
  black altar), never generic
- weather/particulates if relevant
CRITICAL: the SUBJECT is the empty place ITSELF (architecture/terrain) — do NOT put any
creature, monster, beast, or living figure in it. No animal, no giant, no person as the focus.""",
    "object": """This is an INTERACTIVE OBJECT (a thing, NOT a creature — do not give it a face
or limbs). visual_prompt (English, comma-separated fragments, 8-12 concrete details) must show
the OBJECT clearly as the focal point:
- what the object is and its overall shape/size
- material and construction (iron-banded oak, carved black stone, tarnished brass, bone and sinew)
- mechanism/interactive feature (heavy bar, rusted lock, glowing runes, a lever, a seam/crack)
- surface state: wear, damage, corrosion, dried blood, arcane markings, moss
- how it sits in its setting (set into a wall, chained shut, half-buried, on a pedestal)
- lighting/glow cues if magical
- centered composition, object as the subject (no people unless incidental)""",
}


def _world_context(bible: dict | None, current_milestone: dict | None, kind: str = "") -> str:
    """Vài dòng ngữ cảnh thế giới (tone + nơi chốn hiện tại) nhét vào prompt
    resolve — để thứ DM bịa ra NGOÀI roster vẫn ĐỒNG NHẤT với thẩm mỹ/bối cảnh
    Bible+Milestone, không lệch tông.

    QUAN TRỌNG: KHÔNG đưa mô tả QUÁI (possible_encounters) vào khi đang resolve
    location/object — 8B thấy mô tả quái trong context sẽ vẽ luôn con quái vào
    ảnh tòa tháp/cái rương (bug thật đã gặp: 'Obsidian Spire' ra 1 con thú khổng
    lồ). Creature refs CHỈ hữu ích khi resolve monster/npc để bám thẩm mỹ."""
    lines = []
    if bible:
        try:
            b = campaign._normalize_bible(bible)
            c = b["campaign"]
            lines.append(f"World tone: {c.get('genre', '')}, {c.get('tone', '')}.")
        except Exception:
            pass
    if current_milestone:
        loc = current_milestone.get("location") or {}
        if isinstance(loc, dict) and loc.get("name"):
            lines.append(f"Current setting: {loc.get('name')} — {loc.get('description', '')}".strip(" —"))
        # Mô tả quái CHỈ cho monster/npc — tuyệt đối không cho location/object.
        if kind in ("monster", "npc"):
            refs = []
            for o in (current_milestone.get("possible_encounters") or [])[:2]:
                if isinstance(o, dict) and o.get("appearance"):
                    refs.append(f"{o.get('name')}: {o.get('appearance')}")
            if refs:
                lines.append("Nearby creatures for aesthetic consistency — " + " | ".join(refs))
    return "\n".join(lines)


def _llm_resolve(kind: str, name: str, scene_snippet: str, world_ctx: str = "") -> dict:
    """Đường MISS (DM bịa tên ngoài roster): sinh cả 3 field trong 1 lời gọi —
    name_vi (tên hiển thị tiếng Việt), description (tiếng Việt, 2-3 câu sống
    động), visual_prompt (tiếng Anh, cực chi tiết cho image-gen)."""
    kind_label = _KIND_LABEL.get(kind, "a thing")
    guidance = _KIND_GUIDANCE.get(kind, "")
    world_block = f"\nWORLD/SCENE CONTEXT (keep everything consistent with this — same tone, same setting):\n{world_ctx}\n" if world_ctx else ""
    prompt = f"""/no_think
A D&D dark-fantasy DM just introduced {kind_label} named "{name}" in this scene (scene text may
be in Vietnamese):
\"\"\"{(scene_snippet or "")[:700]}\"\"\"
{world_block}
Produce three things:
1. name_vi: a natural Vietnamese display name for "{name}" (keep proper-noun flavour; for a
   monster/place you may transliterate or give an evocative Vietnamese equivalent).
2. description: 2-3 vivid Vietnamese sentences shown to the player — describe how it looks and
   what feeling it evokes in THIS scene. Rich and concrete, NOT one dry sentence. 100% Vietnamese.
3. visual_prompt: follow this guidance exactly —
{guidance}

FINAL CHECK on name_vi and description: re-read word by word — they must be 100% Vietnamese, not
even ONE stray English word (a common mistake: an English adjective/verb like "bared" slips in
mid-sentence, e.g. "răng nanh dài bared ra" — WRONG, should be "răng nanh dài lộ ra"). Fix before
outputting.

Output ONLY this JSON, no markdown:
{{"name_vi": "<Vietnamese display name>",
"description": "<Vietnamese, 2-3 vivid sentences, shown to the player>",
"visual_prompt": "<English, per the guidance above — for an image generator, NOT shown to the player>"}}"""

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
        name_vi = str(parsed.get("name_vi") or "").strip()
        desc = str(parsed.get("description") or "").strip()
        vp = str(parsed.get("visual_prompt") or "").strip()
    except Exception as e:
        print(f"[DEBUG] context_writer._llm_resolve lỗi ({kind}='{name}'): {e} -> dùng fallback rỗng")
        name_vi, desc, vp = "", "", ""

    return {
        "name_vi": text_utils.strip_cjk(name_vi) or name,
        "description": text_utils.strip_cjk(desc) or f"{name} vừa xuất hiện.",
        "visual_prompt": vp or name,
    }


def _translate_display(name_en: str, appearance_en: str) -> dict:
    """Đường HIT (khớp roster Bible/Milestone): appearance+visual_prompt đã có
    sẵn (English, đã đồng nhất canon) — chỉ cần sinh phần HIỂN THỊ tiếng Việt:
    name_vi + description (2-3 câu, BÁM sát appearance gốc, không bịa thêm chi
    tiết ngoài nó). 1 lời gọi cho cả hai."""
    appearance_en = (appearance_en or "").strip()
    prompt = f"""/no_think
Translate/adapt into Vietnamese for a dark-fantasy RPG. Given an English name and appearance,
output a Vietnamese display name and a 2-3 sentence vivid Vietnamese description that stays
FAITHFUL to the appearance (do not invent details not implied by it). 100% Vietnamese — re-read
and remove any stray English word before answering.

NAME (English): "{name_en}"
APPEARANCE (English): \"\"\"{appearance_en}\"\"\"

Output ONLY this flat JSON:
{{"name_vi": "<Vietnamese display name>", "description": "<2-3 vivid Vietnamese sentences>"}}"""
    try:
        response = ollama.chat(
            model=RESOLVER_MODEL,
            messages=[{"role": "user", "content": prompt}],
            format="json",
            options={"num_ctx": _DM_OPTIONS["num_ctx"], "num_predict": 260, "temperature": 0.4},
            think=False,
        )
        raw = response["message"]["content"].replace("```json", "").replace("```", "").strip()
        parsed = json.loads(raw)
        name_vi = text_utils.strip_cjk((_find_key(parsed, "name_vi") or "").strip())
        desc = text_utils.strip_cjk((_find_key(parsed, "description") or "").strip())
    except Exception as e:
        print(f"[DEBUG] context_writer._translate_display lỗi: {e} -> fallback")
        name_vi, desc = "", ""
    return {
        "name_vi": name_vi or name_en,
        "description": desc or _translate_to_vi(appearance_en),
    }


def _find_key(obj, key: str):
    """Tìm đệ quy 1 key trong dict/list lồng nhau — phòng model bọc thêm cấu
    trúc thừa (vd {"output": {"vi": "..."}}) dù prompt đã yêu cầu flat JSON."""
    if isinstance(obj, dict):
        if key in obj and isinstance(obj[key], str):
            return obj[key]
        for v in obj.values():
            found = _find_key(v, key)
            if found:
                return found
    elif isinstance(obj, list):
        for v in obj:
            found = _find_key(v, key)
            if found:
                return found
    return None


def _translate_to_vi(text: str) -> str:
    """appearance/description trong Bible/Milestone LUÔN là tiếng Anh (rule
    LANGUAGE của campaign.py/milestone.py — chỉ campaign.theme mới tiếng
    Việt), nhưng description trả về từ resolve_visual() là field HIỂN THỊ
    CHO NGƯỜI CHƠI -> phải dịch, không được lộ tiếng Anh ra context panel.
    1 lời gọi nhỏ, riêng, dùng LẠI model DM (xem RESOLVER_MODEL) — chỉ dịch
    không sáng tác thêm.

    format="json" bắt buộc (giống _llm_resolve) — think=False + prompt dạng
    câu thường hay bị model "lan man" viết cả đoạn suy luận vào content
    (không tôn trọng /no_think tốt bằng khi bị ép khuôn JSON), dễ bị cắt cụt
    giữa chừng bởi num_predict trước khi ra tới câu dịch thật."""
    text = (text or "").strip()
    if not text:
        return text
    prompt = f"""/no_think
Translate the ENGLISH sentence below into natural, vivid Vietnamese (dark-fantasy tone). The
translation must be 100% Vietnamese — not even one leftover English word (re-read it before
answering; a leftover English adjective/verb mid-sentence is a common mistake, fix it).

Output ONLY this EXACT flat JSON shape, no extra keys, no nesting, no explanation:
{{"vi": "<Vietnamese translation>"}}

Example:
ENGLISH: "A rusty iron gate creaks in the wind."
{{"vi": "Cánh cổng sắt gỉ sét kẽo kẹt trong gió."}}

ENGLISH: \"\"\"{text}\"\"\""""
    try:
        response = ollama.chat(
            model=RESOLVER_MODEL,
            messages=[{"role": "user", "content": prompt}],
            format="json",
            options={"num_ctx": _DM_OPTIONS["num_ctx"], "num_predict": 200, "temperature": 0.3},
            think=False,
        )
        raw = response["message"]["content"].replace("```json", "").replace("```", "").strip()
        parsed = json.loads(raw)
        translated = (_find_key(parsed, "vi") or "").strip()
        return text_utils.strip_cjk(translated) or text
    except Exception as e:
        print(f"[DEBUG] context_writer._translate_to_vi lỗi: {e} -> giữ nguyên bản gốc (tiếng Anh)")
        return text


def resolve_visual(kind: str, name: str, bible: dict | None, current_milestone: dict | None, scene_snippet: str) -> dict:
    """kind: "npc" | "monster" | "location" | "object". Trả
    {"name_vi", "description", "visual_prompt"}:
    - name_vi: tên hiển thị tiếng Việt cho người chơi (canonical name vẫn là
      tiếng Anh dùng cho key/cache ảnh — đây chỉ là nhãn hiển thị).
    - description: LUÔN tiếng Việt, 2-3 câu (hiện cho người chơi).
    - visual_prompt: LUÔN tiếng Anh, cực chi tiết (cho image-gen, không hiện).

    Đường HIT (khớp roster Bible/Milestone): dùng thẳng appearance+visual_prompt
    đã soạn sẵn (đồng nhất canon), chỉ sinh phần hiển thị tiếng Việt. Đường MISS
    (DM bịa ngoài roster): sinh cả 3, kèm world-context để giữ đồng nhất tone."""
    hit = _lookup(kind, name, bible, current_milestone)
    if hit:
        disp = _translate_display(name, hit["description"])
        return {
            "name_vi": disp["name_vi"],
            "description": disp["description"],
            "visual_prompt": hit["visual_prompt"],
        }
    world_ctx = _world_context(bible, current_milestone, kind)
    return _llm_resolve(kind, name, scene_snippet, world_ctx)
