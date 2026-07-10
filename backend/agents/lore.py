"""
lore.py — RAG nhẹ cho địa điểm/NPC Forgotten Realms.

Thiết kế cho session ngắn (45p-1h):
- Không dùng vector DB / embedding model -> không tốn thêm lần gọi ollama,
  không thêm dependency. Chỉ so khớp keyword/tag trong RAM, load 1 lần khi
  server start.
- Data được tách thành nhiều file nhỏ dưới lore_data/regions/*.json (1 file
  = 1 region) để dễ chỉnh sửa/mở rộng độc lập, không phải sửa 1 file khổng lồ.
- Mỗi campaign (mỗi lần /start_game) chốt NGẪU NHIÊN 1 region duy nhất và
  lưu vào DB (character.region). Toàn bộ session chỉ retrieve trong phạm vi
  region đó -> tập dữ liệu tìm kiếm nhỏ, nội dung nhất quán về mặt địa lý
  cho một buổi chơi.
- NPC không cố định tên: mỗi region định nghĩa các NPC ARCHETYPE (vai trò +
  mô tả + pool tên khả dĩ). Lần đầu 1 archetype được nhắc tới trong session,
  instantiate_npc() random 1 tên từ pool và CHỐT LẠI (lưu vào npc_pool, do
  main.py persist xuống DB) để tên không đổi giữa các lượt — nhưng session
  mới (playthrough mới) sẽ ra tên khác, tăng tính đa dạng giữa các lần chơi.
- retrieve_lore() trả về top-k location + top-k npc-archetype liên quan nhất
  tới ngữ cảnh hiện tại (lượt trước + hành động người chơi), để nhét vào
  prompt dưới dạng vài dòng gợi ý — KHÔNG phải để model đọc thuộc lòng.
"""

import glob
import json
import os
import random
import re

_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LORE_DIR = os.path.join(_BACKEND_DIR, "game-data", "lore_data", "regions")
_LORE_CACHE = None


def _load_lore():
    global _LORE_CACHE
    if _LORE_CACHE is None:
        regions = []
        for path in sorted(glob.glob(os.path.join(_LORE_DIR, "*.json"))):
            with open(path, "r", encoding="utf-8") as f:
                regions.append(json.load(f))
        _LORE_CACHE = regions
    return _LORE_CACHE


def list_region_names():
    return [r["region"] for r in _load_lore()]


def pick_random_region():
    return random.choice(list_region_names())


def get_weather_pool(region_name):
    """Trả về weather_pool (list of {weather, weight}) của 1 region, dùng
    bởi world_state.py để roll thời tiết phù hợp khí hậu vùng đó."""
    region = _get_region_data(region_name)
    if not region:
        return []
    return region.get("weather_pool", [])


def _get_region_data(region_name):
    for r in _load_lore():
        if r["region"] == region_name:
            return r
    return None


_WORD_RE = re.compile(r"[a-zA-ZÀ-ỹ]+")


def _tokenize(text):
    return set(w.lower() for w in _WORD_RE.findall(text or ""))


def _score_entry(entry, query_tokens, query_text_lower, instantiated_name=None):
    """Điểm số đơn giản: cộng dồn theo số tag/keyword khớp + thưởng thêm nếu
    tên (đã sinh ra hoặc name_en cố định của location) xuất hiện nguyên văn
    trong query (proper noun thường được giữ nguyên tiếng Anh trong output
    của DM theo đúng rule prompt)."""
    score = 0
    name = entry.get("name_en") or instantiated_name or ""
    if name and name.lower() in query_text_lower:
        score += 5  # nhắc thẳng tên -> gần như chắc chắn liên quan
    tags = entry.get("tags", [])
    for tag in tags:
        if tag.lower() in query_tokens:
            score += 1
    return score


def instantiate_npc(archetype: dict, npc_pool: dict):
    """Chốt tên cho 1 archetype NPC. Nếu archetype này đã được sinh tên trong
    session hiện tại (có trong npc_pool), TÁI SỬ DỤNG tên cũ để giữ nhất quán
    (không đổi tên NPC giữa chừng game). Nếu chưa, random 1 tên mới từ
    name_pool và trả về npc_pool đã cập nhật.

    Trả về: (name, updated_npc_pool)
    """
    key = archetype["key"]
    if key in npc_pool:
        return npc_pool[key], npc_pool
    name = random.choice(archetype.get("name_pool") or [archetype.get("role", "Stranger")])
    updated = dict(npc_pool)
    updated[key] = name
    return name, updated


def retrieve_lore(region_name, query_text, npc_pool=None, top_k_locations=2, top_k_npcs=2):
    """Trả về (locations, npc_archetypes) liên quan nhất trong phạm vi 1 region.
    Nếu không entry nào khớp keyword nào (turn mở đầu, hội thoại chung chung),
    trả về ngẫu nhiên 1 location + 1 archetype để DM có sẵn "mồi" giới thiệu,
    thay vì trả rỗng."""
    region = _get_region_data(region_name)
    if not region:
        return [], []

    npc_pool = npc_pool or {}
    query_tokens = _tokenize(query_text)
    query_text_lower = (query_text or "").lower()

    scored_locations = [
        (_score_entry(loc, query_tokens, query_text_lower), loc)
        for loc in region["locations"]
    ]
    scored_npcs = [
        (_score_entry(a, query_tokens, query_text_lower, npc_pool.get(a["key"])), a)
        for a in region["npc_archetypes"]
    ]

    scored_locations.sort(key=lambda x: x[0], reverse=True)
    scored_npcs.sort(key=lambda x: x[0], reverse=True)

    top_locations = [loc for score, loc in scored_locations[:top_k_locations] if score > 0]
    top_npc_archetypes = [a for score, a in scored_npcs[:top_k_npcs] if score > 0]

    if not top_locations:
        top_locations = [random.choice(region["locations"])]
    if not top_npc_archetypes:
        top_npc_archetypes = [random.choice(region["npc_archetypes"])]

    return top_locations, top_npc_archetypes


def allowed_tiers_for_level(character_level: int) -> list:
    """Cap độ nguy hiểm của monster mới sinh theo level nhân vật, tránh việc
    model quăng dragon vào cho nhân vật level 1. Ngưỡng cố ý rộng rãi (không
    theo đúng CR table D&D) vì đây chỉ là rào chắn "hợp lý hoá", không phải
    balance số học chính xác."""
    level = character_level or 1
    if level <= 2:
        return ["low"]
    if level <= 5:
        return ["low", "medium"]
    return ["low", "medium", "high"]


def suggest_creatures(region_name, character_level, max_items=4):
    """Trả về danh sách tên quái/mối đe doạ phù hợp với region + level hiện
    tại, để nhét vào prompt như gợi ý (không phải danh sách bắt buộc)."""
    region = _get_region_data(region_name)
    if not region or "creature_tiers" not in region:
        return []

    tiers = region["creature_tiers"]
    allowed = allowed_tiers_for_level(character_level)

    pool = []
    for tier in allowed:
        pool.extend(tiers.get(tier, []))

    random.shuffle(pool)
    return pool[:max_items]


def format_lore_context(region_name, query_text, character_level=1, npc_pool=None):
    """Trả về (context_text, updated_npc_pool).

    npc_pool: dict {archetype_key: generated_name} đã có từ trước (load từ DB
    qua main.py). Archetype nào chưa có tên trong pool sẽ được random 1 tên
    mới; archetype đã có tên thì TÁI SỬ DỤNG (giữ nhất quán trong session).
    Caller (main.py) chịu trách nhiệm persist updated_npc_pool trở lại DB."""
    npc_pool = dict(npc_pool or {})
    locations, npc_archetypes = retrieve_lore(region_name, query_text, npc_pool=npc_pool)

    loc_lines = "\n".join(
        f"- {loc['name_en']}: {loc['description']}" for loc in locations
    )

    npc_line_parts = []
    for archetype in npc_archetypes:
        name, npc_pool = instantiate_npc(archetype, npc_pool)
        npc_line_parts.append(f"- {name} ({archetype['role']}): {archetype['description']}")
    npc_lines = "\n".join(npc_line_parts)

    suggested_creatures = suggest_creatures(region_name, character_level)
    creature_lines = ", ".join(suggested_creatures) if suggested_creatures else "None"
    allowed = allowed_tiers_for_level(character_level)

    context_text = f"""## WORLD LORE (background reference — weave in naturally, do NOT dump as a list, do NOT mention these notes exist)
Region: {region_name}

Relevant locations nearby:
{loc_lines}

Relevant NPCs who could plausibly be present or referenced (names are already
finalized for this playthrough — use EXACTLY as given, do not rename them):
{npc_lines}

Use these ONLY if they fit the current scene naturally. You are not required to
introduce all of them this turn. Never contradict a detail already established
earlier in this conversation — established facts always override this reference.

## THREAT SCALING (applies when introducing a NEW monster/hostile entity)
Character is level {character_level}. Plausible threats for this region at this power
level: {creature_lines}
Allowed danger tiers right now: {", ".join(allowed)}.
- In a settled/populated location (city, village, inn, market), default to mundane
  human/beast threats (thugs, wild animals, petty criminals) — NOT monsters, unless the
  scene has clearly established something supernatural is happening there.
- Reserve tougher or supernatural threats for wilderness, ruins, or dungeons, and only
  introduce a "high" tier threat if the character has leveled up enough (see allowed
  tiers above) AND the story has built toward it (e.g. after several turns of rising
  danger, a boss-like climax) — never as a random first encounter.
- Do not introduce an apex/legendary creature (dragon, lich, demon lord, etc.) unless it
  fits the allowed tiers above and the narrative stakes clearly justify it."""

    return context_text, npc_pool