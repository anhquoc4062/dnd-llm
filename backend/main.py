import json
import sqlite3

import ollama
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import lore
import entities
import classification
import world_state
import social
import translation
import summarizer
import campaign

DB_PATH = "game.db"
MODEL = "qwen3:14b"
# MODEL = "mistral-small:22b"
# MODEL = "phi4:14b"
OPTIONS = {
    "num_ctx": 8196,
    "num_predict": 800,
    "temperature": 0.7,
}

# MODEL = "mistral-nemo:12b"

ATTR_KEYS = ["str", "dex", "con", "int", "wis", "cha"]


# ---------------------------------------------------------------------------
# DB
# ---------------------------------------------------------------------------

def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_conn()
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS character(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            gender TEXT,
            race TEXT,
            race_en TEXT,
            character_class TEXT,
            character_class_en TEXT,

            attr_str INTEGER,
            attr_dex INTEGER,
            attr_con INTEGER,
            attr_int INTEGER,
            attr_wis INTEGER,
            attr_cha INTEGER,

            hp INTEGER,
            max_hp INTEGER,
            mana INTEGER,
            max_mana INTEGER,
            level INTEGER,
            xp INTEGER,
            xp_target INTEGER,
            gold INTEGER,

            strengths TEXT,   -- JSON list of {name, en, note}
            weaknesses TEXT,  -- JSON list of {name, en, note}
            equipment TEXT,   -- JSON list of {key, vi, en}
            skills TEXT,      -- JSON list of {key, vi, en}
            items TEXT        -- JSON list of {key, vi, en}
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS history(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT,
            content TEXT
        )
    """)

    conn.commit()

    # RAG: bảng entity (NPC/quái sinh động) + world_loot (ledger loot)
    entities.init_entity_tables(conn)

    # Cho phép nâng cấp DB cũ (được tạo trước khi có race_en/character_class_en)
    # mà không cần xoá file game.db thủ công.
    existing_cols = {row["name"] for row in c.execute("PRAGMA table_info(character)")}
    int_cols_default_0 = ("turns_since_event", "weather_since_turn", "current_turn", "summarized_up_to_turn")
    for col in ("race_en", "character_class_en", "turns_since_event", "region", "npc_pool",
                "last_result", "weather", "weather_since_turn", "current_turn",
                "history_summary", "summarized_up_to_turn", "campaign_theme", "campaign_data"):
        if col not in existing_cols:
            if col in int_cols_default_0:
                c.execute(f"ALTER TABLE character ADD COLUMN {col} INTEGER DEFAULT 0")
            else:
                c.execute(f"ALTER TABLE character ADD COLUMN {col} TEXT")
    conn.commit()

    # Migration bảng history: thêm cột turn_number (đánh dấu row thuộc turn
    # nào) — cần cho cơ chế summarization (biết row nào đã gộp vào summary).
    existing_history_cols = {row["name"] for row in c.execute("PRAGMA table_info(history)")}
    if "turn_number" not in existing_history_cols:
        c.execute("ALTER TABLE history ADD COLUMN turn_number INTEGER DEFAULT 0")
    conn.commit()

    conn.close()


init_db()

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory="../"), name="static")


@app.middleware("http")
async def no_cache_for_api(request: Request, call_next):
    """Chặn browser/proxy cache cho toàn bộ API JSON (GET /start_game,
    /character_info, POST /chat...). Đây là nguyên nhân phổ biến nhất khiến
    'choices bị cache' — GET request không có Cache-Control sẽ bị trình
    duyệt tự cache và trả lại y hệt response cũ dù server đã có state mới,
    đặc biệt rõ với /start_game vì URL không đổi giữa các lần gọi."""
    response = await call_next(request)
    if not request.url.path.startswith("/static"):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def safe_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _list_json(value):
    """Accepts a list of strings OR a list of dicts and stores them as-is
    (JSON-encoded). Handles both the trait shape [{name, en, note}] and the
    equipment/skill/item shape [{key, vi, en}] without dropping fields."""
    if not value:
        return json.dumps([])
    return json.dumps(value, ensure_ascii=False)


def _load_json(text, default=None):
    if not text:
        return default if default is not None else []
    try:
        return json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return default if default is not None else []


def _load_json_dict(text):
    """Biến thể của _load_json nhưng default {} thay vì [] — dùng cho các
    cột lưu dict (vd npc_pool: {archetype_key: generated_name})."""
    if not text:
        return {}
    try:
        result = json.loads(text)
        return result if isinstance(result, dict) else {}
    except (TypeError, json.JSONDecodeError):
        return {}


def _item_matches(item, name):
    """So khớp một mục trong inventory (dict {key,vi,en} hoặc chuỗi thường)
    với tên do AI trả về (thường là tiếng Anh).

    Thử so khớp tuyệt đối trước; nếu không khớp, fallback sang so khớp
    "chứa nhau" (substring) theo cả 2 chiều để tránh bỏ sót khi model trả về
    tên hơi khác cách viết trong sheet (thừa/thiếu từ, dấu câu...)."""
    name = (name or "").strip().lower()
    if not name:
        return False
    if isinstance(item, dict):
        candidates = [item.get("en"), item.get("vi"), item.get("key"), item.get("name")]
    else:
        candidates = [str(item)]
    candidates = [(c or "").strip().lower() for c in candidates if c]
    if any(c == name for c in candidates):
        return True
    return any(len(c) >= 4 and (c in name or name in c) for c in candidates)


def character_row_to_dict(char: sqlite3.Row) -> dict:
    return {
        "name": char["name"],
        "gender": char["gender"],
        "race": char["race"],
        "race_en": char["race_en"] or char["race"],
        "character_class": char["character_class"],
        "character_class_en": char["character_class_en"] or char["character_class"],
        "region": char["region"] if "region" in char.keys() else None,
        # "npc_pool": _load_json_dict(char["npc_pool"] if "npc_pool" in char.keys() else None),
        "attrs": {
            "str": char["attr_str"],
            "dex": char["attr_dex"],
            "con": char["attr_con"],
            "int": char["attr_int"],
            "wis": char["attr_wis"],
            "cha": char["attr_cha"],
        },
        "hp": char["hp"],
        "max_hp": char["max_hp"],
        "mana": char["mana"],
        "max_mana": char["max_mana"],
        "level": char["level"],
        "xp": char["xp"],
        "xp_target": char["xp_target"],
        "gold": char["gold"],
        "strengths": _load_json(char["strengths"]),
        "weaknesses": _load_json(char["weaknesses"]),
        "equipment": _load_json(char["equipment"]),
        "skills": [_normalize_skill(s) for s in _load_json(char["skills"])],
        "items": [_normalize_item(i) for i in _load_json(char["items"])],
    }


def get_latest_character():
    conn = get_conn()
    c = conn.cursor()
    char = c.execute("SELECT * FROM character ORDER BY id DESC LIMIT 1").fetchone()
    conn.close()
    return char


# ---------------------------------------------------------------------------
# Static pages
# ---------------------------------------------------------------------------

@app.get("/")
async def get_index():
    return FileResponse("../index.html")


@app.get("/game")
async def get_game():
    return FileResponse("../screen/game/game.html")


# ---------------------------------------------------------------------------
# Campaign seed (khung truyện tổng — main goal/plot/npc/monster/boss cho DM
# bám theo xuyên suốt ván chơi). Người chơi chỉ thấy "theme"; phần còn lại bị
# giấu khỏi UI, chỉ gắn vào system prompt cho DM đọc (xem campaign.py).
# ---------------------------------------------------------------------------

@app.get("/campaign_seeds")
async def campaign_seeds():
    """AI tự bịa 5 campaign seed, mỗi cái một 'vị' (thể loại) khác nhau.
    Trả về ĐẦY ĐỦ dữ liệu (kể cả phần bí mật) — frontend chỉ RENDER field
    'theme' trong dropdown, giữ nguyên phần còn lại trong state JS để gửi lại
    lúc tạo nhân vật, không hiển thị gì thêm cho người chơi thấy."""
    seeds = campaign.generate_campaign_seeds()
    return {"seeds": seeds}


@app.post("/campaign_seed/expand")
async def campaign_seed_expand(data: dict):
    """Người chơi tự viết 1 ý tưởng/theme ngắn -> khai triển thành đúng cấu
    trúc campaign seed (main_goal/plot/npcs/monsters/boss), bám sát ý tưởng
    gốc thay vì bịa lạc đề."""
    text = (data.get("text") or "").strip()
    if not text:
        return {"error": "Thiếu nội dung kịch bản."}
    result = campaign.expand_custom_seed(text)
    return result


# ---------------------------------------------------------------------------
# Character creation
# ---------------------------------------------------------------------------

@app.post("/create_character")
async def create_character(data: dict):
    attrs = data.get("attrs", {}) or {}
    hp = safe_int(data.get("hp", 100), 100)
    mana = safe_int(data.get("mana", 50), 50)
    xp_target = safe_int(data.get("xpTarget", 10), 10)

    race = data.get("race", "")
    character_class = data.get("class", "")

    campaign_seed = data.get("campaignSeed")
    campaign_theme = None
    campaign_data_json = None
    if isinstance(campaign_seed, dict):
        campaign_theme = campaign_seed.get("theme")
        campaign_data_json = json.dumps(campaign_seed, ensure_ascii=False)

    conn = get_conn()
    c = conn.cursor()

    # Một save-slot duy nhất: xóa nhân vật & lịch sử cũ trước khi tạo mới
    c.execute("DELETE FROM character")
    c.execute("DELETE FROM history")
    conn.commit()

    # RAG: dọn sạch entity/loot của session cũ (single save-slot)
    entities.reset_session(conn)

    c.execute("""
        INSERT INTO character (
            name, gender, race, race_en, character_class, character_class_en,
            attr_str, attr_dex, attr_con, attr_int, attr_wis, attr_cha,
            hp, max_hp, mana, max_mana, level, xp, xp_target, gold,
            strengths, weaknesses, equipment, skills, items,
            campaign_theme, campaign_data
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        data.get("name", ""),
        data.get("gender", ""),
        race,
        data.get("raceEn") or race,
        character_class,
        data.get("classEn") or character_class,
        safe_int(attrs.get("str", 8), 8),
        safe_int(attrs.get("dex", 8), 8),
        safe_int(attrs.get("con", 8), 8),
        safe_int(attrs.get("int", 8), 8),
        safe_int(attrs.get("wis", 8), 8),
        safe_int(attrs.get("cha", 8), 8),
        hp, hp,        # hp = max_hp lúc mới tạo
        mana, mana,    # mana = max_mana lúc mới tạo
        1,             # level
        0,             # xp
        xp_target,
        0,             # gold
        _list_json(data.get("strengths")),
        _list_json(data.get("weaknesses")),
        _list_json(data.get("equipment")),
        _list_json(data.get("skills")),
        _list_json(data.get("items")),
        campaign_theme,
        campaign_data_json,
    ))

    conn.commit()
    conn.close()

    return {"status": "ok"}


@app.get("/character_info")
async def character_info():
    char = get_latest_character()
    if not char:
        return {}
    return character_row_to_dict(char)


@app.get("/game_state")
async def game_state():
    """Trả về TOÀN BỘ trạng thái để frontend dựng lại UI khi load/reload
    trang — khác với /start_game (chỉ trả lượt gần nhất để tiếp tục chơi).

    Dùng cái này khi cần vẽ lại scrollback đầy đủ; dùng /start_game khi chỉ
    cần "tiếp tục từ đây, tôi sẽ chọn hành động tiếp theo".
    """
    char = get_latest_character()
    if not char:
        return {"started": False, "history": [], "last_result": None, "character": None}

    region = char["region"] if "region" in char.keys() else None
    started = bool(region)

    conn = get_conn()
    c = conn.cursor()
    history_rows = c.execute(
        "SELECT id, role, content FROM history ORDER BY id ASC"
    ).fetchall()
    conn.close()

    history = []
    for i, row in enumerate(history_rows):
        content = row["content"]
        # Dòng user đầu tiên (nếu game đã start) là opening_instruction nội bộ
        # (tiếng Anh, hướng dẫn model — KHÔNG phải điều người chơi thực sự gõ).
        # Thay bằng nhãn thân thiện để frontend không hiển thị nhầm nó như
        # một tin nhắn của người chơi.
        if i == 0 and row["role"] == "user" and started:
            content = "[Bắt đầu cuộc phiêu lưu]"
        history.append({"role": row["role"], "content": content})

    last_result_raw = char["last_result"] if "last_result" in char.keys() else None
    last_result = None
    if last_result_raw:
        try:
            last_result = json.loads(last_result_raw)
        except (TypeError, json.JSONDecodeError):
            last_result = None

    return {
        "started": started,
        "region": region,
        "history": history,
        "last_result": last_result,
        "character": character_row_to_dict(char),
    }


# ---------------------------------------------------------------------------
# DM system prompt
# ---------------------------------------------------------------------------

def build_system_prompt(char: sqlite3.Row) -> str:
    c = character_row_to_dict(char)

    attrs_line = ", ".join(f"{k.upper()}: {v}" for k, v in c["attrs"].items())

    def fmt_traits(traits):
        """traits: [{name (vi), en, note}, ...] — ưu tiên tên tiếng Anh vì
        toàn bộ output của DM phải là tiếng Anh."""
        if not traits:
            return "None"
        parts = []
        for t in traits:
            label = t.get("en") or t.get("name") or ""
            note = t.get("note") or ""
            parts.append(f"{label} ({note})" if note else label)
        return "; ".join(p for p in parts if p) or "None"

    def fmt_list(values):
        """values: list of {key, vi, en} dicts HOẶC list chuỗi thường."""
        if not values:
            return "None"
        names = []
        for v in values:
            if isinstance(v, dict):
                names.append(v.get("en") or v.get("vi") or v.get("name") or "")
            else:
                names.append(str(v))
        joined = ", ".join(n for n in names if n)
        return joined or "None"

    def fmt_skills(skills):
        """Hiển thị skill kèm trạng thái cooldown hiện tại để model biết skill
        nào KHÔNG được phép cho là dùng thành công."""
        if not skills:
            return "None"
        parts = []
        for s in skills:
            label = s.get("en") or s.get("vi") or ""
            cd_cur = s.get("cooldown_current", 0)
            cd_max = s.get("cooldown_max", 0)
            if cd_cur > 0:
                parts.append(f"{label} [ON COOLDOWN: {cd_cur} turn(s) left]")
            elif cd_max > 0:
                parts.append(f"{label} [ready, cooldown {cd_max} turns]")
            else:
                parts.append(label)
        return "; ".join(p for p in parts if p) or "None"

    campaign_data_raw = char["campaign_data"] if "campaign_data" in char.keys() else None
    campaign_block = ""
    if campaign_data_raw:
        try:
            campaign_block = campaign.format_campaign_context(json.loads(campaign_data_raw))
        except (TypeError, json.JSONDecodeError):
            campaign_block = ""

    return f"""/no_think

You are the Dungeon Master for a D&D 5e dark-fantasy solo campaign. Stay in character. Never break the fourth wall, explain your reasoning, or mention being an AI. Output ONLY valid JSON, no markdown, no text outside the JSON.

PRIORITY: Character Sheet > Campaign Seed > Story Context > D&D 5e Rules > Narrative Quality.

## CHARACTER SHEET (absolute truth — never invent beyond this)
Name: {c['name']} | Race: {c['race']} | Class: {c['character_class']} | Gender: {c['gender']}
Attributes: {attrs_line}
Strengths: {fmt_traits(c['strengths'])}
Weaknesses: {fmt_traits(c['weaknesses'])}
Equipment: {fmt_list(c['equipment'])}
Skills: {fmt_skills(c['skills'])}
Items: {fmt_list(c['items'])}

{campaign_block}

{social.format_social_context(c['race_en'], c['character_class_en'])}

## ACTION VALIDATION (do this FIRST, every turn)
1. Does the action use a weapon/item/skill? Check it exists in Equipment/Skills/Items above.
   - If NOT owned: success=false. The character hesitates/fumbles reaching for something they don't have, and suffers a concrete penalty (HP loss from an enemy strike, or item damage, or being spotted) — never a "soft" consequence.
   - If a SKILL is marked "[ON COOLDOWN]" above: success=false. The character tries to use it but it isn't ready yet (still recovering / out of charge) — narrate this explicitly and apply a concrete penalty, same as an unowned skill.
   - Example: player says "I cast Fireball" but Fireball is not in Skills → success:false, mechanics.changes.hp: -8, story explains the character wastes a beat gesturing uselessly and takes a hit.
2. Is the action physically/logically possible given attributes and story context? If not, same rule applies.
3. Otherwise resolve normally: pick advantage/disadvantage/normal based on ONE attribute, strength, or weakness (never invent other justification).

## ITEMS & SKILLS — DO NOT MANAGE INVENTORY YOURSELF
- You must NEVER decide on your own that the character finds, loses, or uses an item or
  skill outside of what the backend explicitly tells you via INTERNAL MECHANICS each turn.
- Only put a name in mechanics.changes.items_added if the backend's turn note explicitly
  allows a loot drop this turn (it will say so). Otherwise items_added MUST be [].
- Only put a name in mechanics.changes.items_removed if the backend's turn note explicitly
  names an item that was just consumed. Otherwise items_removed MUST be [].
- Narrate item use/finding richly in the story text, but the actual bookkeeping (removing
  used items, applying cooldowns) is handled by the backend — do not duplicate it.

## CONSEQUENCES (must show up in mechanics.changes)
- Success: reward XP/gold/items/story progress as fits.
- Failure: HP loss (combat), mana loss (magic), or a concrete story setback (alerted enemies, lost item, dropped guard) — always something mechanical, never just narrative.

## DEATH RULE
If this turn's damage would reduce the character's HP to 0 or below, set
mechanics.character_died = true. When character_died is true, choices MUST be an
empty array [] — the story ends here, no further action is possible. Never set
character_died = true unless HP truly reaches 0 this turn.

## DEATH SCENE (when mechanics.character_died = true)
This is the FINAL story the character will ever hear — make it unmistakable and brutal.
Requirements:
- 150-220 words, no choices follow, no ambiguity — the character is unquestionably dead.
- The tormentor's voice cuts through the final moment — mocking, triumphant, merciless
  (use {c['name']}, never "bạn", per NARRATION STYLE; scorn is woven into the scene itself).
- Describe the death itself in full, unflinching physical detail — the wound, the
  failing body, the last sensation before darkness. No fade-to-black, no "everything
  goes dark" cop-out — commit to the visceral moment.
- End with a final, cold, damning line from the tormentor — a verdict on their failure,
  not comfort.
- Do NOT mention dice, DC, or numeric mechanics even here.

## CHOICES
Always exactly 4. No lettering/numbering prefixes.

Each choice must be a SPECIFIC action tied to concrete details of THIS scene (the
creature/NPC/object/environment just described) — never a generic verb alone.
- BAD (too generic): "Attack", "Investigate the area", "Try to negotiate", "Run away"
- GOOD (specific to scene): "Lao vào ném dao nhằm mắt con quái trước khi nó vồ tới",
  "Giả vờ đồng ý giao vàng để dụ tên cướp lại gần trong tầm kiếm", "Hỏi lão già vì sao
  tay ông ta run khi nhắc tới cánh cửa", "Nhảy qua khe nứt để thoát khỏi luồng khí độc"

Vary the UNDERLYING approach across the 4 choices (combat/stealth/diplomacy/
investigation/escape/trickery/sacrifice-a-resource), but do NOT force all 5 categories
every turn — pick whichever 4 make sense for the current threat/NPC/object. A pure
combat scene may reasonably have 2 combat-flavored choices (e.g. aggressive vs
defensive) plus 1 escape + 1 clever/desperate option, as long as they read as
genuinely different plans, not reworded synonyms of each other.

Avoid repeating the wording, sentence structure, or core tactical idea
from recent turns and from the immediately previous turn.

When a similar action is necessary, introduce a meaningful tactical difference
or a different objective.

If the previous action failed, at least one choice must directly address the new
consequence (retreat, defend, counter-attack, improvise, use a specific item/skill
from the sheet).

At least one of the 4 choices each turn should involve risk/cost beyond combat damage
(e.g. spending an item, provoking a worse outcome, moral compromise, revealing
information) to avoid choices feeling mechanically identical turn after turn.

At least ONE of the 4 choices each turn should leverage something SPECIFIC to this
character's identity — their race, class, a strength, or a weakness from the sheet —
rather than a generic option anyone could take. Do not force this if the scene truly
gives no plausible opening (e.g. mid-fall, no agency), but actively look for one.
- Race-flavored example: a Dwarf "xem xét kết cấu đá để tìm điểm yếu" (dwarven stonework
  instinct); an Elf "để dòng máu tiên tộc trấn an trước ảo ảnh" (fey resilience).
- Class-flavored example: a Rogue "lặng lẽ cạy khóa thay vì phá cửa"; a Cleric "cầu
  nguyện để giữ vững tinh thần trước nỗi sợ".
- Strength/weakness-flavored: reference the exact sheet entry concretely (already
  required elsewhere in this prompt) — don't invent a trait not on the sheet.
Keep it thematically plausible for the race/class, not mechanically precise 5e rules.

## SCENE CONTINUITY
You will receive [CURRENT SCENE STATE] each turn — this is the ONLY source of truth for where
the character physically is. NEVER move the character backward to a previously resolved
location/puzzle/obstacle unless they explicitly choose to retreat. Once a door/lock/trap is
resolved, it stays resolved — do not reintroduce it. Always advance scene_state forward.

Before narrating a new scene, first imagine an interesting situation rather than a quest.

A good situation should usually contain:

- An opportunity or reward.
- An obstacle or danger.
- A mystery, uncertainty, or hidden truth.
- A source of tension that may worsen if ignored.

The player should immediately want to make a meaningful decision.

## NARRATION STYLE (story text only)
- Jump straight into events — sensory detail, action, or world reaction. Do NOT start the story sentence with
  "[CharacterName] + verb" (e.g. "{c['name']} rút kiếm...").
- NEVER call the player "bạn" / "you" in story text.
- When the character must be referenced, use their proper name ({c['name']}) — NOT a generic
  class/race label like "tên chiến binh"/"người lùn"/"kẻ lẩn trốn". Use it sparingly; prefer
  describing what happens (blade swings, foot slips, door groans) over labeling who acts, but
  when you do need to name who acts, it's always {c['name']}.
- BAD openings: "Bạn cảm thấy lạnh...", "Tên chiến binh rút kiếm..." (never lead with name/label + verb).
- GOOD openings: "Khói mùi lưu huỳnh bốc lên từ khe đá.", "Mũi kiếm vạt ngang bụng con quái.",
  "Tiếng gõ cửa vang lên — ba nhịp, rồi im bặt."

## ENTITIES (NPCs/monsters) — PERSISTENCE RULES
- mechanics.entities is OPTIONAL and only needed when: (a) a new NPC/monster appears this
  turn, or (b) an EXISTING one (listed in ACTIVE ENTITIES context) takes damage/heals/dies/flees.
- NEW entity: include key (short snake_case, unique), name (English), type ("npc"|"monster"),
  max_hp, hp (usually = max_hp), hostile (true/false).
- NEW hostile monster source: if the CAMPAIGN section above includes a MONSTER ROSTER, a new
  hostile monster MUST come from that roster (reuse its exact name/appearance/moveset/
  behavior) — do not default to a generic/unrelated creature out of habit when the roster
  already has something that fits.
- CONSISTENCY (critical): the creature/NPC you write into mechanics.entities MUST be the
  EXACT same creature you just described in the story text this turn — same species/kind,
  same weapon/method of attack. If the story says a snake bit/coiled around the character,
  name MUST be a snake (e.g. "Venomous Serpent"), never an unrelated creature like a spider.
  Never invent a name/species that contradicts what the story prose just narrated.
- EXISTING entity (its key already appears in ACTIVE ENTITIES): include ONLY key and
  hp_change (negative=damage dealt to it, positive=healing). Do NOT re-send max_hp/type/name
  for existing entities. Add "status": "dead"/"fled" when applicable. The story text must
  keep describing it as the SAME creature it was introduced as — never switch species mid-scene.
- Never invent stats for an entity already listed in ACTIVE ENTITIES — use its key as given.
- Do not use Vietnamese for monster name.

## LOOT — PERSISTENCE RULES
- mechanics.loot_dropped: only populate when the story text this turn explicitly shows an
  item becoming available in the world (monster killed and drops something, a chest is
  opened, an item is found). name = English item name, source_key = the entity key it
  dropped from (or null if not from a creature).
- items_added in changes must only reference items that were dropped THIS turn via
  loot_dropped, OR that already appear in the LOOT AVAILABLE list from context, OR are a
  direct narrative reward (quest gold/item) clearly justified by the story — never invent
  combat loot without declaring it via loot_dropped first.

## LANGUAGE
Output must be 100% Vietnamese. Never switch to another language for any reason, including
internal reasoning, even if you notice unclear or conflicting instructions — just make
the most reasonable interpretation silently and continue in Vietnamese.
Only keep monsters and locations name in English.


## OUTPUT FORMAT (Vietnamese only for story/choices, JSON keys in English)
{{
  "story": "...",
  "mechanics": {{
    "success": true,
    "roll_type": "normal",
    "reasoning": "",
    "event_occurred": false,
    "character_died": false,
    "changes": {{"hp": 0, "mana": 0, "gold": 0, "xp": 0, "items_added": [], "items_removed": []}},
    "entities": [
      {{
        "key": "<unique_entity_key>",
        "name": "<entity_name>",
        "type": "monster|npc|companion|object",
        "max_hp": 0,
        "hp": 0,
        "hostile": false,
        "status": "alive|dead"
      }}
    ],
    "loot_dropped": [
        {{
        "name": "<item_name>",
        "source_key": "<entity_key>"
      }}
    ]
  }},
  "choices": [
    {{"text": "...", "needs_roll": true, "roll": "advantage", "reason": {{"type": "attribute", "name": "WIS"}}}}
    {{"text": "...", "needs_roll": true, "roll": "advantage", "reason": {{"type": "race", "name": "Elf"}}}}
    {{"text": "...", "needs_roll": true, "roll": "disadvantage", "reason": {{"type": "class", "name": "Fighter"}}}}
    {{"text": "...", "needs_roll": false, "roll": "normal", "reason": null}}
  ]
}}
roll must be advantage/disadvantage/normal. If normal, reason is null. If advantage/disadvantage, reason.type is one of attribute/strength/weakness/skill/item/race/class, citing exactly one real value from the sheet.
needs_roll: true if the choice has any real chance of failure worth rolling for; false for a
pure flavor/no-risk choice (e.g. "quan sát xung quanh", "im lặng lắng nghe") that always succeeds.
IMPORTANT: these needs_roll/roll/reason values are NOT just flavor — if the player picks this
exact choice next turn, the backend reuses them AS-IS to resolve the roll (no re-classification),
so they must already reflect your true, final judgement for that action.


"""


# ---------------------------------------------------------------------------
# Chat / gameplay
# ---------------------------------------------------------------------------

def _parse_dm_json(reply: str) -> dict:
    try:
        clean_reply = reply.replace("```json", "").replace("```", "").strip()
        return json.loads(clean_reply)
    except json.JSONDecodeError:
        return {"story": reply, "mechanics": {"roll_type": "normal", "reasoning": ""}, "choices": []}


@app.post("/chat")
async def chat(data: dict):
    user_input = data.get("message", "")
    char = get_latest_character()
    if not char:
        return {"error": "Chưa có nhân vật."}
    char_dict = character_row_to_dict(char)
    suspicious = _mentions_missing_item(user_input, char_dict)

    turns_since_event = char["turns_since_event"] or 0
    force_event = turns_since_event >= 2  # giảm từ 2 xuống 1 — chỉ cho phép đúng 1 lượt "thở"

    turn_note = f"[STATE] turns_since_major_event={turns_since_event}."
    if force_event:
        turn_note += (
            " A major event (combat, ambush, hostile NPC, trap triggering, or critical "
            "discovery) MUST occur THIS turn. Do NOT offer another search/examine/investigate "
            "choice as a safe option — every choice this turn must carry real risk or "
            "immediately escalate the danger."
        )
    if suspicious:
        turn_note += (" WARNING: the player's action may reference an item/skill/weapon "
                    "not present on the character sheet. Verify against Equipment/Items/Skills "
                    "before resolving. If not owned, success MUST be false and the character "
                    "must suffer a penalty for hesitating (e.g. HP loss from being struck).")

    # --- Call 1: classify (module classification.py) ---
    prev_result_raw = char["last_result"] if "last_result" in char.keys() else None
    known_choices = None
    if prev_result_raw:
        try:
            known_choices = json.loads(prev_result_raw).get("choices")
        except (TypeError, json.JSONDecodeError, AttributeError):
            known_choices = None

    class_result = classification.classify_action(
        MODEL, OPTIONS, user_input, char_dict, known_choices=known_choices
    )

    def _localize_reason_name(reason_type, name, char_dict):
        """Model chỉ biết tên EN của item/skill/trait — map ngược lại tên tiếng Việt
        để hiển thị cho người chơi."""
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


    def _localize_choices(choices, char_dict):
        for ch in choices or []:
            if not isinstance(ch, dict):
                continue  # phòng model trả sai schema (vd list string thay vì object)
            reason = ch.get("reason")
            if reason and reason.get("name"):
                reason["name"] = _localize_reason_name(reason.get("type"), reason.get("name"), char_dict)
        return choices

    # --- Kiểm tra tài nguyên + tung xúc xắc thật (module classification.py) ---
    resolution = classification.resolve_action(class_result, char_dict)

    success = resolution["success"]
    roll_type = resolution["roll_type"]
    dice = resolution["dice"]
    dc = resolution["dc"]
    attribute = resolution["attribute"]
    used_name = resolution["used_name"]
    consumed_kind = resolution["consumed_kind"]
    mana_cost = resolution["mana_cost"]
    resource_note = resolution["resource_note"]
    adv_reason = resolution["adv_reason"]

    # --- Tiêu hao tài nguyên trong DB (bất kể thành công hay fail) ---
    if consumed_kind == "item":
        _consume_item(char["id"], used_name)
    elif consumed_kind == "skill":
        _put_skill_on_cooldown(char["id"], used_name)
        _consume_mana(char["id"], mana_cost)

    # Hồi cooldown các skill khác đi 1 lượt
    _tick_cooldowns(char["id"], skip=used_name if consumed_kind == "skill" else None)

    reason_str = ""
    if adv_reason:
        reason_str = f" (due to {adv_reason['type']}: {adv_reason['name']})"

    dice_fact = (
        f"INTERNAL MECHANICS (never mention numbers/DC/roll in story): "
        f"outcome={'SUCCESS' if success else 'FAIL'}, roll_type={roll_type}{reason_str}. "
        f"If roll_type is disadvantage/advantage, the story may subtly hint at the reason "
        f"(e.g. character struggling due to their weakness) WITHOUT naming stats or rules."
    )

    if resource_note:
        dice_fact += (
            f" RESOURCE FAILURE REASON (narrate this specific cause, not a generic miss): "
            f"{resource_note}"
        )

    current_hp = char["hp"]

    dice_fact += (
        f" Character's CURRENT HP before this turn: {current_hp}/{char['max_hp']}. "
        f"If your hp change this turn would bring HP to 0 or below, you MUST narrate the "
        f"character's death explicitly in the story (final blow, collapse, darkness taking "
        f"them) — do not continue the adventure past this point. Set mechanics.changes.hp "
        f"such that final HP does not go below 0 (clamp your intended damage if needed)."
    )

    dice_fact += (
        f" REMINDER: hp change must be NEGATIVE if the story shows the character taking "
        f"damage, POSITIVE only for healing, ZERO for no change. Success={success} means "
        f"the character's intended action actually works — narrate accordingly. Do not negative hp if Success=true"
    )

    system_prompt = build_system_prompt(char)  # bản gọn ở tin nhắn trước

    conn = get_conn()
    c = conn.cursor()

    # turn_number: tăng dần từ current_turn đã lưu, dùng để đánh dấu
    # last_seen_turn cho entity và để summarizer biết turn nào chưa gộp.
    turn_number = (char["current_turn"] or 0) + 1

    entities_context = entities.format_entities_context(conn, char["id"])

    # RAG lore context + world state (weather/day-night) TẠM THỜI TẮT — không
    # build/gọi lore.format_lore_context() hay world_state.roll_weather() nữa,
    # để model đỡ phải đọc thêm lore mỗi turn (giảm tải context/tốc độ). Bật
    # lại bằng cách khôi phục khối code cũ (xem git history) khi cần.
    region = char_dict.get("region")
    lore_context = None
    world_state_context = None

    summarized_up_to_turn = char["summarized_up_to_turn"] or 0
    history_summary = char["history_summary"] or ""

    # Nếu số turn CHƯA gộp (tính tới trước turn hiện tại) đã vượt ngưỡng ->
    # tóm tắt lại, đẩy summarized_up_to_turn lên. Làm TRƯỚC khi build context
    # cho turn hiện tại để cửa sổ verbatim luôn nhỏ gọn.
    if summarizer.needs_summarization(char["current_turn"] or 0, summarized_up_to_turn):
        rows_to_fold = c.execute(
            "SELECT role, content FROM history WHERE turn_number > ? AND turn_number <= ? ORDER BY id ASC",
            (summarized_up_to_turn, char["current_turn"] or 0),
        ).fetchall()
        if rows_to_fold:
            events_text = summarizer.format_events_for_summarization(rows_to_fold)
            history_summary = summarizer.update_summary(history_summary, events_text)
            summarized_up_to_turn = char["current_turn"] or 0
            c.execute(
                "UPDATE character SET history_summary = ?, summarized_up_to_turn = ? WHERE id = ?",
                (history_summary, summarized_up_to_turn, char["id"]),
            )
            conn.commit()

    # Chỉ lấy các turn CHƯA gộp vào summary, nguyên văn (thường rất ít dòng
    # vì vừa tóm tắt xong ở trên nếu cần)
    history_rows = c.execute(
        "SELECT role, content FROM history WHERE turn_number > ? ORDER BY id ASC",
        (summarized_up_to_turn,),
    ).fetchall()
    conn.close()

    summary_context = summarizer.format_summary_context(history_summary)

    messages = [{"role": "system", "content": system_prompt}]
    if lore_context:
        messages.append({"role": "user", "content": lore_context})
    if entities_context:
        messages.append({"role": "user", "content": entities_context})
    if world_state_context:
        messages.append({"role": "user", "content": world_state_context})
    if summary_context:
        messages.append({"role": "user", "content": summary_context})
    for row in history_rows:
        messages.append({"role": row["role"], "content": row["content"]})
    messages.append({
        "role": "user",
        "content": f"{turn_note}\n\n{dice_fact}\n\nPlayer action: {user_input}\n\n"
                    f"mechanics.success MUST be {str(success).lower()}. "
                    f"mechanics.roll_type MUST be \"{roll_type}\"."
    })
    messages.append({
        "role": "user",
        "content": f"""
        Never narrate the player's internal thoughts.

        Never narrate the player's emotions.

        Never narrate the player's intentions.

        Never continue the player's action beyond what they explicitly stated.

        Describe only the world's reaction.

        NARRATION: Do not use "bạn". Do not open with "[name] + verb" — lead with the
        event/environment/action unfolding. When you do need to refer to who acts, use
        {char_dict['name']}, never a generic class/race label.
"""
    })

    response = ollama.chat(
        model=MODEL,
        messages=messages,
        format="json",
        options=OPTIONS,
        think=False
    )
    result = _parse_dm_json(response["message"]["content"])

    # Ép cứng lại success/roll_type theo xúc xắc thật, phòng model vẫn cãi
    result.setdefault("mechanics", {})
    result["mechanics"]["success"] = success
    result["mechanics"]["roll_type"] = roll_type
    if dice:
        result["mechanics"]["dice"] = dice
        result["mechanics"]["dc"] = dc
        result["mechanics"]["attribute"] = attribute

    if "changes" in result["mechanics"]:
        changes = result["mechanics"]["changes"]
        hp_delta = safe_int(changes.get("hp", 0))
        if not success and hp_delta > 0:
            print(f"[DEBUG] success=False nhưng hp=+{hp_delta} -> ép về -{abs(hp_delta) or 5}")
            changes["hp"] = -abs(hp_delta) if hp_delta != 0 else -5
        elif success and hp_delta < 0:
            changes["hp"] = -abs(hp_delta) if hp_delta != 0 else 10

        # --- RAG: entity/loot xử lý TRƯỚC khi apply items_added, vì cần
        # ledger loot mới nhất để validate ---
        conn2 = get_conn()
        entities.apply_entity_changes(
            conn2, char["id"], result["mechanics"].get("entities"), turn_number
        )
        entities.register_loot_drops(
            conn2, char["id"], result["mechanics"].get("loot_dropped"), turn_number
        )
        items_added = changes.get("items_added") or []
        validated_items, unverified = entities.validate_items_added(conn2, char["id"], items_added)
        changes["items_added"] = validated_items
        if unverified:
            print(f"[DEBUG] items_added không khớp loot ledger (vẫn cho qua): {unverified}")
        conn2.close()

        apply_changes_to_db(char["id"], result["mechanics"]["changes"])

        if mana_cost is not None and mana_cost > 0:
            print(f"mana_cost: {mana_cost}")
            changes["mana"] = -mana_cost  # ép cứng mana trừ đi, không cho model tự bịa +mana
        
        if used_name is not None:
            changes["items_removed"] = [used_name]

    result["choices"] = _localize_choices(result.get("choices"), char_dict)

    character_died = result["mechanics"].get("character_died", False)

    conn = get_conn()
    c = conn.cursor()
    updated_hp = c.execute("SELECT hp FROM character WHERE id=?", (char["id"],)).fetchone()["hp"]
    conn.close()

    # Ép cứng: nếu model báo chết HOẶC HP thực tế đã <=0 -> chốt HP về 0, khóa choices
    if character_died or updated_hp <= 0:
        conn = get_conn()
        c = conn.cursor()
        c.execute("UPDATE character SET hp = 0 WHERE id = ?", (char["id"],))
        conn.commit()
        conn.close()
        result["mechanics"]["character_died"] = True
        result["mechanics"]["is_dead"] = True
        result["choices"] = []
    else:
        result["mechanics"]["is_dead"] = False

    conn = get_conn()
    c = conn.cursor()
    c.execute(
        "INSERT INTO history (role, content, turn_number) VALUES ('user', ?, ?)",
        (user_input, turn_number),
    )
    c.execute(
        "INSERT INTO history (role, content, turn_number) VALUES ('assistant', ?, ?)",
        (result.get("story", ""), turn_number),
    )
    c.execute(
        "UPDATE character SET last_result = ?, current_turn = ? WHERE id = ?",
        (json.dumps(result, ensure_ascii=False), turn_number, char["id"]),
    )
    conn.commit()
    conn.close()

    return result


@app.get("/start_game")
async def start_game():
    char = get_latest_character()
    if not char:
        return {"story": "Chưa có nhân vật.", "mechanics": {}, "choices": []}

    # --- RESUME: nếu region đã được chốt từ trước, nghĩa là /start_game đã
    # từng chạy cho save-slot này (đang chơi dở, có thể do frontend reload
    # trang / mất kết nối rồi gọi lại /start_game) -> tiếp tục phiên cũ,
    # KHÔNG tạo scene mở đầu mới (tránh việc model "quên" và narrate lại từ
    # đầu trong khi history/DB vẫn còn nguyên state cũ, gây gãy mạch truyện). ---
    region = char["region"] if "region" in char.keys() and char["region"] else None
    if region:
        last_result_raw = char["last_result"] if "last_result" in char.keys() else None
        if last_result_raw:
            try:
                return json.loads(last_result_raw)
            except (TypeError, json.JSONDecodeError):
                pass  # last_result hỏng/thiếu -> rơi xuống fallback bên dưới

        # Fallback cho save cũ (tạo trước khi có cột last_result): dựng lại
        # state tối thiểu từ history + character sheet thay vì bịa scene mới.
        last_story = _get_last_story() or "Cuộc phiêu lưu đang tiếp diễn..."
        is_dead = char["hp"] <= 0
        return {
            "story": last_story,
            "mechanics": {
                "success": True,
                "roll_type": "normal",
                "character_died": is_dead,
                "is_dead": is_dead,
                "changes": {"hp": 0, "mana": 0, "gold": 0, "xp": 0, "items_added": [], "items_removed": []},
            },
            "choices": [] if is_dead else [
                {"text": "Quan sát xung quanh trước khi tiếp tục", "roll": "normal", "reason": None},
            ],
        }

    # --- Chưa từng chơi (region chưa chốt) -> chốt region ngẫu nhiên, tạo scene mở đầu mới ---
    region = lore.pick_random_region()
    conn = get_conn()
    c = conn.cursor()
    c.execute("UPDATE character SET region = ? WHERE id = ?", (region, char["id"]))
    conn.commit()
    conn.close()
    char = get_latest_character()  # reload để có region mới

    system_prompt = build_system_prompt(char)
    opening_char = character_row_to_dict(char)

    # Ép AI trả về JSON ngay cả trong màn mở đầu
    opening_instruction = f"""
Start a brand-new Dungeons & Dragons 5e campaign.
This is the opening scene only.
The adventure takes place in the {region} region of the Forgotten Realms. Invent a fitting
minor location within this region.
Open by plunging into the scene — location, atmosphere, or immediate tension first.

EXCEPTION TO NARRATION STYLE — this opening scene only: within the first couple of
sentences you MUST explicitly establish, in plain narration or through an NPC/narrator line:
- the character's name: {opening_char['name']}
- their race: {opening_char['race']} — describe a concrete visual/physical trait of this race
  (not just naming it in passing)
- their class: {opening_char['character_class']} — describe a concrete gear/skill/demeanor
  detail that signals this class
Simply using the name later in the scene is NOT enough — race and class must each be
clearly conveyed, either by stating the word directly or through unmistakable descriptive
detail a reader would recognize as that race/class. Every later turn goes back to the
normal NARRATION STYLE (name only, no need to restate race/class).

All encountered monsters and locations should be in the Forgotten Realms.
All monster and location names must be English.
"""

    # RAG lore context + world state (weather/day-night) TẠM THỜI TẮT — xem
    # ghi chú tương tự trong /chat. region vẫn được chốt/lưu bình thường vì
    # opening_instruction cần nó để đặt bối cảnh mở đầu.

    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": opening_instruction},
        ],
        format="json",
        options=OPTIONS,
        think=False
    )
    reply = response["message"]["content"]

    result = _parse_dm_json(reply)
    result.setdefault("mechanics", {})

    # RAG: nếu scene mở đầu đã giới thiệu quái/NPC/loot ngay lập tức, vẫn phải
    # lưu vào DB — nếu không, lượt /chat kế tiếp sẽ không nhận ra key đó đã tồn tại.
    conn = get_conn()
    entities.apply_entity_changes(conn, char["id"], result["mechanics"].get("entities"), 0)
    entities.register_loot_drops(conn, char["id"], result["mechanics"].get("loot_dropped"), 0)
    conn.close()

    # Lưu vào DB (bao gồm last_result để /start_game gọi lại sau này resume đúng)
    conn = get_conn()
    c = conn.cursor()
    # c.execute("INSERT INTO history (role, content) VALUES ('user', ?)", (opening_instruction,))
    c.execute(
        "INSERT INTO history (role, content, turn_number) VALUES ('assistant', ?, 0)",
        (result.get("story", ""),),
    )
    c.execute(
        "UPDATE character SET last_result = ?, current_turn = 0 WHERE id = ?",
        (json.dumps(result, ensure_ascii=False), char["id"]),
    )
    conn.commit()
    conn.close()

    return result  # Trả về cùng cấu trúc với /chat


def apply_changes_to_db(char_id, changes):
    conn = get_conn()
    c = conn.cursor()

    # 1. Update các trường số (HP, Mana, Gold, XP) — chặn cả trên lẫn dưới 0
    c.execute("""
        UPDATE character
        SET hp = MAX(0, MIN(max_hp, hp + ?)),
            mana = MAX(0, MIN(max_mana, mana + ?)),
            gold = MAX(0, gold + ?),
            xp = MAX(0, xp + ?)
        WHERE id = ?
    """, (
        safe_int(changes.get("hp", 0)),
        safe_int(changes.get("mana", 0)),
        safe_int(changes.get("gold", 0)),
        safe_int(changes.get("xp", 0)),
        char_id,
    ))

    items_added = changes.get("items_added") or []
    if items_added:
        row = c.execute("SELECT items FROM character WHERE id = ?", (char_id,)).fetchone()
        items = [_normalize_item(i) for i in _load_json(row["items"])]

        for name in items_added:
            name = (name or "").strip()
            if not name:
                continue
            existing = next((it for it in items if _item_matches(it, name)), None)
            if existing:
                existing["quantity"] = existing.get("quantity", 1) + 1
            else:
                # AI chỉ biết tiếng Anh nên vi/en tạm giống nhau cho vật phẩm mới nhặt được.
                items.append({"key": None, "vi": name, "en": name, "consumable": True, "quantity": 1})

        c.execute("UPDATE character SET items = ? WHERE id = ?", (_list_json(items), char_id))

    conn.commit()
    conn.close()

def _mentions_missing_item(user_input: str, char_dict: dict) -> str | None:
    """Heuristic: nếu câu lệnh có dạng 'dùng/sử dụng/tấn công bằng <X>' mà X
    không khớp equipment/items/skills nào của nhân vật -> trả về tên X.
    Không hoàn hảo nhưng đủ để chặn case rõ ràng (sai tên vũ khí, bịa skill)."""
    import re
    triggers = ["dùng", "sử dụng", "use", "cast", "attack with", "tấn công bằng"]
    lowered = user_input.lower()
    if not any(t in lowered for t in triggers):
        return None

    owned_names = set()
    for bucket in ("equipment", "items", "skills"):
        for it in char_dict.get(bucket, []):
            if isinstance(it, dict):
                for k in ("en", "vi", "key"):
                    if it.get(k):
                        owned_names.add(it[k].strip().lower())
            else:
                owned_names.add(str(it).strip().lower())

    # nếu không tên nào trong owned_names xuất hiện trong câu -> nghi ngờ có item lạ
    if owned_names and not any(name in lowered for name in owned_names):
        return user_input  # để model tự detect chi tiết tên, ta chỉ báo "nghi vấn"
    return None

def _get_last_story(char_id=None):
    """Lấy story text của lượt gần nhất. LƯU Ý: content trong bảng history vốn
    đã là plain story text (xem chỗ INSERT ở /chat và /start_game), KHÔNG
    phải JSON — bản cũ của hàm này gọi json.loads() lên plain text, luôn ném
    JSONDecodeError và âm thầm trả về None (bug), khiến RAG lore context mất
    tín hiệu "last_story" khi tính relevance. Sửa lại: đọc trực tiếp."""
    conn = get_conn(); c = conn.cursor()
    row = c.execute(
        "SELECT content FROM history WHERE role='assistant' ORDER BY id DESC LIMIT 1"
    ).fetchone()
    conn.close()
    if not row:
        return None
    return row["content"] or None

# build_classify_prompt() và _reason_is_valid() đã chuyển sang classification.py

def _normalize_item(it):
    if isinstance(it, str):
        it = {"key": None, "vi": it, "en": it}
    # Bucket "items" (khác với "equipment" vốn là trang bị vĩnh viễn) mặc định
    # là vật phẩm tiêu hao (potion, cuộn giấy...). Trước đây default=False khiến
    # _consume_item không bao giờ thực sự trừ số lượng dù dùng đúng item.
    it.setdefault("consumable", True)
    it.setdefault("quantity", 1)
    return it

def _normalize_skill(sk):
    if isinstance(sk, str):
        sk = {"key": None, "vi": sk, "en": sk}
    # Frontend (gamedata.js) gửi field "cooldown" (số lượt hồi chiêu thiết kế sẵn
    # cho mỗi skill). Trước đây field này KHÔNG được map sang "cooldown_max" nên
    # setdefault luôn nhét giá trị 0 -> cooldown không bao giờ thực sự áp dụng.
    if "cooldown_max" not in sk and sk.get("cooldown") is not None:
        sk["cooldown_max"] = safe_int(sk.get("cooldown"), 0)
    sk.setdefault("cooldown_max", 0)     # 0 = không giới hạn lượt hồi
    sk.setdefault("cooldown_current", 0) # >0 nghĩa là đang "nghỉ", chưa dùng lại được
    return sk

def _consume_mana(char_id, amount):
    """Trừ mana trực tiếp trong DB khi dùng skill. Trước đây việc trừ mana
    hoàn toàn phụ thuộc vào model tự ý ghi mechanics.changes.mana trong JSON
    trả về — không có gì ép buộc, nên rất hay bị bỏ sót (dùng skill mà mana
    không giảm). Giờ backend chủ động trừ ngay khi skill thực sự được dùng,
    không phụ thuộc vào model nữa."""
    amount = safe_int(amount, 0)
    if amount <= 0:
        return
    conn = get_conn(); c = conn.cursor()
    c.execute("UPDATE character SET mana = MAX(0, mana - ?) WHERE id = ?", (amount, char_id))
    conn.commit(); conn.close()


def _consume_item(char_id, name):
    conn = get_conn(); c = conn.cursor()
    row = c.execute("SELECT items FROM character WHERE id=?", (char_id,)).fetchone()
    items = [_normalize_item(i) for i in _load_json(row["items"])]
    for it in items:
        if _item_matches(it, name):
            if it.get("consumable", False):
                it["quantity"] = max(0, it.get("quantity", 1) - 1)
            break
    items = [it for it in items if not (it.get("consumable") and it.get("quantity", 1) <= 0)]
    c.execute("UPDATE character SET items=? WHERE id=?", (_list_json(items), char_id))
    conn.commit(); conn.close()


def _put_skill_on_cooldown(char_id, name):
    conn = get_conn(); c = conn.cursor()
    row = c.execute("SELECT skills FROM character WHERE id=?", (char_id,)).fetchone()
    skills = [_normalize_skill(s) for s in _load_json(row["skills"])]
    for sk in skills:
        if _item_matches(sk, name) and sk.get("cooldown_max", 0) > 0:
            sk["cooldown_current"] = sk["cooldown_max"]
            break
    c.execute("UPDATE character SET skills=? WHERE id=?", (_list_json(skills), char_id))
    conn.commit(); conn.close()


def _tick_cooldowns(char_id, skip=None):
    conn = get_conn(); c = conn.cursor()
    row = c.execute("SELECT skills FROM character WHERE id=?", (char_id,)).fetchone()
    skills = [_normalize_skill(s) for s in _load_json(row["skills"])]
    for sk in skills:
        if skip and _item_matches(sk, skip):
            continue
        if sk.get("cooldown_current", 0) > 0:
            sk["cooldown_current"] -= 1
    c.execute("UPDATE character SET skills=? WHERE id=?", (_list_json(skills), char_id))
    conn.commit(); conn.close()

# attr_modifier() và roll_d20() đã chuyển sang classification.py