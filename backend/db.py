"""
db.py — SQLite access + character-row helpers dùng chung giữa main.py, dm.py,
assistant.py. Tách riêng khỏi main.py để main.py chỉ còn chứa route handlers.
"""

import json
import sqlite3

from agents import classification, entities, campaign

DB_PATH = "game.db"


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
    int_cols_default_0 = (
        "turns_since_event", "weather_since_turn", "current_turn", "summarized_up_to_turn",
        "campaign_milestone_index", "milestone_advanced_turn",
        "campaign_act_index", "campaign_milestone_number",
    )
    for col in ("race_en", "character_class_en", "turns_since_event", "region", "npc_pool",
                "last_result", "weather", "weather_since_turn", "current_turn",
                "history_summary", "summarized_up_to_turn", "campaign_theme", "campaign_data",
                "campaign_milestone_index", "pre_turn_snapshot", "milestone_advanced_turn",
                "pending_action", "last_turn_resolution",
                "context_kind", "context_name", "context_desc", "context_image_path",
                "campaign_act_index", "campaign_milestone_number", "current_milestone",
                "story_state", "setup_stage", "setup_error"):
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


def _load_campaign_bible(char) -> dict | None:
    """Campaign Bible giờ ưu tiên đọc từ file JSON riêng (xem agents/campaign.py:
    save/load_campaign_bible) — dễ mở lên xem/theo dõi hơn hẳn so với 1 cột
    TEXT lớn trong SQLite. Fallback về cột campaign_data cũ trong DB chỉ để
    đọc ngược các save được tạo TRƯỚC khi có cơ chế file này."""
    bible = campaign.load_campaign_bible()
    if bible:
        return bible
    campaign_data_raw = char["campaign_data"] if "campaign_data" in char.keys() else None
    if campaign_data_raw:
        try:
            return json.loads(campaign_data_raw)
        except (TypeError, json.JSONDecodeError):
            return None
    return None


def save_current_milestone(char_id: int, milestone: dict):
    """Milestone hiện tại (khác Campaign Bible — đổi liên tục mỗi khi hoàn
    thành 1 cái, không tĩnh như bible) — lưu thẳng vào cột DB (JSON), không
    cần file riêng."""
    conn = get_conn()
    conn.execute(
        "UPDATE character SET current_milestone = ? WHERE id = ?",
        (json.dumps(milestone, ensure_ascii=False), char_id),
    )
    conn.commit()
    conn.close()


def load_current_milestone(char) -> dict | None:
    raw = char["current_milestone"] if "current_milestone" in char.keys() else None
    if not raw:
        return None
    try:
        return json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return None


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
        "ac": 10 + classification.attr_modifier(char["attr_dex"]),
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
