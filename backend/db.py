"""
db.py — SQLite access + character-row helpers dùng chung giữa main.py, dm.py,
assistant.py. Tách riêng khỏi main.py để main.py chỉ còn chứa route handlers.

2 bảng chính, tách RÕ 2 khái niệm khác nhau:
- `character`: CHỈ thông tin nhân vật (tên/chỉ số/trang bị/kỹ năng...) — thứ
  người chơi thực sự "sở hữu", gần như tĩnh trong 1 phiên chơi.
- `campaign_state`: TOÀN BỘ state của phiên chơi/campaign hiện tại (turn số
  mấy, đang ở milestone nào, story_state, context panel, pending_action...)
  — đổi liên tục mỗi turn, không phải "thông tin nhân vật". 1:1 với character
  (character_id làm PK luôn, không cần AUTOINCREMENT riêng vì mỗi lúc chỉ có
  đúng 1 character — xem "single save-slot" trong main.py).
"""

import json
import sqlite3

from agents import classification, entities, campaign

DB_PATH = "game.db"

# Cột nào của campaign_state PHẢI sống sót qua pre-turn restore (retry) —
# xem _snapshot_pre_turn/_restore_pre_turn ở dungeon_master.py.
_STATE_SURVIVE_RESTORE = ("pending_action", "pre_turn_snapshot", "last_turn_resolution")


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
            content TEXT,
            turn_number INTEGER DEFAULT 0
        )
    """)

    # Xem docstring đầu file — toàn bộ state "phiên chơi", KHÔNG phải "nhân vật".
    c.execute("""
        CREATE TABLE IF NOT EXISTS campaign_state(
            character_id INTEGER PRIMARY KEY,
            region TEXT,
            campaign_theme TEXT,
            current_turn INTEGER DEFAULT 0,
            history_summary TEXT,
            summarized_up_to_turn INTEGER DEFAULT 0,
            campaign_act_index INTEGER DEFAULT 0,
            campaign_milestone_number INTEGER DEFAULT 0,
            milestone_advanced_turn INTEGER DEFAULT 0,
            current_milestone TEXT,
            story_state TEXT,
            last_result TEXT,
            last_turn_resolution TEXT,
            pending_action TEXT,
            pre_turn_snapshot TEXT,
            context_kind TEXT,
            context_name TEXT,
            context_name_vi TEXT,
            context_desc TEXT,
            context_image_path TEXT,
            setup_stage TEXT,
            setup_error TEXT
        )
    """)
    conn.commit()

    # Migration: DB tạo trước khi có cột context_name_vi (tên hiển thị tiếng
    # Việt cho context panel — context_name vẫn giữ tên English canonical dùng
    # cho cache ảnh/khớp race trong imagegen).
    existing_state_cols = {row["name"] for row in c.execute("PRAGMA table_info(campaign_state)")}
    if "context_name_vi" not in existing_state_cols:
        c.execute("ALTER TABLE campaign_state ADD COLUMN context_name_vi TEXT")
        conn.commit()

    # Migration: DB tạo trước khi có bảng history.turn_number.
    existing_history_cols = {row["name"] for row in c.execute("PRAGMA table_info(history)")}
    if "turn_number" not in existing_history_cols:
        c.execute("ALTER TABLE history ADD COLUMN turn_number INTEGER DEFAULT 0")
        conn.commit()

    # Migration: DB tạo TRƯỚC khi tách bảng campaign_state (các cột session
    # từng nằm lẫn trong character) -> copy sang bảng mới cho campaign đang
    # chơi dở, rồi thôi không đụng cột cũ đó nữa (không DROP COLUMN — không
    # phải bản SQLite nào cũng hỗ trợ tốt, để lại vô hại vì code không còn
    # đọc/ghi chúng).
    existing_char_cols = {row["name"] for row in c.execute("PRAGMA table_info(character)")}
    legacy_session_cols = [
        "region", "campaign_theme", "current_turn",
        "history_summary", "summarized_up_to_turn", "campaign_act_index",
        "campaign_milestone_number", "milestone_advanced_turn", "current_milestone",
        "story_state", "last_result", "last_turn_resolution", "pending_action",
        "pre_turn_snapshot", "context_kind", "context_name", "context_desc",
        "context_image_path", "setup_stage", "setup_error",
    ]
    if any(col in existing_char_cols for col in legacy_session_cols):
        rows = c.execute("SELECT * FROM character").fetchall()
        for row in rows:
            already = c.execute(
                "SELECT 1 FROM campaign_state WHERE character_id = ?", (row["id"],)
            ).fetchone()
            if already:
                continue  # server đã chạy lại sau lần migrate trước -> bỏ qua
            row_keys = row.keys()
            values = [row[col] if col in row_keys else None for col in legacy_session_cols]
            c.execute(
                f"INSERT INTO campaign_state (character_id, {', '.join(legacy_session_cols)}) "
                f"VALUES (?, {', '.join('?' * len(legacy_session_cols))})",
                (row["id"], *values),
            )
        conn.commit()

    # RAG: bảng entity (NPC/quái sinh động) + world_loot (ledger loot)
    entities.init_entity_tables(conn)

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


def create_campaign_state(char_id: int):
    """Tạo row campaign_state RỖNG cho nhân vật MỚI — gọi ngay sau khi INSERT
    character, nếu không thì mọi UPDATE campaign_state SET ... WHERE
    character_id=? sau đó sẽ không match dòng nào."""
    conn = get_conn()
    conn.execute("INSERT OR IGNORE INTO campaign_state (character_id) VALUES (?)", (char_id,))
    conn.commit()
    conn.close()


def get_campaign_state(char_id: int) -> sqlite3.Row | None:
    conn = get_conn()
    row = conn.execute("SELECT * FROM campaign_state WHERE character_id = ?", (char_id,)).fetchone()
    conn.close()
    return row


def update_campaign_state(char_id: int, **fields):
    """Update nhiều cột campaign_state cùng lúc bằng keyword args — thay cho
    việc viết tay từng câu UPDATE ... SET x=? WHERE character_id=? rải rác
    khắp dungeon_master.py."""
    if not fields:
        return
    conn = get_conn()
    set_clause = ", ".join(f"{k} = ?" for k in fields)
    conn.execute(
        f"UPDATE campaign_state SET {set_clause} WHERE character_id = ?",
        (*fields.values(), char_id),
    )
    conn.commit()
    conn.close()


def _load_campaign_bible() -> dict | None:
    """Campaign Bible đọc từ file JSON riêng (xem agents/campaign.py:
    save/load_campaign_bible) — dễ mở lên xem/theo dõi hơn hẳn so với 1 cột
    TEXT lớn trong SQLite."""
    return campaign.load_campaign_bible()


def save_current_milestone(char_id: int, milestone: dict):
    """Milestone hiện tại (khác Campaign Bible — đổi liên tục mỗi khi hoàn
    thành 1 cái, không tĩnh như bible) — lưu vào campaign_state.current_milestone."""
    update_campaign_state(char_id, current_milestone=json.dumps(milestone, ensure_ascii=False))


def load_current_milestone(state: sqlite3.Row | None) -> dict | None:
    raw = state["current_milestone"] if state and "current_milestone" in state.keys() else None
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
