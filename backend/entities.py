"""
entities.py — Theo dõi NPC/quái vật được sinh ra trong lúc chơi + ledger loot,
lưu DB để nhất quán qua các lượt (không để model tự bịa lại HP/loot mỗi turn).

Nguyên tắc giống hệt cách file chính đang xử lý HP nhân vật:
- Model KHÔNG được tự quyết absolute HP của entity đã tồn tại — chỉ báo
  hp_change (delta). Backend tự cộng dồn, tự clamp, tự đánh dấu chết.
- Entity MỚI (lần đầu xuất hiện) thì model báo max_hp/hp khởi tạo — backend
  chỉ sanity-check (không cho HP âm/quá lớn) rồi insert.
- Loot rơi ra được ghi vào ledger `world_loot`. Khi model báo items_added,
  backend cố khớp với ledger để xác nhận loot đó thực sự "có nguồn gốc"
  (đã được công bố rơi ra ở lượt nào đó) trước khi cộng vào inventory —
  tránh việc model tự bịa loot không có căn cứ.
"""

import sqlite3


def init_entity_tables(conn: sqlite3.Connection):
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS entity(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            character_id INTEGER,
            key TEXT,
            name TEXT,
            entity_type TEXT,      -- 'npc' | 'monster'
            hp INTEGER,
            max_hp INTEGER,
            ac INTEGER DEFAULT 12,
            attack_bonus INTEGER DEFAULT 3,
            damage_dice TEXT DEFAULT '1d6',
            hostile INTEGER DEFAULT 0,
            status TEXT DEFAULT 'alive',  -- alive | dead | fled | gone
            note TEXT,
            first_seen_turn INTEGER,
            last_seen_turn INTEGER
        )
    """)
    existing_entity_cols = {row["name"] for row in c.execute("PRAGMA table_info(entity)")}
    if "ac" not in existing_entity_cols:
        c.execute("ALTER TABLE entity ADD COLUMN ac INTEGER DEFAULT 12")
    if "attack_bonus" not in existing_entity_cols:
        c.execute("ALTER TABLE entity ADD COLUMN attack_bonus INTEGER DEFAULT 3")
    if "damage_dice" not in existing_entity_cols:
        c.execute("ALTER TABLE entity ADD COLUMN damage_dice TEXT DEFAULT '1d6'")

    c.execute("""
        CREATE TABLE IF NOT EXISTS world_loot(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            character_id INTEGER,
            name TEXT,
            source_key TEXT,       -- key của entity đã rơi ra item này, có thể NULL
            status TEXT DEFAULT 'available',  -- available | picked_up
            created_turn INTEGER
        )
    """)
    conn.commit()


def reset_session(conn: sqlite3.Connection):
    """Gọi khi tạo nhân vật mới (xóa save-slot cũ) để dọn sạch entity/loot cũ.
    Hệ thống chỉ có 1 save-slot duy nhất (character bị DELETE toàn bộ mỗi lần
    tạo mới) nên ở đây cũng xóa toàn bộ, không cần lọc theo character_id."""
    c = conn.cursor()
    c.execute("DELETE FROM entity")
    c.execute("DELETE FROM world_loot")
    conn.commit()


def _safe_int(v, default=0):
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


_DICE_NOTATION_RE = None


def _sanitize_dice(notation) -> str:
    """Validate '<n>d<sides>[+/-k]' — nếu model bịa chuỗi không hợp lệ,
    fallback về mặc định thay vì lưu rác vào DB (sẽ khiến roll_dice() ở
    classification.py trả về 0 khi dùng lại sau này)."""
    global _DICE_NOTATION_RE
    if _DICE_NOTATION_RE is None:
        import re
        _DICE_NOTATION_RE = re.compile(r"^\d+d\d+([+-]\d+)?$")
    notation = (notation or "").strip()
    if notation and _DICE_NOTATION_RE.match(notation):
        return notation
    return "1d6"


def _fuzzy_match(a: str, b: str) -> bool:
    a = (a or "").strip().lower()
    b = (b or "").strip().lower()
    if not a or not b:
        return False
    if a == b:
        return True
    return len(a) >= 4 and len(b) >= 4 and (a in b or b in a)


# ---------------------------------------------------------------------------
# Đọc trạng thái hiện tại -> format context cho prompt
# ---------------------------------------------------------------------------

def get_active_entities(conn: sqlite3.Connection, character_id: int):
    c = conn.cursor()
    return c.execute(
        "SELECT * FROM entity WHERE character_id = ? AND status = 'alive' "
        "ORDER BY last_seen_turn DESC",
        (character_id,),
    ).fetchall()


def get_available_loot(conn: sqlite3.Connection, character_id: int):
    c = conn.cursor()
    return c.execute(
        "SELECT * FROM world_loot WHERE character_id = ? AND status = 'available' "
        "ORDER BY id DESC",
        (character_id,),
    ).fetchall()


def format_entities_context(conn: sqlite3.Connection, character_id: int) -> str:
    """Block text nhét vào messages mỗi turn. Nếu rỗng cả hai, trả về chuỗi
    rỗng (không cần thêm section thừa vào prompt)."""
    entities = get_active_entities(conn, character_id)
    loot = get_available_loot(conn, character_id)

    if not entities and not loot:
        return ""

    lines = ["## ACTIVE ENTITIES & LOOT (persisted state — authoritative, do not override)"]

    if entities:
        lines.append("Currently alive NPCs/monsters (reference by key, report hp_change only, "
                    "NEVER restate their max_hp/type):")
        for e in entities:
            tag = "MONSTER" if e["entity_type"] == "monster" else "NPC"
            hostility = " [hostile]" if e["hostile"] else ""
            status = " [status]" if e["status"] else ""
            lines.append(f"- key=\"{e['key']}\" [{tag}] {e['name']}: HP {e['hp']}/{e['max_hp']} AC {e['ac']}{hostility} STATUS: {status}")
    else:
        lines.append("No NPCs/monsters currently active in the scene.")

    if loot:
        lines.append("Loot already established as existing in the world but not yet picked up "
                    "(only these may be picked up via items_added unless you declare a NEW drop "
                    "this turn via mechanics.loot_dropped):")
        for it in loot:
            lines.append(f"- {it['name']}")

    lines.append(
        "\nRULES: To introduce a NEW npc/monster, invent a short lowercase snake_case key "
        "(Exemple: \"goblin_2\") not already listed above, and report it in mechanics.entities with "
        "type/max_hp/hp/ac(8-20, tougher/armored foes get higher AC)/hostile. To damage/heal/kill an "
        "EXISTING one, report ONLY its key and "
        "hp_change (negative=damage, positive=heal) — never invent a new key for something "
        "already listed. Set status=\"dead\" when it dies, \"fled\" if it escapes. To drop new "
        "loot, add it to mechanics.loot_dropped with a name and the source key (or null if not "
        "from a specific creature). IMPORTANT: any NEW monster's power level must respect the "
        "THREAT SCALING section given elsewhere in context (region, location type, character "
        "level) — do not introduce a threat above the allowed tier."
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Áp dụng thay đổi từ output của model
# ---------------------------------------------------------------------------

def apply_entity_changes(conn: sqlite3.Connection, character_id: int, entities_payload, turn_number: int):
    """entities_payload: list of dicts, mỗi dict có thể là:
    - Entity MỚI: {"key", "name", "type", "max_hp", "hp", "hostile"}
    - Entity ĐÃ CÓ: {"key", "hp_change", "status"(optional)}
    Tự phân biệt bằng cách tra key đã tồn tại trong DB chưa."""
    if not entities_payload:
        return

    c = conn.cursor()
    for item in entities_payload:
        if not isinstance(item, dict):
            continue
        key = (item.get("key") or "").strip().lower().replace(" ", "_")
        if not key:
            continue

        existing = c.execute(
            "SELECT * FROM entity WHERE character_id = ? AND key = ?",
            (character_id, key),
        ).fetchone()

        if existing:
            hp_change = _safe_int(item.get("hp_change", 0), 0)
            new_hp = max(0, min(existing["max_hp"], existing["hp"] + hp_change))
            status = item.get("status") or existing["status"]
            if new_hp <= 0:
                status = "dead"
            c.execute(
                "UPDATE entity SET hp = ?, status = ?, last_seen_turn = ? WHERE id = ?",
                (new_hp, status, turn_number, existing["id"]),
            )
        else:
            # Entity mới — sanity check số liệu, không tin tuyệt đối vào model
            max_hp = _safe_int(item.get("max_hp", item.get("hp", 10)), 10)
            max_hp = max(1, min(max_hp, 500))  # chặn số vô lý
            hp = _safe_int(item.get("hp", max_hp), max_hp)
            hp = max(0, min(hp, max_hp))
            entity_type = item.get("type") if item.get("type") in ("npc", "monster") else "npc"
            hostile = 1 if item.get("hostile") else 0
            name = (item.get("name") or key).strip()
            ac = _safe_int(item.get("ac", 12), 12)
            ac = max(8, min(ac, 20))  # chặn số vô lý, giữ trong khung 5e hợp lý
            attack_bonus = _safe_int(item.get("attack_bonus", 3), 3)
            attack_bonus = max(0, min(attack_bonus, 10))
            damage_dice = _sanitize_dice(item.get("damage_dice"))
            c.execute(
                """INSERT INTO entity
                (character_id, key, name, entity_type, hp, max_hp, ac, attack_bonus, damage_dice,
                hostile, status, first_seen_turn, last_seen_turn)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'alive', ?, ?)""",
                (character_id, key, name, entity_type, hp, max_hp, ac, attack_bonus, damage_dice,
                 hostile, turn_number, turn_number),
            )
    conn.commit()


def register_loot_drops(conn: sqlite3.Connection, character_id: int, loot_payload, turn_number: int):
    """loot_payload: list of {"name": ..., "source_key": ... (optional)}"""
    if not loot_payload:
        return
    c = conn.cursor()
    for item in loot_payload:
        if isinstance(item, dict):
            name = (item.get("name") or "").strip()
            source_key = (item.get("source_key") or "").strip().lower() or None
        else:
            name = str(item).strip()
            source_key = None
        if not name:
            continue
        c.execute(
            "INSERT INTO world_loot (character_id, name, source_key, status, created_turn) "
            "VALUES (?, ?, ?, 'available', ?)",
            (character_id, name, source_key, turn_number),
        )
    conn.commit()


def validate_items_added(conn: sqlite3.Connection, character_id: int, items_added):
    """Đối chiếu items_added (mà DM model báo cáo) với ledger world_loot.
    - Khớp được -> đánh dấu picked_up, giữ item trong danh sách trả về.
    - Không khớp -> VẪN giữ lại (soft validation, không chặn cứng để không vỡ
    các luồng thưởng quest/kho báu không đi qua loot_dropped), nhưng trả kèm
    danh sách "unverified" để log debug, giúp bạn phát hiện model đang bịa
    loot không qua cơ chế công bố trước."""
    if not items_added:
        return items_added, []

    c = conn.cursor()
    available = c.execute(
        "SELECT * FROM world_loot WHERE character_id = ? AND status = 'available'",
        (character_id,),
    ).fetchall()

    unverified = []
    for name in items_added:
        match = next((row for row in available if _fuzzy_match(row["name"], name)), None)
        if match:
            c.execute("UPDATE world_loot SET status = 'picked_up' WHERE id = ?", (match["id"],))
        else:
            unverified.append(name)
    conn.commit()
    return items_added, unverified