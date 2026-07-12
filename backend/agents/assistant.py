"""
assistant.py — Trợ lý ngoài-truyện (out-of-character) cho người chơi.

Side-channel hoàn toàn tách biệt khỏi pipeline dungeon_master.py: chỉ ĐỌC
state (character, entity, campaign bible, history) để trả lời câu hỏi kiểu
"NPC này là ai", "tôi đang ở mốc truyện nào" — KHÔNG bao giờ ghi history,
không đổi character/entity, không đụng turn_number/mechanics.

QUAN TRỌNG: dù đây là công cụ ngoài-truyện, nó KHÔNG ĐƯỢC phép spoil — chỉ
được tiết lộ những gì bản thân câu chuyện ĐÃ hé lộ tới thời điểm hiện tại
(dựa theo milestone_index và những gì đã thực sự xuất hiện trong lịch sử hội
thoại), y hệt nguyên tắc revelation-by-tier mà dungeon_master.py áp dụng cho
DM chính. NPC/bí mật/milestone CHƯA xuất hiện trong truyện thì bị lọc khỏi
prompt hoàn toàn trước khi gửi cho model — không chỉ dặn suông "đừng spoil"
(dặn suông không đủ tin cậy với model nhỏ)."""

import sqlite3

import ollama

import db
from config import MODEL, OPTIONS
from . import entities, campaign


def _npc_is_known(name: str, full_story_text: str) -> bool:
    """NPC được coi là 'đã biết' nếu tên của họ từng xuất hiện trong lời kể
    của DM (role='assistant') — cách duy nhất đáng tin cậy để biết người chơi
    đã thực sự gặp/nghe tới NPC này chưa, vì không phải NPC nào cũng được
    đăng ký vào bảng entity (chỉ NPC có tương tác cơ chế mới có, NPC thuần
    thoại như nhân chứng/đầu mối thường không)."""
    name = (name or "").strip()
    return bool(name) and name.lower() in full_story_text.lower()


def build_assistant_prompt(char: sqlite3.Row, active_entities: list, bible: dict | None,
                            act_index: int, history_summary: str,
                            recent_history_text: str, full_story_text: str) -> str:
    c = db.character_row_to_dict(char)

    entity_lines = "\n".join(
        f"- {e['name']} ({'quái' if e['entity_type'] == 'monster' else 'NPC'}"
        f"{', thù địch' if e['hostile'] else ''}): HP {e['hp']}/{e['max_hp']}"
        + (f" — {e['note']}" if e.get("note") else "")
        for e in active_entities
    ) or "(không có NPC/quái nào đang xuất hiện)"

    bible_block = "(chưa có campaign bible)"
    if bible:
        nb = campaign._normalize_bible(bible)
        acts = nb["acts"]
        idx = max(0, min(act_index, len(acts) - 1))
        current_act = acts[idx]

        # NPC: chỉ liệt kê những người ĐÃ xuất hiện trong truyện — không lộ
        # trước danh tính/vai trò của NPC người chơi chưa từng gặp. Trường
        # "secrets"/động cơ ngầm của antagonist KHÔNG BAO GIỜ được đưa vào
        # đây dù NPC đã biết, vì đó là twist dành riêng cho DM.
        known_npcs = [n for n in nb["characters"]["key_npcs"] if _npc_is_known(n["name"], full_story_text)]
        npc_lines = "\n".join(
            f"- {n['name']} ({n['role']}): {n['appearance']} | động cơ: {n['motivation']}"
            for n in known_npcs
        ) or "Người chơi chưa gặp NPC nào có tên riêng trong campaign bible tính tới lúc này."

        antagonist = nb["characters"]["main_antagonist"]
        antagonist_line = (
            f"{antagonist['name']}: {antagonist['desc']}"
            if _npc_is_known(antagonist["name"], full_story_text)
            else "Người chơi chưa từng nghe tới danh tính của thế lực đứng sau mọi chuyện."
        )

        # Quái: chỉ liệt kê loài đã thực sự chạm trán (có trong bảng entity),
        # tránh hé trước tên/loài quái sắp gặp.
        encountered_monster_names = {
            e["name"] for e in active_entities if e.get("entity_type") == "monster"
        }
        # active_entities chỉ gồm entity CÒN SỐNG — bổ sung cả quái đã từng
        # xuất hiện (kể cả đã chết/bỏ chạy) để không bị "quên" sau khi qua đời.
        conn_hist = db.get_conn()
        past_monsters = conn_hist.execute(
            "SELECT DISTINCT name FROM entity WHERE character_id = ? AND entity_type = 'monster'",
            (char["id"],),
        ).fetchall()
        conn_hist.close()
        encountered_monster_names |= {row["name"] for row in past_monsters}
        monster_lines = ", ".join(sorted(encountered_monster_names)) or "Người chơi chưa chạm trán quái nào trong campaign này."

        # Act: chỉ hiện act HIỆN TẠI — không hé purpose/exit_condition của các
        # act SAU (đó chính là spoil cốt truyện sắp tới).
        act_line = f"Act {current_act['act']}/3 (hiện tại): {current_act['purpose']} | kết thúc khi: {current_act['exit_condition']}"

        # Milestone hiện tại (disposable, sinh dần — xem agents/milestone.py):
        # chỉ hiện objective/story_purpose, KHÔNG hiện required_reveal/
        # forbidden_reveal/success_condition/failure_condition — những field
        # đó dành cho DM tự vận hành, không phải thứ kể thẳng cho người chơi.
        current_milestone = db.load_current_milestone(db.get_campaign_state(char["id"]))
        milestone_line = (
            f"{current_milestone['title']} — mục tiêu: {current_milestone['objective']}"
            if current_milestone else "(đang chuẩn bị mốc truyện tiếp theo...)"
        )

        faction_lines = "\n".join(
            f"- {f['name']}: mục tiêu {f['goal']} | quan hệ với người chơi: {f['relationship_to_player']}"
            for f in nb["world"]["major_factions"]
        ) or "Không có"

        bible_block = (
            f"NPC người chơi đã gặp (KHÔNG được bịa thêm NPC ngoài danh sách này):\n{npc_lines}\n\n"
            f"Thế lực đứng sau mọi chuyện (chỉ nếu đã lộ danh tính trong truyện): {antagonist_line}\n\n"
            f"Quái vật đã chạm trán: {monster_lines}\n\n"
            f"{act_line}\n\n"
            f"Mốc truyện hiện tại: {milestone_line}\n\n"
            f"Phe phái đã biết:\n{faction_lines}"
        )

    summary_block = history_summary or "(chưa có tóm tắt — truyện còn ngắn)"

    return f"""Bạn là TRỢ LÝ NGOÀI-TRUYỆN (out-of-character) cho một game nhập vai D&D dạng chữ. Người chơi đang tạm dừng hành động để hỏi bạn về bối cảnh/nhân vật/mốc truyện — đây KHÔNG phải một hành động trong truyện.

QUY TẮC (RẤT QUAN TRỌNG):
- Trả lời bằng tiếng Việt, ngắn gọn, đúng trọng tâm câu hỏi.
- Bạn KHÔNG phải Người Kể Chuyện — không kể tiếp truyện, không tạo lựa chọn, không roll xúc xắc.
- CẤM SPOIL TUYỆT ĐỐI: dữ liệu bên dưới ĐÃ được lọc để chỉ còn những gì câu chuyện thực sự hé lộ tới lúc này — bạn CHỈ được trả lời dựa trên đúng phần dữ liệu này. Nếu người chơi hỏi về một NPC/bí mật/diễn biến KHÔNG có trong dữ liệu bên dưới (vì chưa xảy ra trong truyện), trả lời rằng bạn chưa có thông tin đó / nhân vật đó chưa xuất hiện trong hành trình của họ — TUYỆT ĐỐI không tiết lộ nó dù bạn có thể suy luận ra từ câu hỏi.
- Nếu câu hỏi không liên quan tới ván chơi này, trả lời ngắn rằng bạn chỉ hỗ trợ các câu hỏi về ván chơi hiện tại.
- Không tự bịa thông tin ngoài dữ liệu bên dưới.

## NHÂN VẬT NGƯỜI CHƠI
Tên: {c['name']} | {c['race']} {c['character_class']} | Cấp {c['level']} | HP {c['hp']}/{c['max_hp']} | Mana {c['mana']}/{c['max_mana']}

## NPC/QUÁI ĐANG CÓ MẶT TRONG CẢNH HIỆN TẠI
{entity_lines}

## TÓM TẮT DIỄN BIẾN TỪ TRƯỚC
{summary_block}

## DIỄN BIẾN GẦN ĐÂY (nguyên văn)
{recent_history_text}

## CAMPAIGN BIBLE (đã lọc — chỉ gồm phần đã lộ ra trong truyện tính tới hiện tại)
{bible_block}
"""


async def handle_ask(data: dict) -> dict:
    question = (data.get("question") or "").strip()
    if not question:
        return {"error": "Thiếu câu hỏi."}

    char = db.get_latest_character()
    if not char:
        return {"error": "Chưa có nhân vật."}

    conn = db.get_conn()
    c = conn.cursor()
    active_entities = [dict(row) for row in entities.get_active_entities(conn, char["id"])]
    recent_rows = list(reversed(c.execute(
        "SELECT role, content FROM history ORDER BY id DESC LIMIT 20"
    ).fetchall()))
    # Toàn bộ lời kể của DM từ đầu game — dùng để xác định NPC nào đã thực sự
    # xuất hiện trong truyện (xem _npc_is_known), không chỉ 20 dòng gần nhất.
    story_rows = c.execute(
        "SELECT content FROM history WHERE role = 'assistant' ORDER BY id ASC"
    ).fetchall()
    conn.close()

    recent_history_text = "\n".join(
        f"{'Người chơi' if row['role'] == 'user' else 'Người kể chuyện'}: {row['content']}"
        for row in recent_rows
    ) or "(chưa có diễn biến nào)"
    full_story_text = "\n".join(row["content"] or "" for row in story_rows)

    state = db.get_campaign_state(char["id"])
    history_summary = (state["history_summary"] if state else "") or ""
    bible = db._load_campaign_bible()
    act_index = (state["campaign_act_index"] if state else 0) or 0

    system_prompt = build_assistant_prompt(
        char, active_entities, bible, act_index, history_summary, recent_history_text,
        full_story_text,
    )

    resp = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        options=OPTIONS,
        think=False,
    )
    return {"answer": resp["message"]["content"].strip()}
