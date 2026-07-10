"""
assistant.py — Trợ lý ngoài-truyện (out-of-character) cho người chơi.

Side-channel hoàn toàn tách biệt khỏi pipeline dungeon_master.py: chỉ ĐỌC
state (character, entity, campaign bible, history) để trả lời câu hỏi kiểu
"NPC này là ai", "tôi đang ở mốc truyện nào" — KHÔNG bao giờ ghi history,
không đổi character/entity, không đụng turn_number/mechanics. Vì vậy nó được
phép tiết lộ cả secret/twist CHƯA mở khoá nếu người chơi hỏi thẳng (khác DM
chính vốn phải giấu để giữ bất ngờ) — đây là công cụ hỗ trợ người chơi, không
phải một nhân vật trong truyện.
"""

import sqlite3

import ollama

import db
from config import MODEL, OPTIONS
from . import entities, campaign


def build_assistant_prompt(char: sqlite3.Row, active_entities: list, bible: dict | None,
                            milestone_index: int, history_summary: str,
                            recent_history_text: str) -> str:
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
        npc_lines = "\n".join(
            f"- {n['name']} ({n['key']}, {n['role']}): {n['desc']} | động cơ: {n['motivation']} "
            f"| bí mật: {n.get('secret') or '-'}"
            for n in nb["npcs"]
        ) or "Không có"
        monster_lines = ", ".join(m["name"] for m in nb["monsters"] if m.get("name")) or "Không có"
        milestone_lines = "\n".join(
            ("[HIỆN TẠI] " if i == milestone_index else ("[ĐÃ QUA] " if i < milestone_index else "[CHƯA TỚI] "))
            + f"{m['title']} — {m['description']}"
            for i, m in enumerate(nb["story_milestones"])
        )
        faction_lines = "\n".join(
            f"- {f['name']}: mục tiêu {f['goal']} | quan hệ với người chơi: {f['relationship_to_player']}"
            for f in nb["factions"]
        ) or "Không có"
        secret_lines = "\n".join(f"- [{s['id']}] {s['description']}" for s in nb["secrets"]) or "Không có"
        bible_block = (
            f"NPCs:\n{npc_lines}\n\n"
            f"Quái vật trong campaign: {monster_lines}\n\n"
            f"Mốc truyện (story milestones, theo thứ tự):\n{milestone_lines}\n\n"
            f"Phe phái:\n{faction_lines}\n\n"
            f"Bí mật/twist của campaign (kể cả CHƯA lộ ra trong truyện):\n{secret_lines}"
        )

    summary_block = history_summary or "(chưa có tóm tắt — truyện còn ngắn)"

    return f"""Bạn là TRỢ LÝ NGOÀI-TRUYỆN (out-of-character) cho một game nhập vai D&D dạng chữ. Người chơi đang tạm dừng hành động để hỏi bạn về bối cảnh/nhân vật/mốc truyện — đây KHÔNG phải một hành động trong truyện.

QUY TẮC:
- Trả lời bằng tiếng Việt, ngắn gọn, đúng trọng tâm câu hỏi.
- Bạn KHÔNG phải Người Kể Chuyện — không kể tiếp truyện, không tạo lựa chọn, không roll xúc xắc.
- Bạn ĐƯỢC PHÉP tiết lộ thông tin trong CAMPAIGN BIBLE bên dưới, kể cả bí mật/twist chưa lộ ra trong truyện, NẾU người chơi hỏi thẳng.
- Nếu câu hỏi không liên quan tới ván chơi này, trả lời ngắn rằng bạn chỉ hỗ trợ các câu hỏi về ván chơi hiện tại.
- Không tự bịa thông tin ngoài dữ liệu bên dưới — nếu không chắc/không có, nói rõ là không có thông tin đó.

## NHÂN VẬT NGƯỜI CHƠI
Tên: {c['name']} | {c['race']} {c['character_class']} | Cấp {c['level']} | HP {c['hp']}/{c['max_hp']} | Mana {c['mana']}/{c['max_mana']}

## NPC/QUÁI ĐANG CÓ MẶT TRONG CẢNH HIỆN TẠI
{entity_lines}

## TÓM TẮT DIỄN BIẾN TỪ TRƯỚC
{summary_block}

## DIỄN BIẾN GẦN ĐÂY (nguyên văn)
{recent_history_text}

## CAMPAIGN BIBLE (toàn bộ, kể cả phần chưa lộ trong truyện)
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
    conn.close()

    recent_history_text = "\n".join(
        f"{'Người chơi' if row['role'] == 'user' else 'Người kể chuyện'}: {row['content']}"
        for row in recent_rows
    ) or "(chưa có diễn biến nào)"

    history_summary = char["history_summary"] if "history_summary" in char.keys() else ""
    bible = db._load_campaign_bible(char)
    milestone_index = char["campaign_milestone_index"] if "campaign_milestone_index" in char.keys() else 0

    system_prompt = build_assistant_prompt(
        char, active_entities, bible, milestone_index, history_summary, recent_history_text,
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
