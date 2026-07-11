"""
dungeon_master.py — Dungeon Master: system prompt + pipeline xử lý 1 lượt
/chat, /chat/retry, /start_game. Đây là phần "trong truyện" — khác với
agents/assistant.py (ngoài truyện).
"""

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor

import ollama

import db
from config import MODEL, OPTIONS
from . import classification, context_writer, entities, imagegen, milestone, social, summarizer, campaign, lore, text_utils, world_state

ATTR_LABELS = {
    "str": "STR/Sức mạnh", "dex": "DEX/Nhanh nhẹn", "con": "CON/Thể chất",
    "int": "INT/Trí tuệ", "wis": "WIS/Khôn ngoan", "cha": "CHA/Sức hút",
}

# ollama.chat() ở milestone.generate_milestone() (và, khi gọi từ orchestrator
# tạo campaign mới trong main.py, cả handle_start_game() bên dưới) là lời gọi
# ĐỒNG BỘ (blocking) — chạy qua executor riêng khi cần bất đồng bộ thật, nếu
# không sẽ chặn event loop chính (giống lý do imagegen.py dùng executor riêng).
_BG_LLM_EXECUTOR = ThreadPoolExecutor(max_workers=1)


async def _generate_next_milestone_async(char_id, bible, story_state, act_index, milestone_number, target_total):
    loop = asyncio.get_running_loop()
    try:
        ms = await loop.run_in_executor(
            _BG_LLM_EXECUTOR, milestone.generate_milestone,
            bible, story_state, act_index, milestone_number, target_total,
        )
        db.save_current_milestone(char_id, ms)
        print(f"[DEBUG] milestone kế tiếp đã sinh xong: {ms.get('title')}")
    except Exception as e:
        print(f"[DEBUG] lỗi generate milestone kế tiếp: {e}")

# Milestone bị "kẹt" (model không tự báo milestone_complete dù nội dung đã đủ
# điều kiện) từng khiến 1 campaign đứng yên ở milestone đầu tiên suốt 50+ lượt
# -> câu chuyện lặp lại nhàm chán và chỉ còn biết spam quái làm "sự kiện" lấp
# đầy turn_note ép buộc bên dưới. Hai ngưỡng này là van an toàn: SOFT chỉ nhắc
# mạnh hơn, HARD ép backend tự chuyển milestone kể cả model không tự báo.
MILESTONE_SOFT_STUCK_TURNS = 12
MILESTONE_HARD_STUCK_TURNS = 22

# Thưởng XP/gold khi giết quái — code tự tính theo tier (suy từ max_hp), không
# để model tự quyết (trước đây model lúc cộng xp lúc quên, không nhất quán).
# Ngưỡng max_hp -> (xp, gold dice), tier tăng dần.
MONSTER_TIER_REWARD = [
    (10, 10, "2d6"),
    (25, 25, "3d8"),
    (50, 50, "4d10"),
    (100, 100, "6d10"),
    (10**9, 250, "8d12"),
]


def _monster_kill_reward(max_hp) -> dict:
    max_hp = db.safe_int(max_hp, 10)
    for threshold, xp, gold_dice in MONSTER_TIER_REWARD:
        if max_hp <= threshold:
            gold = classification.roll_dice(gold_dice)["total"]
            return {"xp": xp, "gold": gold}
    return {"xp": 250, "gold": 0}


# Số lượt story gần nhất xét tới khi tìm bế tắc hội thoại, và ngưỡng số lượt
# (trong đó) mà 1 NPC bị nhắc tới liên tục mới coi là "quanh quẩn không lối
# thoát". difflib so toàn đoạn văn KHÔNG đủ nhạy cho kiểu bế tắc này — model
# đổi câu chữ mỗi lượt (khác hẳn nhau về mặt ký tự) trong khi vẫn giữ nguyên
# KẾT CỤC (NPC từ chối/lảng tránh, không ai tiến triển gì) — nên dấu hiệu
# đáng tin hơn nhiều là: 1 cái tên NPC cụ thể cứ xuất hiện liên tục trong
# story text nhiều lượt liền, bất kể câu chữ quanh nó thế nào.
_DEADLOCK_LOOKBACK_TURNS = 4
_DEADLOCK_MIN_MENTIONS = 3


def _detect_dialogue_deadlock(bible: dict | None) -> str | None:
    """Trả về tên NPC đang gây bế tắc nếu phát hiện (NPC đó bị nhắc tới trong
    hầu hết story text của vài lượt gần nhất), ngược lại None. Dùng để ép
    backend buộc model tạo bước ngoặt thật sự thay vì nhắc suông "đừng lặp
    lại" (vốn không đủ mạnh với model 14B khi nó chỉ đổi câu chữ chứ không
    đổi tình huống)."""
    if not bible:
        return None
    npc_names = [n.get("name") for n in (bible.get("npcs") or []) if n.get("name")]
    if not npc_names:
        return None

    conn = db.get_conn()
    rows = conn.execute(
        "SELECT content FROM history WHERE role = 'assistant' ORDER BY id DESC LIMIT ?",
        (_DEADLOCK_LOOKBACK_TURNS,),
    ).fetchall()
    conn.close()
    if len(rows) < _DEADLOCK_LOOKBACK_TURNS:
        return None
    texts_lower = [(r["content"] or "").lower() for r in rows]

    for name in npc_names:
        name_lower = name.strip().lower()
        if not name_lower:
            continue
        mentions = sum(1 for t in texts_lower if name_lower in t)
        if mentions >= _DEADLOCK_MIN_MENTIONS:
            return name
    return None


# ---------------------------------------------------------------------------
# Retry lượt gần nhất — undo 1 cấp (xem nút "🔄 Thử lại" trên message bubble)
# ---------------------------------------------------------------------------

def _snapshot_pre_turn(conn, char_id: int, user_input: str):
    """Chụp lại TOÀN BỘ state có thể bị đổi trong 1 lượt /chat (character row,
    entity, world_loot, mốc history hiện tại) NGAY TRƯỚC KHI xử lý lượt đó —
    để nút "Thử lại" trên UI khôi phục đúng về đây rồi chạy lại CÙNG
    user_input, nếu model lỡ sinh lỗi (lộ tên NPC, quái sai roster, lẫn tiếng
    Trung...). Ghi đè snapshot cũ mỗi lượt -> chỉ thử lại được lượt GẦN NHẤT,
    không phải toàn bộ lịch sử (đủ dùng cho "lỡ tay" tức thời, không phải undo
    vô hạn)."""
    c = conn.cursor()
    char_row = c.execute("SELECT * FROM character WHERE id = ?", (char_id,)).fetchone()
    if not char_row:
        return
    char_dict = dict(char_row)
    char_dict.pop("pre_turn_snapshot", None)  # đừng lồng snapshot vào chính nó
    # last_turn_resolution PHẢI sống sót qua restore — đây chính là kết quả
    # xúc xắc THẬT mà người chơi đã bấm roll cho lượt sắp xử lý; nếu snapshot
    # nó ở đây rồi restore đè lại, "Thử lại" sẽ mất kết quả roll gốc và không
    # còn gì để tái dùng (phải roll lại — đúng thứ cần tránh).
    char_dict.pop("last_turn_resolution", None)

    entity_rows = [dict(r) for r in c.execute(
        "SELECT * FROM entity WHERE character_id = ?", (char_id,)
    ).fetchall()]
    loot_rows = [dict(r) for r in c.execute(
        "SELECT * FROM world_loot WHERE character_id = ?", (char_id,)
    ).fetchall()]
    max_history_id = c.execute("SELECT COALESCE(MAX(id), 0) AS m FROM history").fetchone()["m"]

    snapshot = {
        "user_input": user_input,
        "character": char_dict,
        "entities": entity_rows,
        "world_loot": loot_rows,
        "max_history_id": max_history_id,
    }
    c.execute(
        "UPDATE character SET pre_turn_snapshot = ? WHERE id = ?",
        (json.dumps(snapshot, ensure_ascii=False), char_id),
    )
    conn.commit()


def _restore_pre_turn(conn, char_id: int) -> str | None:
    """Khôi phục state từ pre_turn_snapshot (nếu có), trả về user_input đã lưu
    để gọi lại y hệt lượt vừa rồi. Trả None nếu chưa có lượt /chat nào để thử
    lại (vd vừa /start_game xong, chưa hành động lần nào)."""
    c = conn.cursor()
    row = c.execute("SELECT pre_turn_snapshot FROM character WHERE id = ?", (char_id,)).fetchone()
    if not row or not row["pre_turn_snapshot"]:
        return None
    try:
        snap = json.loads(row["pre_turn_snapshot"])
    except (TypeError, json.JSONDecodeError):
        return None

    char_fields = snap.get("character") or {}
    cols = [k for k in char_fields.keys() if k not in ("id", "pre_turn_snapshot", "last_turn_resolution")]
    if cols:
        set_clause = ", ".join(f"{col} = ?" for col in cols)
        values = [char_fields[col] for col in cols]
        c.execute(f"UPDATE character SET {set_clause} WHERE id = ?", (*values, char_id))

    c.execute("DELETE FROM entity WHERE character_id = ?", (char_id,))
    for e in snap.get("entities") or []:
        e = {k: v for k, v in e.items() if k != "id"}
        if not e:
            continue
        cols2 = list(e.keys())
        c.execute(
            f"INSERT INTO entity ({', '.join(cols2)}) VALUES ({', '.join('?' * len(cols2))})",
            [e[k] for k in cols2],
        )

    c.execute("DELETE FROM world_loot WHERE character_id = ?", (char_id,))
    for l in snap.get("world_loot") or []:
        l = {k: v for k, v in l.items() if k != "id"}
        if not l:
            continue
        cols3 = list(l.keys())
        c.execute(
            f"INSERT INTO world_loot ({', '.join(cols3)}) VALUES ({', '.join('?' * len(cols3))})",
            [l[k] for k in cols3],
        )

    c.execute("DELETE FROM history WHERE id > ?", (snap.get("max_history_id", 0),))
    conn.commit()
    return snap.get("user_input")


# ---------------------------------------------------------------------------
# DM system prompt
# ---------------------------------------------------------------------------

def build_system_prompt(char) -> str:
    c = db.character_row_to_dict(char)

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

    # Bible cố định (canon) + milestone HIỆN TẠI (sinh dần theo diễn biến thật
    # — xem agents/milestone.py, KHÔNG còn là 1 mảng cố định trong bible nữa)
    # -> chỉ mớm đúng milestone đang active, không phải cả campaign, vừa cắt
    # mạnh context mỗi lượt vừa chặn model tự spoil chuyện chưa tới.
    campaign_block = campaign.format_campaign_context(
        db._load_campaign_bible(char),
        current_milestone=db.load_current_milestone(char),
        act_index=char["campaign_act_index"] if "campaign_act_index" in char.keys() else 0,
        turn_number=char["current_turn"] or 0,
    )

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
1. Action uses a weapon/item/skill not in Equipment/Skills/Items above, OR a skill marked
   "[ON COOLDOWN]"? -> success=false, concrete penalty (HP loss/spotted/etc, never "soft"). E.g.
   "cast Fireball" with no Fireball skill -> success:false, changes.hp:-8, character fumbles.
2. Physically/logically impossible given attributes/context? -> same rule.
3. Otherwise resolve normally: advantage/disadvantage/normal from ONE attribute/strength/
   weakness (never invent other justification).

## ITEMS & SKILLS — backend-managed, do not duplicate
Never decide yourself that the character finds/loses/uses an item or skill outside what
INTERNAL MECHANICS tells you each turn. items_added/items_removed: only set when the turn
note explicitly says so this turn; otherwise []. Narrate richly, but bookkeeping is the
backend's job.

## CONSEQUENCES
Killing a monster: do NOT invent xp/gold — backend auto-awards on status="dead"; leave
changes.xp/gold at 0 for the kill. Non-combat rewards (quest turn-in, payment, treasure,
progress): your call, set changes yourself. Failure: HP loss / mana loss / concrete story
setback — always mechanical, never just narrative.
HP LOSS MUST COME FROM A REAL RESOLVED ATTACK, never invented: if success=true this turn,
changes.hp CANNOT be negative — the backend forces it to 0 regardless of what you write. A
brand-new hostile monster you introduce THIS turn cannot deal damage the same turn it appears
(it isn't in ACTIVE ENTITIES yet, so the backend has no real attack roll for it) — narrate its
entrance as a threat/ambush/lunge that hasn't connected yet ("lao tới nhưng chưa kịp trúng đòn"),
not as a landed hit. Its attack only actually lands (and deals real rolled damage) starting NEXT
turn once it exists in ACTIVE ENTITIES and the backend resolves its attack roll for you.

## DEATH RULE
Damage reduces HP to <=0 this turn -> mechanics.character_died=true, choices=[] (story ends).
Never set true unless HP truly hits 0 this turn.

## DEATH SCENE (when character_died=true)
150-220 words, no choices, unambiguous death. Tormentor's voice cuts through — mocking,
triumphant, merciless (use {c['name']}, never "bạn"). Full unflinching physical detail of the
death itself — no fade-to-black. End on a cold, damning line from the tormentor, not comfort.
No dice/DC/numeric mechanics even here.

## CHOICES
Exactly 4, no lettering/numbering. Each SPECIFIC to this scene's concrete details — never a
bare verb ("Attack"/"Investigate"/"Run"). E.g. "Lao vào ném dao nhằm mắt con quái trước khi nó
vồ tới", not "Tấn công". Vary the underlying approach (combat/stealth/diplomacy/investigation/
escape/trickery/sacrifice-a-resource) — pick whichever 4 fit the scene, don't force every
category every turn. Avoid repeating wording/tactics from recent turns. If the previous action
failed, >=1 choice must directly address that consequence. >=1 choice should carry risk/cost
beyond combat damage (item spent, worse outcome, moral compromise, info revealed). >=1 choice
should leverage something SPECIFIC to this character (race/class/strength/weakness from the
sheet, not generic) whenever the scene plausibly allows it — e.g. a Dwarf reading stonework, a
Rogue picking a lock quietly; reference the exact sheet entry, never invent a trait not on it.

CONSISTENCY WITH STORY TEXT: never name an NPC/creature in a choice unless that exact name (or
an unambiguous description of them, e.g. "lão già", "tên lính canh") already appeared in THIS
turn's story text or was established in an earlier turn. If the story only just introduced
someone as "một bóng người"/"một người lạ" without naming them yet, choices must refer to them
the same vague way too — never let a choice leak a name the player hasn't been told yet.

## SCENE CONTINUITY
[CURRENT SCENE STATE] each turn is the ONLY truth for where the character is — never move
backward to a resolved location/puzzle/obstacle unless they explicitly retreat; always advance
scene_state forward. Frame each new scene as a SITUATION (opportunity + danger + mystery +
rising tension), not a bare quest step — the player should want to act immediately.

NEVER REPEAT PROSE VERBATIM: if the player's action this turn closely mirrors/repeats one from
recent turns (e.g. asking the same question again, in different words), do NOT reproduce the
same story text again, even partially — the scene must move forward. React to the repetition
itself: the NPC grows impatient/suspicious of being asked twice, reveals a NEW detail this time,
deflects differently, or the situation escalates. Every turn's story must contain something the
player hasn't already read.

## NARRATION STYLE (story text only)
Jump straight into action/sensory detail/world reaction — never open with "{c['name']} + verb"
or call the player "bạn". When naming who acts, always {c['name']} (never a generic class/race
label) — prefer describing what happens over stating who acts. BAD: "Bạn cảm thấy lạnh...",
"Tên chiến binh rút kiếm...". GOOD: "Khói mùi lưu huỳnh bốc lên từ khe đá.", "Mũi kiếm vạt
ngang bụng con quái."

WRITE NATURAL VIETNAMESE, not a literal English-to-Vietnamese translation. A common failure
mode: chaining 3+ comma-separated descriptive clauses in one sentence like an English appositive
list — this reads as awkward, translated Vietnamese. BAD (translated-English rhythm): "Đó là
một sinh vật không mặt, mặc áo choàng rách nát, chuyển động với sự khinh thường." GOOD (natural
Vietnamese rhythm, split into short punchy sentences): "Nó không có mặt. Áo choàng rách bươm
phất phơ theo từng bước chân khinh khỉnh." Keep sentences short and concrete; let action verbs
carry the scene instead of stacking adjective clauses.

## ENTITIES (NPCs/monsters) — PERSISTENCE RULES
mechanics.entities is OPTIONAL — only when (a) a new NPC/monster appears, or (b) an EXISTING
one (in ACTIVE ENTITIES) takes damage/heals/dies/flees.
NEW: key (snake_case unique), name (English), type ("npc"|"monster"), max_hp, hp(=max_hp),
ac (8-20: weak=8-11, average=12-14, armored=15-18, legendary=19-20), attack_bonus (0-10:
weak=0-2, average=3-5, strong=6-8, boss=9-10), damage_dice (5e notation, e.g. "1d6"/"2d8+2"),
hostile (bool). These are set ONCE and drive all later attack rolls to/from it — match how the
story describes it (a lightly-clad goblin ≠ AC 18). You never compute hit/damage yourself —
backend rolls it, you only narrate the result.
gender ("male"|"female"): REQUIRED whenever the entity is a PERSON (human/humanoid) — even if
type="monster" because they're hostile (e.g. a cultist, a rival, a bandit are still people).
Skip only for genuinely non-human/genderless creatures (beasts, constructs, aberrations, undead
without a clear human identity). Set once at creation, becomes the fixed ground truth for
pronouns ("anh ta"/"cô ta" etc.) every future turn — never switch a named person's gender.
NEW hostile monster MUST come from the CAMPAIGN's MONSTER ROSTER if one fits (exact name/
appearance/moveset/behavior) — don't default to a generic creature out of habit.
CONSISTENCY: the entity you write MUST match what the story text this turn just described —
same species, same attack method (a snake bite ≠ "spider" in entities). Never contradict prose.
EXISTING (key already in ACTIVE ENTITIES): send ONLY key + hp_change (negative=damage,
positive=heal) + "status":"dead"/"fled" if applicable — never resend max_hp/type/name, never
switch its species mid-scene, never invent stats for it.
Monster names must be English, never Vietnamese.
Do NOT write "description" or "visual_prompt" for entities — a separate system resolves the
player-facing description and the image prompt from your story text + the campaign roster after
you respond, so writing them here only wastes your token budget.

## LOOT — PERSISTENCE RULES
loot_dropped: only when the story this turn explicitly shows an item becoming available (kill
drop, chest opened, found item) — name (English), source_key (entity key or null).
items_added must only reference items dropped THIS turn via loot_dropped, already in LOOT
AVAILABLE, or a direct narrative reward clearly justified by the story — never invent combat
loot without declaring it via loot_dropped first.

## LOCATION
mechanics.location: null on most turns; set ONLY the turn the character arrives at a NEW
distinct place (never repeat while staying put, never trigger from a passing mention). Just
name (English, short) — do NOT write description or visual_prompt, same reason as ENTITIES
above: a separate system resolves those from your story text afterward.

## LANGUAGE
100% Vietnamese output, no exceptions, even in unclear/conflicting cases — just decide silently
and continue in Vietnamese. NEVER mix in Chinese, English, or any other language/script — not
even a single stray character inside an otherwise-Vietnamese word. Monster and location names
stay English (only those, nothing else).

## OUTPUT FORMAT (Vietnamese only for story/choices, JSON keys in English)
Do NOT output "success"/"roll_type" — backend already decided those; writing them wastes budget.
{{
  "story": "...",
  "mechanics": {{
    "reasoning": "",
    "event_occurred": false,
    "character_died": false,
    "milestone_complete": false,
    "milestone_outcome_summary": "",
    "act_complete": false,
    "changes": {{"hp": 0, "mana": 0, "gold": 0, "xp": 0, "items_added": [], "items_removed": []}},
    "entities": [
      {{"key": "<unique_entity_key>", "name": "<entity_name>", "type": "monster|npc|companion|object",
        "gender": "male|female (omit/null if not a person)", "max_hp": 0, "hp": 0, "ac": 12,
        "hostile": false, "status": "alive|dead"}}
    ],
    "loot_dropped": [{{"name": "<item_name>", "source_key": "<entity_key>"}}],
    "location": null
  }},
  "choices": [
    {{"text": "...", "needs_roll": true, "roll": "advantage", "dc": 12, "reason": {{"type": "attribute", "name": "WIS"}}}}
    {{"text": "...", "needs_roll": true, "roll": "advantage", "dc": 10, "reason": {{"type": "race", "name": "Elf"}}}}
    {{"text": "...", "needs_roll": true, "roll": "disadvantage", "dc": 16, "reason": {{"type": "class", "name": "Fighter"}}}}
    {{"text": "...", "needs_roll": false, "roll": "normal", "dc": null, "reason": null}}
  ]
}}
reasoning: ONE short sentence explaining WHY this turn's roll succeeded/failed, tied to the
actual attribute/skill/item that governed it (dice_fact each turn tells you which one) — write
it like a brief in-world tactical aside, e.g. "Phản xạ nhanh giúp né đòn trong gang tấc." NEVER
mention system/internal terms here (MONSTER ROSTER, mechanics, JSON, backend, dice, DC, entity,
roll) — those words must never appear in this field. If this turn had NO real roll (no "dice" in
INTERNAL MECHANICS this turn — e.g. a pure narrative/no-risk action), leave reasoning as "" —
do not invent a reason for something that didn't happen.

roll: advantage/disadvantage/normal (if normal, reason=null; else reason.type is one of
attribute/strength/weakness/skill/item/race/class, citing exactly one real sheet value).
needs_roll: true if there's any real chance of failure worth rolling; false for pure flavor/
no-risk (e.g. "quan sát xung quanh") that always succeeds.

dc: REQUIRED when needs_roll=true (null otherwise). Set it now, with full scene context, using
the FULL 8-20 range — don't cluster in the middle: 8-9 trivial (no real opposition) | 10-12
easy (minor risk, favorable) | 13-15 moderate (real coin-flip risk, alert opposition) | 16-18
hard (skilled/prepared/hostile opposition) | 19-20 near-impossible (only a desperate gambit
succeeds). Combat vs an alert competent hostile defaults 14-16. roll="disadvantage" -> dc>=14.
roll="advantage" -> can sit lower (8-12), character has a real edge.
MANDATORY SPREAD across the 4 choices: don't let all 4 land in one narrow band (e.g. never all
four at 11-14) — safest choice usually low (8-11), boldest/highest-stakes usually high (16-20).

These needs_roll/roll/dc/reason values are FINAL — if the player picks this exact choice next
turn, the backend reuses them as-is (no re-classification), so they must be your true judgement.

milestone_complete: set true ONLY on the turn where the story you just wrote clearly satisfies
the CURRENT MILESTONE's success_condition OR failure_condition (per the CAMPAIGN BIBLE section
above) — this triggers generating the next milestone, so don't set it early/speculatively, and
don't forget to set it once it's genuinely true (success OR failure both count as "complete",
the story simply continues differently).
milestone_outcome_summary: REQUIRED (English, ONE sentence) whenever milestone_complete=true —
a factual recap of what actually happened (who's alive/dead, what was learned/lost, which path
was taken) for a system that plans the next milestone from it. Leave "" when milestone_complete
is false.
act_complete: set true ONLY when milestone_complete=true AND resolving this milestone also
satisfies the CURRENT ACT's exit_condition (per the campaign section above) — most milestones do
NOT end an act, so default to false. Never true if milestone_complete is false.

FINAL CHECK before you write "story": every single word must be Vietnamese — re-read each word
as you write it; if any character isn't Vietnamese (Chinese, English, or otherwise, other than
an English monster/location name), fix it immediately, don't let it slip through.
"""


# ---------------------------------------------------------------------------
# Chat / gameplay
# ---------------------------------------------------------------------------

def _parse_dm_json(reply: str) -> dict:
    try:
        clean_reply = reply.replace("```json", "").replace("```", "").strip()
        result = json.loads(clean_reply)
    except json.JSONDecodeError:
        result = {"story": reply, "mechanics": {"roll_type": "normal", "reasoning": ""}, "choices": []}
    # Hậu kỳ: qwen3 thỉnh thoảng lẫn ký tự Hán vào câu tiếng Việt dù prompt đã
    # cấm rõ — xem agents/text_utils.py. Áp đệ quy lên toàn bộ result (story,
    # choices[].text, mechanics.entities[]...) tại 1 điểm duy nhất.
    return text_utils.strip_cjk_deep(result)


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


async def handle_chat_classify(data: dict) -> dict:
    """Phase 1/3 của 1 lượt /chat: phân loại hành động (LLM call #1), KHÔNG
    tung xúc xắc, KHÔNG gọi LLM kể chuyện. Lưu mọi state cần cho các phase sau
    vào cột character.pending_action, trả về bản rút gọn để frontend quyết
    định có cần hiện popup dice-roll hay không (needs_roll=false -> FE gọi
    thẳng handle_chat_roll rồi handle_chat_narrate, không có popup)."""
    user_input = data.get("message", "")
    char = db.get_latest_character()
    if not char:
        return {"error": "Chưa có nhân vật."}

    # Chụp state NGAY TRƯỚC khi xử lý lượt này -> cho phép nút "Thử lại" ở
    # frontend khôi phục về đây nếu model sinh lỗi ở lượt sắp chạy.
    conn_snap = db.get_conn()
    _snapshot_pre_turn(conn_snap, char["id"], user_input)
    conn_snap.close()

    char_dict = db.character_row_to_dict(char)
    suspicious = _mentions_missing_item(user_input, char_dict)
    current_turn_before = char["current_turn"] or 0

    conn_pre = db.get_conn()
    active_entities = [dict(row) for row in entities.get_active_entities(conn_pre, char["id"])]
    conn_pre.close()

    turns_since_event = char["turns_since_event"] or 0
    # Ngưỡng nâng từ 2 lên 3 — turns_since_event giờ ĐƯỢC cập nhật thật (xem
    # cuối hàm, dựa vào mechanics.event_occurred model trả về), trước đây cột
    # này không bao giờ được ghi nên force_event luôn = False (chết, vô tác
    # dụng). Nới ngưỡng lên 1 chút để có chỗ thở, tránh dồn dập ép "sự kiện
    # lớn" mỗi 2 lượt chồng lên guard chống spam quái bên dưới.
    force_event = turns_since_event >= 3

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

    # --- Guard chống spam quái: 1 quái thù địch MỚI vừa xuất hiện trong lượt
    # này/lượt trước -> cấm sinh thêm quái mới, bắt buộc phải khai triển/giải
    # quyết con vừa xuất hiện trước. Trước đây không có guard này -> có lúc 3
    # quái MỚI xuất hiện liên tiếp trong 3-4 lượt (vd Erased One turn 30/32/33,
    # Memory Wraith turn 44/45/46 trong 1 lần chơi thật) mà không con nào được
    # giải quyết xong, gây cảm giác "đi xíu là gặp quái liên tục".
    recent_hostile_spawns = [
        e for e in active_entities
        if e.get("entity_type") == "monster" and e.get("hostile")
        and db.safe_int(e.get("first_seen_turn"), -99) >= current_turn_before - 1
    ]
    if recent_hostile_spawns:
        recent_names = ", ".join(e["name"] for e in recent_hostile_spawns)
        turn_note += (
            f" PACING GUARD: {recent_names} vừa xuất hiện rất gần đây — KHÔNG được sinh thêm "
            f"quái thù địch MỚI nào trong lượt này. Hãy khai triển/giải quyết mối đe doạ hiện "
            f"có (giao chiến, để nó bỏ chạy, dùng đối thoại/môi trường) thay vì chồng thêm quái "
            f"mới, trừ khi hành động của người chơi chủ động tìm kiếm nguy hiểm mới."
        )

    # Nhắc lại roster quái của campaign NGAY SÁT cuối context (gần chỗ model
    # sắp sinh ra entity mới nhất) — rule đầy đủ đã có trong system prompt
    # (## CAMPAIGN) nhưng nằm quá xa so với lúc model thực sự quyết định
    # entity mới, dễ bị lãng quên giữa 1 system prompt rất dài. Lặp lại ngắn
    # gọn ở đây (recency) để model khó bỏ qua hơn.
    campaign_bible = db._load_campaign_bible(char)
    current_milestone = db.load_current_milestone(char)
    if campaign_bible:
        roster_names = ", ".join(campaign.monster_roster_names(campaign_bible, current_milestone))
        if roster_names and not recent_hostile_spawns:
            turn_note += (
                f" REMINDER: if a NEW hostile monster must appear THIS turn, prefer one of the "
                f"suggested/recurring names already established: {roster_names} — reuse its exact "
                f"name/species/appearance/moveset/behavior from the CAMPAIGN section above when it "
                f"fits. Only invent something else if none of these fit the scene."
            )

    # --- Van an toàn milestone bị kẹt: model 14B hay quên tự báo
    # mechanics.milestone_complete dù nội dung đã thoả điều kiện -> campaign
    # đứng yên hàng chục lượt, hết chuyện để kể nên chỉ còn biết lặp
    # thoại/spam quái làm "nội dung". SOFT chỉ nhắc mạnh hơn; HARD vừa nhắc
    # mạnh vừa đặt cờ force_milestone_complete (đọc lại ở handle_chat_narrate)
    # để backend TỰ ép milestone_complete=true nếu model vẫn phớt lờ — không
    # còn "chuyển sang milestone kế" bằng cách tăng index vào mảng có sẵn như
    # trước (milestone giờ sinh động, không có mảng để nhảy tới).
    force_milestone_complete = False
    if current_milestone:
        milestone_advanced_turn = char["milestone_advanced_turn"] if "milestone_advanced_turn" in char.keys() else 0
        turns_stuck = current_turn_before - (milestone_advanced_turn or 0)

        if turns_stuck >= MILESTONE_HARD_STUCK_TURNS:
            force_milestone_complete = True
            print(f"[DEBUG] milestone kẹt {turns_stuck} lượt -> ép force_milestone_complete=true")
            turn_note += (
                f" STORY PIVOT: enough time has passed on the current milestone (\"{current_milestone['title']}\") "
                f"— THIS turn, wrap up/resolve it decisively (success or a concrete setback, a clue "
                f"lands, a door opens, the threat is dealt with) and set mechanics.milestone_complete="
                f"true with a milestone_outcome_summary reflecting how it actually ended. Do not stall "
                f"further on this thread."
            )
        elif turns_stuck >= MILESTONE_SOFT_STUCK_TURNS:
            turn_note += (
                f" PACING: {turns_stuck} turns have passed without completing the current milestone "
                f"(\"{current_milestone['title']}\" — {current_milestone['success_condition']}). Actively "
                f"steer this turn's events toward satisfying its success_condition or failure_condition, "
                f"rather than another side detour. Set mechanics.milestone_complete=true as soon as "
                f"either is genuinely met."
            )

    # Nhắc chống lặp thoại/prose ở NGAY SÁT cuối context (recency) — rule đầy
    # đủ đã có trong system prompt (## SCENE CONTINUITY) nhưng nằm quá xa nên
    # model nhỏ (14B) hay quên, dẫn tới lặp gần y hệt 1 câu thoại nhiều lượt
    # liên tiếp (vd NPC từ chối tiết lộ bằng đúng 1 câu 3 lượt liền trong 1
    # lần chơi thật).
    turn_note += (
        " ANTI-REPETITION: if this turn's scene/dialogue would closely restate content, "
        "phrasing, or a refusal already given in the last 2-3 turns, you MUST make it "
        "meaningfully different this time (new detail, changed reaction, escalation, or an "
        "explicit refusal-with-a-reason) — never reuse near-identical lines turn after turn."
    )

    # --- Phá bế tắc hội thoại: nhắc suông ANTI-REPETITION ở trên không đủ với
    # model 14B — nó vẫn có thể đổi câu chữ mỗi lượt nhưng giữ nguyên KẾT CỤC
    # (vd 1 NPC luôn kết thúc bằng "tôi không thể nói gì thêm" dù người chơi
    # đổi chiến thuật, đe doạ, dùng vật phẩm...). _detect_dialogue_deadlock()
    # bắt đúng kiểu lặp này bằng cách đếm tần suất 1 NPC bị nhắc liên tục
    # trong story text, thay vì so độ giống câu chữ (vốn luôn thấp vì model
    # đổi cách diễn đạt mỗi lượt). Khi phát hiện, ép buộc CỨNG: lượt này BẮT
    # BUỘC phải có bước ngoặt thật, chỉ đích danh NPC đó để model không né.
    deadlock_npc = _detect_dialogue_deadlock(campaign_bible)
    if deadlock_npc:
        turn_note += (
            f" BREAKTHROUGH REQUIRED: the last several turns have circled the same standoff with "
            f"\"{deadlock_npc}\" without real progress (same refusal/deflection outcome even if "
            f"worded differently each time). THIS turn MUST break the deadlock decisively — pick "
            f"ONE: (a) \"{deadlock_npc}\" finally cracks and reveals a concrete, specific piece of "
            f"information (even if partial/cryptic, it must give the player something new to act "
            f"on), or (b) an external event forcibly interrupts/ends this standoff (someone else "
            f"arrives, \"{deadlock_npc}\" is taken/killed/flees for good, the location becomes "
            f"unsafe) so the player must move on to a different approach. Under NO circumstances "
            f"repeat another vague refusal, interruption-without-payoff, or stalling non-answer "
            f"this turn — this is the LAST turn this standoff is allowed to continue unresolved."
        )

    # --- Call 1: classify (module classification.py) ---
    prev_result_raw = char["last_result"] if "last_result" in char.keys() else None
    known_choices = None
    if prev_result_raw:
        try:
            known_choices = json.loads(prev_result_raw).get("choices")
        except (TypeError, json.JSONDecodeError, AttributeError):
            known_choices = None

    class_result = classification.classify_action(
        MODEL, OPTIONS, user_input, char_dict, known_choices=known_choices,
        active_entities=active_entities,
    )

    # --- Lưu toàn bộ state cần cho các phase sau (handle_chat_roll rồi
    # handle_chat_narrate) — người chơi
    # có thể bấm popup dice-roll vài giây sau, không gọi lại classify nữa. Đè
    # lên pending cũ nếu có (người chơi đổi ý gõ hành động khác trước khi bấm
    # roll của hành động trước -> hành động cũ bị huỷ, không có gì phải dọn).
    pending = {
        "stage": "classified",
        "user_input": user_input,
        "char_id": char["id"],
        "char_dict": char_dict,
        "active_entities": active_entities,
        "current_turn_before": current_turn_before,
        "turn_note": turn_note,
        "class_result": class_result,
        "force_milestone_complete": force_milestone_complete,
    }
    conn_pending = db.get_conn()
    conn_pending.execute(
        "UPDATE character SET pending_action = ? WHERE id = ?",
        (json.dumps(pending, ensure_ascii=False), char["id"]),
    )
    conn_pending.commit()
    conn_pending.close()

    adv_reason = class_result.get("advantage_reason")
    display_reason = None
    if adv_reason:
        display_reason = {
            "type": adv_reason.get("type"),
            "name": _localize_reason_name(adv_reason.get("type"), adv_reason.get("name"), char_dict),
        }

    return {
        "needs_roll": class_result.get("needs_roll", True),
        "roll_type": class_result.get("advantage_state", "normal"),
        "dc": class_result.get("dc"),
        "contest_type": class_result.get("contest_type", "none"),
        "attribute": class_result.get("attribute"),
        "reason": display_reason,
    }


def _load_pending(expected_stage: str):
    """Đọc pending_action của nhân vật gần nhất, trả (pending, error). error
    khác None nếu không có pending hợp lệ đúng stage đang cần (frontend gọi
    sai thứ tự, hoặc pending đã bị đè bởi 1 hành động mới hơn)."""
    conn_p = db.get_conn()
    row = conn_p.execute(
        "SELECT pending_action FROM character ORDER BY id DESC LIMIT 1"
    ).fetchone()
    conn_p.close()
    if not row or not row["pending_action"]:
        return None, {"error": "Không có hành động nào đang chờ xử lý."}
    try:
        pending = json.loads(row["pending_action"])
    except (TypeError, json.JSONDecodeError):
        return None, {"error": "Không có hành động nào đang chờ xử lý."}
    if pending.get("stage") != expected_stage:
        return None, {"error": "Không có hành động nào đang chờ xử lý."}
    return pending, None


def _apply_consumption(char_id: int, resolution: dict):
    """Tiêu hao tài nguyên (item/mana/cooldown) ứng với 1 resolution đã có sẵn
    — dùng chung cho handle_chat_roll (lần roll đầu) và handle_chat_retry
    (tái áp dụng đúng resolution CŨ sau khi pre-turn snapshot đã hoàn tác nó,
    KHÔNG roll lại)."""
    used_name = resolution["used_name"]
    consumed_kind = resolution["consumed_kind"]
    mana_cost = resolution["mana_cost"]

    if consumed_kind == "item":
        _consume_item(char_id, used_name)
    elif consumed_kind == "skill":
        _put_skill_on_cooldown(char_id, used_name)
        _consume_mana(char_id, mana_cost)

    # Hồi cooldown các skill khác đi 1 lượt
    _tick_cooldowns(char_id, skip=used_name if consumed_kind == "skill" else None)


async def handle_chat_roll() -> dict:
    """Phase 2/3 — gọi ngay khi người chơi bấm "Tung xúc xắc" trên popup (hoặc
    ngay lập tức từ frontend nếu classify trả needs_roll=false). CHỈ tung xúc
    xắc thật + tiêu hao tài nguyên (nhanh, KHÔNG gọi LLM) rồi trả kết quả ngay
    để popup hiện thành/bại tức thì — KHÔNG chờ LLM kể chuyện. Lưu lại kết quả
    resolve vào pending_action (stage "resolved") cho handle_chat_narrate dùng
    tiếp, VÀ vào cột last_turn_resolution (sống sót qua pre-turn snapshot) để
    "Thử lại" sau này tái dùng đúng kết quả roll này, không roll lại."""
    pending, err = _load_pending("classified")
    if err:
        return err

    char_id = pending["char_id"]
    char_dict = pending["char_dict"]
    active_entities = pending["active_entities"]
    class_result = pending["class_result"]

    # --- Kiểm tra tài nguyên + tung xúc xắc thật (module classification.py) ---
    resolution = classification.resolve_action(class_result, char_dict, active_entities=active_entities)

    _apply_consumption(char_id, resolution)

    pending["stage"] = "resolved"
    pending["resolution"] = resolution
    pending_json = json.dumps(pending, ensure_ascii=False)
    conn_r = db.get_conn()
    conn_r.execute(
        "UPDATE character SET pending_action = ?, last_turn_resolution = ? WHERE id = ?",
        (pending_json, pending_json, char_id),
    )
    conn_r.commit()
    conn_r.close()

    return {
        "success": resolution["success"],
        "roll_type": resolution["roll_type"],
        "dice": resolution["dice"],
        "dc": resolution["dc"],
        "target_ac": resolution["target_ac"],
        "attribute": resolution["attribute"],
    }


async def handle_chat_narrate() -> dict:
    """Phase 3/3 — gọi ngay sau handle_chat_roll (người chơi đã thấy kết quả
    dice thật trong popup). Đọc lại resolution đã lưu, gọi LLM kể chuyện, ghi
    DB — y hệt phần sau của /chat trước khi tách."""
    pending, err = _load_pending("resolved")
    if err:
        return err

    conn_clear = db.get_conn()
    conn_clear.execute(
        "UPDATE character SET pending_action = NULL WHERE id = ?",
        (pending["char_id"],),
    )
    conn_clear.commit()
    conn_clear.close()

    user_input = pending["user_input"]
    char_dict = pending["char_dict"]
    active_entities = pending["active_entities"]
    current_turn_before = pending["current_turn_before"]
    turn_note = pending["turn_note"]
    resolution = pending["resolution"]

    char = db.get_latest_character()
    if not char:
        return {"error": "Chưa có nhân vật."}

    # Nạp sẵn ở đây (dùng chung cho cả context_writer.resolve_visual bên dưới
    # lẫn khối milestone_complete cuối hàm) — tránh đọc file/DB 2 lần.
    campaign_bible = db._load_campaign_bible(char)
    current_milestone = db.load_current_milestone(char)

    success = resolution["success"]
    roll_type = resolution["roll_type"]
    dice = resolution["dice"]
    dc = resolution["dc"]
    target_ac = resolution["target_ac"]
    attribute = resolution["attribute"]
    used_name = resolution["used_name"]
    resource_note = resolution["resource_note"]
    adv_reason = resolution["adv_reason"]
    contest_type = resolution["contest_type"]
    target_key = resolution["target_key"]
    damage = resolution["damage"]
    forced_fail = resolution["forced_fail"]
    mana_cost = resolution["mana_cost"]

    reason_str = ""
    if adv_reason:
        reason_str = f" (due to {adv_reason['type']}: {adv_reason['name']})"

    attribute_note = ""
    if attribute:
        attribute_label = ATTR_LABELS.get(attribute.lower(), attribute.upper())
        attribute_note = (
            f" The attribute actually used for this check was {attribute_label} — if your "
            f"\"reasoning\" field mentions a specific stat, it MUST be this one, never a "
            f"different stat, even if another stat would seem more fitting narratively."
        )

    dice_fact = (
        f"INTERNAL MECHANICS (never mention numbers/DC/AC/roll in story): "
        f"outcome={'SUCCESS' if success else 'FAIL'}, roll_type={roll_type}{reason_str}."
        f"{attribute_note} "
        f"If roll_type is disadvantage/advantage, the story may subtly hint at the reason "
        f"(e.g. character struggling due to their weakness) WITHOUT naming stats or rules."
    )

    attacker_name = None
    if contest_type == "hazard" and target_key:
        attacker_name = next(
            (e.get("name") for e in active_entities
             if (e.get("key") or "").strip().lower() == target_key),
            target_key,
        )

    if target_ac is not None:
        dice_fact += (
            f" This was a real ATTACK ROLL (d20 + attack bonus) against the target's Armor "
            f"Class, NOT a generic ability check — {'it beat' if success else 'it failed to beat'} "
            f"the target's AC."
        )
    elif attacker_name is not None:
        dice_fact += (
            f" This was a real ATTACK ROLL made by \"{attacker_name}\" (key={target_key}) against "
            f"the PLAYER's Armor Class, NOT a decision you get to make — the backend already "
            f"rolled it and the attack {'MISSED' if success else 'HIT'}. Narrate only that outcome; "
            f"do not narrate the player dodging/resisting if the roll says it hit, and vice versa."
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

    if contest_type == "none":
        # Chỉ cần nhắc dấu hp khi lượt này KHÔNG phải combat — vì attack/hazard
        # đã bị code ép cứng changes.hp ở bước sau bất kể model viết gì, nhắc
        # ở đây chỉ tốn token vô ích.
        dice_fact += (
            f" REMINDER: hp change must be NEGATIVE if the story shows the character taking "
            f"damage, POSITIVE only for healing, ZERO for no change. Success={success} means "
            f"the character's intended action actually works — narrate accordingly. Do not negative hp if Success=true"
        )
    else:
        dice_fact += (
            " You do NOT need to compute changes.hp yourself this turn — the backend already "
            "decided it from the dice (see DAMAGE fact if any) and will override whatever you "
            "put. Just set changes.hp to 0."
        )

    if mana_cost > 0:
        dice_fact += (
            f" changes.mana is already handled by the backend (-{mana_cost} for using "
            f"\"{used_name}\"); just set changes.mana to 0, it will be overridden regardless."
        )

    if damage:
        if damage["target"] == "entity":
            dice_fact += (
                f" DAMAGE (already rolled by the backend, DO NOT invent a different number): "
                f"the attack on entity key=\"{damage['target_key']}\" deals EXACTLY {damage['total']} "
                f"damage ({damage['dice']['notation']}={damage['dice']['rolls']}+{damage['modifier']}). "
                f"You MUST set mechanics.entities to include {{\"key\": \"{damage['target_key']}\", "
                f"\"hp_change\": -{damage['total']}}} (mark status dead if it drops to 0). "
                f"This attack does NOT hurt the player — set mechanics.changes.hp to 0 for this action."
            )
        elif damage["target"] == "player":
            source = f"\"{attacker_name}\"'s attack" if attacker_name else "the danger"
            dice_fact += (
                f" DAMAGE (already rolled by the backend, DO NOT invent a different number): "
                f"{source} hits the character for EXACTLY {damage['total']} damage "
                f"({damage['dice']['notation']}={damage['dice']['rolls']}). Set mechanics.changes.hp "
                f"to exactly -{damage['total']}."
            )
    elif contest_type == "attack":
        dice_fact += " This attack MISSED — deal 0 damage to the target entity and mechanics.changes.hp must be 0."

    system_prompt = build_system_prompt(char)  # bản gọn ở tin nhắn trước

    conn = db.get_conn()
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
        "content": f"{turn_note}\n\n{dice_fact}\n\nPlayer action: {user_input}"
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
        result["mechanics"]["attribute"] = attribute
        if target_ac is not None:
            result["mechanics"]["target_ac"] = target_ac
        else:
            result["mechanics"]["dc"] = dc

    # --- Ép cứng damage đã roll bằng code — không tin số HP model tự bịa cho
    # attack/hazard. Model chỉ còn quyền quyết magnitude cho các thay đổi HP
    # KHÔNG liên quan combat (vd uống thuốc, nghỉ ngơi).
    if damage or contest_type in ("attack", "hazard"):
        result["mechanics"].setdefault("changes", {})
        result["mechanics"].setdefault("entities", [])
        if damage and damage["target"] == "entity":
            result["mechanics"]["changes"]["hp"] = 0
            ents = result["mechanics"]["entities"]
            target_entry = next(
                (e for e in ents if isinstance(e, dict)
                 and (e.get("key") or "").strip().lower() == damage["target_key"]),
                None,
            )
            if target_entry is None:
                target_entry = {"key": damage["target_key"]}
                ents.append(target_entry)
            target_entry["hp_change"] = -damage["total"]
        elif damage and damage["target"] == "player":
            result["mechanics"]["changes"]["hp"] = -damage["total"]
        elif not damage and contest_type in ("attack", "hazard") and not forced_fail:
            # attack trượt, hoặc hazard né thành công -> không gây sát thương ở đâu cả
            result["mechanics"]["changes"]["hp"] = 0

    new_entities = []
    if "changes" in result["mechanics"]:
        changes = result["mechanics"]["changes"]
        hp_delta = db.safe_int(changes.get("hp", 0))
        if not success and hp_delta > 0:
            print(f"[DEBUG] success=False nhưng hp=+{hp_delta} -> ép về -{abs(hp_delta) or 5}")
            changes["hp"] = -abs(hp_delta) if hp_delta != 0 else -5
        elif success and hp_delta < 0:
            # success=True nhưng model tự bịa HP âm (không qua đường damage đã
            # roll thật ở khối "if damage or contest_type in (...)" phía trên —
            # nếu qua đường đó thì changes["hp"] đã được ép đúng dấu từ trước
            # rồi, không rơi vào đây). Đây luôn là damage KHÔNG có xúc xắc thật
            # đứng sau (vd quái mới xuất hiện "tấn công phủ đầu" nhưng chưa
            # từng được resolve_action() roll) -> không tính là đòn trúng thật,
            # ép về 0. Model chỉ được gây mất máu qua roll_type/dice thật, hoặc
            # đợi lượt sau khi quái đó đã nằm trong ACTIVE ENTITIES để backend
            # tự roll đòn tấn công của nó (xem "attacker_entity" ở resolve_action).
            print(f"[DEBUG] success=True nhưng hp={hp_delta} không qua damage roll thật -> ép về 0")
            changes["hp"] = 0

        # --- RAG: entity/loot xử lý TRƯỚC khi apply items_added, vì cần
        # ledger loot mới nhất để validate ---
        conn2 = db.get_conn()
        pre_turn_alive_keys = {(e.get("key") or "").strip().lower() for e in active_entities}
        new_entities = entities.apply_entity_changes(
            conn2, char["id"], result["mechanics"].get("entities"), turn_number
        )
        entities.register_loot_drops(
            conn2, char["id"], result["mechanics"].get("loot_dropped"), turn_number
        )

        # --- Thưởng XP/gold khi quái VỪA chuyển sang chết TRONG LƯỢT NÀY (chỉ
        # xét những key đã sống trước lượt này, tránh cộng lại cho quái đã
        # chết từ trước) — code tự cộng dồn, không phụ thuộc model có nhớ
        # cộng xp hay không. ---
        if pre_turn_alive_keys:
            placeholders = ",".join("?" * len(pre_turn_alive_keys))
            newly_dead = conn2.execute(
                f"SELECT max_hp FROM entity WHERE character_id = ? AND status = 'dead' "
                f"AND key IN ({placeholders})",
                (char["id"], *pre_turn_alive_keys),
            ).fetchall()
            if newly_dead:
                rewards = [_monster_kill_reward(row["max_hp"]) for row in newly_dead]
                total_xp = sum(r["xp"] for r in rewards)
                total_gold = sum(r["gold"] for r in rewards)
                changes["xp"] = db.safe_int(changes.get("xp", 0)) + total_xp
                changes["gold"] = db.safe_int(changes.get("gold", 0)) + total_gold
                print(f"[DEBUG] {len(newly_dead)} quái chết lượt này -> +{total_xp} xp, +{total_gold} gold (code roll)")
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

    conn = db.get_conn()
    c = conn.cursor()
    updated_hp = c.execute("SELECT hp FROM character WHERE id=?", (char["id"],)).fetchone()["hp"]
    conn.close()

    # Ép cứng: nếu model báo chết HOẶC HP thực tế đã <=0 -> chốt HP về 0, khóa choices
    if character_died or updated_hp <= 0:
        conn = db.get_conn()
        c = conn.cursor()
        c.execute("UPDATE character SET hp = 0 WHERE id = ?", (char["id"],))
        conn.commit()
        conn.close()
        result["mechanics"]["character_died"] = True
        result["mechanics"]["is_dead"] = True
        result["choices"] = []
    else:
        result["mechanics"]["is_dead"] = False

    # --- Campaign milestone: DM tự báo hoàn thành milestone hiện tại qua
    # mechanics.milestone_complete/milestone_outcome_summary/act_complete ->
    # backend append story_state, chuyển act nếu cần, rồi trigger generate
    # milestone KẾ TIẾP bất đồng bộ (agents/milestone.py, KHÔNG await — dùng
    # đúng pattern asyncio.create_task như imagegen). Không tăng nếu nhân vật
    # vừa chết (câu chuyện đã kết thúc). Bỏ 3 field này khỏi mechanics trả về
    # frontend — chỉ là tín hiệu nội bộ, không phải thứ người chơi cần thấy.
    force_milestone_complete = pending.get("force_milestone_complete", False)
    milestone_complete = bool(result["mechanics"].pop("milestone_complete", False))
    outcome_summary = str(result["mechanics"].pop("milestone_outcome_summary", "") or "").strip()
    act_complete_flag = bool(result["mechanics"].pop("act_complete", False))

    if force_milestone_complete and not milestone_complete:
        milestone_complete = True
        if not outcome_summary:
            outcome_summary = "The milestone stalled without a clear resolution and the story was forced to move on."
        act_complete_flag = False  # backend ép complete -> không tự bịa act cũng xong theo

    milestone_advanced_this_turn = False
    if milestone_complete and not result["mechanics"].get("character_died"):
        bible_for_progress = campaign_bible
        current_ms = current_milestone
        if bible_for_progress and current_ms:
            nb = campaign._normalize_bible(bible_for_progress)
            total_acts = len(nb["acts"])
            current_act_idx = char["campaign_act_index"] if "campaign_act_index" in char.keys() else 0
            is_campaign_finale = current_act_idx >= total_acts - 1
            campaign_finished = bool(act_complete_flag and is_campaign_finale)
            new_act_idx = min(current_act_idx + 1, total_acts - 1) if (act_complete_flag and not is_campaign_finale) else current_act_idx

            prior_state = char["story_state"] or ""
            advanced_beats = current_ms.get("story_beats_advanced") or []
            beats_note = f" [Beats: {', '.join(advanced_beats)}]" if advanced_beats else ""
            new_state = (prior_state + f"\n[Milestone: {current_ms.get('title', '')}] {outcome_summary or 'Completed.'}{beats_note}").strip()
            new_milestone_number = (char["campaign_milestone_number"] or 0) + 1

            conn_ms = db.get_conn()
            conn_ms.execute(
                "UPDATE character SET story_state = ?, campaign_act_index = ?, campaign_milestone_number = ?, "
                "milestone_advanced_turn = ?, current_milestone = NULL WHERE id = ?",
                (new_state, new_act_idx, new_milestone_number, turn_number, char["id"]),
            )
            conn_ms.commit()
            conn_ms.close()
            milestone_advanced_this_turn = True
            print(
                f"[DEBUG] milestone_complete=true -> story_state appended, act {current_act_idx}->{new_act_idx}, "
                f"milestone_number={new_milestone_number}, campaign_finished={campaign_finished}"
            )

            if not campaign_finished:
                # Bất đồng bộ hoàn toàn — KHÔNG await, response trả về ngay,
                # milestone mới sẽ sẵn sàng trước khi player thật sự cần tới
                # (build_system_prompt có fallback text nếu chưa kịp xong).
                asyncio.create_task(_generate_next_milestone_async(
                    char["id"], bible_for_progress, new_state, new_act_idx,
                    new_milestone_number, nb["campaign"]["estimated_length"]["target_milestones"],
                ))
            else:
                print("[DEBUG] Act 3 hoàn thành -> campaign kết thúc, không sinh milestone mới.")

    # --- turns_since_event: đếm số lượt liên tiếp KHÔNG có "sự kiện lớn" (theo
    # mechanics.event_occurred model tự báo) -> dùng để ép force_event ở đầu
    # hàm lượt sau. Cột này trước đây không bao giờ được ghi (luôn = 0 chết
    # cứng) khiến force_event vô tác dụng — giờ mới thực sự wiring theo đúng
    # thiết kế ban đầu. Milestone vừa tiến cũng tính là 1 "sự kiện lớn" (reset
    # về 0), vì bản thân việc đó đã đủ nặng đô cho lượt này.
    event_occurred = bool(result["mechanics"].get("event_occurred", False)) or milestone_advanced_this_turn
    new_turns_since_event = 0 if event_occurred else (char["turns_since_event"] or 0) + 1

    # --- Context panel (ảnh location/quái/NPC mới xuất hiện) — pop "location"
    # khỏi mechanics (chỉ tín hiệu nội bộ, giống milestone_complete, không phải
    # thứ frontend cần trong response text). Ưu tiên entity mới (quái/NPC) hơn
    # location vì "nóng" hơn về mặt kể chuyện nếu cả 2 cùng xảy ra 1 lượt.
    location_context = result["mechanics"].pop("location", None)
    if not isinstance(location_context, dict):
        location_context = None

    # DM không còn tự viết description/visual_prompt cho entity/location MỚI
    # nữa (xem system prompt ## ENTITIES / ## LOCATION) — resolve_visual() tra
    # bảng Bible/Milestone trước (miễn phí), chỉ fallback qua 1 LLM call RIÊNG
    # (model nhỏ) khi DM bịa tên hoàn toàn mới ngoài roster. Chỉ cần resolve
    # đúng 1 lần cho context_update (thứ duy nhất thật sự dùng đến 2 field
    # này), không phải cho mọi entity mới trong lượt.
    context_update = None
    if new_entities:
        e = new_entities[-1]
        kind = "monster" if e["entity_type"] == "monster" else "npc"
        loop_ctx = asyncio.get_running_loop()
        resolved = await loop_ctx.run_in_executor(
            _BG_LLM_EXECUTOR, context_writer.resolve_visual,
            kind, e["name"], campaign_bible, current_milestone, result.get("story", ""),
        )
        context_update = {
            "kind": kind,
            "name": e["name"],
            "description": resolved["description"],
            "visual_prompt": resolved["visual_prompt"],
        }
    elif location_context and location_context.get("name"):
        loc_name = str(location_context["name"]).strip()
        loop_ctx = asyncio.get_running_loop()
        resolved = await loop_ctx.run_in_executor(
            _BG_LLM_EXECUTOR, context_writer.resolve_visual,
            "location", loc_name, campaign_bible, current_milestone, result.get("story", ""),
        )
        context_update = {
            "kind": "location",
            "name": loc_name,
            "description": resolved["description"],
            "visual_prompt": resolved["visual_prompt"],
        }

    if context_update:
        conn_ctx = db.get_conn()
        conn_ctx.execute(
            "UPDATE character SET context_kind = ?, context_name = ?, context_desc = ?, "
            "context_image_path = NULL WHERE id = ?",
            (context_update["kind"], context_update["name"], context_update["description"], char["id"]),
        )
        conn_ctx.commit()
        conn_ctx.close()
        # Bất đồng bộ hoàn toàn — KHÔNG await, response trả về frontend ngay lập
        # tức, ảnh xuất hiện sau (frontend tự poll /scene_context tới khi có).
        asyncio.create_task(imagegen.ensure_context_image(
            char["id"], context_update["kind"], context_update["name"], context_update["visual_prompt"]
        ))

    conn = db.get_conn()
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
        "UPDATE character SET last_result = ?, current_turn = ?, turns_since_event = ? WHERE id = ?",
        (json.dumps(result, ensure_ascii=False), turn_number, new_turns_since_event, char["id"]),
    )
    conn.commit()
    conn.close()

    return result


async def handle_chat_retry() -> dict:
    """Khôi phục state về NGAY TRƯỚC lượt /chat gần nhất (xem
    _snapshot_pre_turn/_restore_pre_turn) rồi kể lại câu chuyện cho ĐÚNG kết
    quả xúc xắc người chơi đã roll (last_turn_resolution) — dùng khi model
    kể chuyện sinh lỗi (lộ tên NPC chưa giới thiệu, quái ngoài roster, lẫn
    tiếng khác...). KHÔNG classify lại, KHÔNG roll lại — xúc xắc là quyết định
    của người chơi, "Thử lại" chỉ redo phần kể chuyện của LLM. Chỉ thử lại
    được lượt GẦN NHẤT."""
    char = db.get_latest_character()
    if not char:
        return {"error": "Chưa có nhân vật."}

    last_turn_raw = char["last_turn_resolution"] if "last_turn_resolution" in char.keys() else None
    last_turn = None
    if last_turn_raw:
        try:
            last_turn = json.loads(last_turn_raw)
        except (TypeError, json.JSONDecodeError):
            last_turn = None

    # Kiểm tra last_turn TRƯỚC khi phục hồi state — nếu không có gì để thử
    # lại, đừng hoàn tác state rồi mới báo lỗi (sẽ để game kẹt ở trạng thái đã
    # rollback nhưng chưa kể lại lượt nào cả).
    if last_turn is None:
        return {"error": "Không có lượt nào để thử lại."}

    conn = db.get_conn()
    replayed_input = _restore_pre_turn(conn, char["id"])
    conn.close()

    if replayed_input is None:
        return {"error": "Không có lượt nào để thử lại."}

    # pre-turn snapshot vừa hoàn tác tiêu hao item/mana/cooldown của lượt gốc
    # -> áp lại ĐÚNG resolution cũ (không random lại) để state nhất quán với
    # kết quả roll người chơi đã thấy, rồi để handle_chat_narrate kể lại.
    _apply_consumption(last_turn["char_id"], last_turn["resolution"])

    last_turn["stage"] = "resolved"
    conn2 = db.get_conn()
    conn2.execute(
        "UPDATE character SET pending_action = ? WHERE id = ?",
        (json.dumps(last_turn, ensure_ascii=False), last_turn["char_id"]),
    )
    conn2.commit()
    conn2.close()

    result = await handle_chat_narrate()
    if isinstance(result, dict):
        result["replayed_input"] = replayed_input
    return result


async def handle_start_game() -> dict:
    char = db.get_latest_character()
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
    conn = db.get_conn()
    c = conn.cursor()
    c.execute("UPDATE character SET region = ? WHERE id = ?", (region, char["id"]))
    conn.commit()
    conn.close()
    char = db.get_latest_character()  # reload để có region mới

    system_prompt = build_system_prompt(char)
    opening_char = db.character_row_to_dict(char)

    # Nếu nhân vật có Campaign Bible + milestone 1 (đã sinh sẵn bởi
    # run_campaign_setup trước khi hàm này chạy — xem main.py), cảnh mở đầu
    # PHẢI diễn ra đúng tại location của milestone 1 đó (không phải 1 vùng
    # Forgotten Realms bịa ngẫu nhiên) — milestone 1's location CHÍNH LÀ điểm
    # bắt đầu, bible không còn field starting_location riêng nữa.
    campaign_bible = db._load_campaign_bible(char)
    opening_milestone = db.load_current_milestone(char)
    opening_location = (opening_milestone or {}).get("location") or {}

    if campaign_bible and opening_location.get("name"):
        location_instruction = (
            f"This campaign's world is NOT the Forgotten Realms — it is the original setting "
            f"defined in the CAMPAIGN BIBLE section above. The opening scene MUST take place at "
            f"the exact location already specified there: \"{opening_location['name']}\" — "
            f"{opening_location.get('description', '')}. Weave the CURRENT MILESTONE's objective "
            f"naturally into the first couple of sentences — do not invent a different location."
        )
    else:
        location_instruction = (
            f"The adventure takes place in the {region} region of the Forgotten Realms. Invent a "
            f"fitting minor location within this region."
        )

    opening_instruction = f"""
Start a brand-new Dungeons & Dragons 5e campaign.
This is the opening scene only.
{location_instruction}
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

{"All encountered monsters and locations should fit this campaign's original world/setting from the CAMPAIGN BIBLE above, not the Forgotten Realms." if campaign_bible else "All encountered monsters and locations should be in the Forgotten Realms."}
All monster and location names must be English.
"""

    # RAG lore context + world state (weather/day-night) TẠM THỜI TẮT — xem
    # ghi chú tương tự trong /chat. region vẫn được chốt/lưu bình thường vì
    # opening_instruction cần nó để đặt bối cảnh mở đầu.

    # Chạy qua executor thay vì gọi ollama.chat() trực tiếp — hàm này giờ
    # cũng được orchestrator tạo campaign mới (main.py) await TRONG lúc
    # frontend đang poll /setup_status; gọi blocking trực tiếp ở đây sẽ chặn
    # luôn cả polling đó. Khi gọi từ route /start_game bình thường (request-
    # response 1 lượt, không cần polling song song) hành vi vẫn đúng y hệt,
    # chỉ là chạy trên thread pool thay vì thread chính.
    loop = asyncio.get_running_loop()
    response = await loop.run_in_executor(
        _BG_LLM_EXECUTOR,
        lambda: ollama.chat(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": opening_instruction},
            ],
            format="json",
            options=OPTIONS,
            think=False,
        ),
    )
    reply = response["message"]["content"]

    result = _parse_dm_json(reply)
    result.setdefault("mechanics", {})
    # Field nội bộ — opening turn không cần model tự báo cái nào trong số này
    # (milestone 1 vừa được tạo sẵn, chưa thể complete/act_complete ngay).
    result["mechanics"].pop("location", None)
    result["mechanics"].pop("milestone_complete", None)
    result["mechanics"].pop("milestone_outcome_summary", None)
    result["mechanics"].pop("act_complete", None)

    # RAG: nếu scene mở đầu đã giới thiệu quái/NPC/loot ngay lập tức, vẫn phải
    # lưu vào DB — nếu không, lượt /chat kế tiếp sẽ không nhận ra key đó đã tồn tại.
    conn = db.get_conn()
    entities.apply_entity_changes(conn, char["id"], result["mechanics"].get("entities"), 0)
    entities.register_loot_drops(conn, char["id"], result["mechanics"].get("loot_dropped"), 0)
    conn.close()

    # Context panel: cảnh mở đầu = location của milestone 1 -> ghi text ngay
    # (đồng bộ, rẻ) rồi generate ảnh bất đồng bộ (KHÔNG await) — tái dùng
    # đúng imagegen.ensure_context_image() đã build cho tính năng ảnh context.
    if opening_location.get("name"):
        conn_ctx = db.get_conn()
        conn_ctx.execute(
            "UPDATE character SET context_kind = 'location', context_name = ?, context_desc = ?, "
            "context_image_path = NULL WHERE id = ?",
            (opening_location["name"], opening_location.get("description", ""), char["id"]),
        )
        conn_ctx.commit()
        conn_ctx.close()
        asyncio.create_task(imagegen.ensure_context_image(
            char["id"], "location", opening_location["name"],
            opening_location.get("visual_prompt") or opening_location["name"],
        ))

    # Lưu vào DB (bao gồm last_result để /start_game gọi lại sau này resume đúng)
    conn = db.get_conn()
    c = conn.cursor()
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
    conn = db.get_conn()
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
        db.safe_int(changes.get("hp", 0)),
        db.safe_int(changes.get("mana", 0)),
        db.safe_int(changes.get("gold", 0)),
        db.safe_int(changes.get("xp", 0)),
        char_id,
    ))

    items_added = changes.get("items_added") or []
    if items_added:
        row = c.execute("SELECT items FROM character WHERE id = ?", (char_id,)).fetchone()
        items = [db._normalize_item(i) for i in db._load_json(row["items"])]

        for name in items_added:
            name = (name or "").strip()
            if not name:
                continue
            existing = next((it for it in items if db._item_matches(it, name)), None)
            if existing:
                existing["quantity"] = existing.get("quantity", 1) + 1
            else:
                # AI chỉ biết tiếng Anh nên vi/en tạm giống nhau cho vật phẩm mới nhặt được.
                items.append({"key": None, "vi": name, "en": name, "consumable": True, "quantity": 1})

        c.execute("UPDATE character SET items = ? WHERE id = ?", (db._list_json(items), char_id))

    conn.commit()
    conn.close()


def _mentions_missing_item(user_input: str, char_dict: dict) -> str | None:
    """Heuristic: nếu câu lệnh có dạng 'dùng/sử dụng/tấn công bằng <X>' mà X
    không khớp equipment/items/skills nào của nhân vật -> trả về tên X.
    Không hoàn hảo nhưng đủ để chặn case rõ ràng (sai tên vũ khí, bịa skill)."""
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
    conn = db.get_conn(); c = conn.cursor()
    row = c.execute(
        "SELECT content FROM history WHERE role='assistant' ORDER BY id DESC LIMIT 1"
    ).fetchone()
    conn.close()
    if not row:
        return None
    return row["content"] or None


def _consume_mana(char_id, amount):
    """Trừ mana trực tiếp trong DB khi dùng skill. Trước đây việc trừ mana
    hoàn toàn phụ thuộc vào model tự ý ghi mechanics.changes.mana trong JSON
    trả về — không có gì ép buộc, nên rất hay bị bỏ sót (dùng skill mà mana
    không giảm). Giờ backend chủ động trừ ngay khi skill thực sự được dùng,
    không phụ thuộc vào model nữa."""
    amount = db.safe_int(amount, 0)
    if amount <= 0:
        return
    conn = db.get_conn(); c = conn.cursor()
    c.execute("UPDATE character SET mana = MAX(0, mana - ?) WHERE id = ?", (amount, char_id))
    conn.commit(); conn.close()


def _consume_item(char_id, name):
    conn = db.get_conn(); c = conn.cursor()
    row = c.execute("SELECT items FROM character WHERE id=?", (char_id,)).fetchone()
    items = [db._normalize_item(i) for i in db._load_json(row["items"])]
    for it in items:
        if db._item_matches(it, name):
            if it.get("consumable", False):
                it["quantity"] = max(0, it.get("quantity", 1) - 1)
            break
    items = [it for it in items if not (it.get("consumable") and it.get("quantity", 1) <= 0)]
    c.execute("UPDATE character SET items=? WHERE id=?", (db._list_json(items), char_id))
    conn.commit(); conn.close()


def _put_skill_on_cooldown(char_id, name):
    conn = db.get_conn(); c = conn.cursor()
    row = c.execute("SELECT skills FROM character WHERE id=?", (char_id,)).fetchone()
    skills = [db._normalize_skill(s) for s in db._load_json(row["skills"])]
    for sk in skills:
        if db._item_matches(sk, name) and sk.get("cooldown_max", 0) > 0:
            sk["cooldown_current"] = sk["cooldown_max"]
            break
    c.execute("UPDATE character SET skills=? WHERE id=?", (db._list_json(skills), char_id))
    conn.commit(); conn.close()


def _tick_cooldowns(char_id, skip=None):
    conn = db.get_conn(); c = conn.cursor()
    row = c.execute("SELECT skills FROM character WHERE id=?", (char_id,)).fetchone()
    skills = [db._normalize_skill(s) for s in db._load_json(row["skills"])]
    for sk in skills:
        if skip and db._item_matches(sk, skip):
            continue
        if sk.get("cooldown_current", 0) > 0:
            sk["cooldown_current"] -= 1
    c.execute("UPDATE character SET skills=? WHERE id=?", (db._list_json(skills), char_id))
    conn.commit(); conn.close()
