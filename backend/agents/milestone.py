"""
milestone.py — Sinh MILESTONE kế tiếp, TỪNG CÁI MỘT, dựa trên Campaign Bible
(cố định — xem campaign.py) + story_state (những gì người chơi đã thực sự làm
ở các milestone trước). Đây là tầng NỘI DUNG DISPOSABLE: NPC/quái/location của
1 milestone có thể biến mất vĩnh viễn sau khi qua, không cần nhồi vào Bible —
đúng nguyên tắc "xoá cái này có hỏng cốt truyện chính không? Không -> ở đây".

Vì milestone được sinh DỰA TRÊN diễn biến thật (không viết sẵn từ đầu như
story_milestones cũ), nó luôn phản ánh đúng những gì người chơi vừa làm — nếu
người chơi né combat, milestone tiếp theo không giả vờ như trận đó đã xảy ra.
Đây là cách giải quyết tận gốc vấn đề "DM lặp vòng quanh 1 milestone cố định
khi người chơi không đi đúng kịch bản đã viết sẵn".

Bible có field "story_beats" (các điểm cốt truyện BẮT BUỘC phải xảy ra, gắn
với 1 act cụ thể — xem campaign.py) — mỗi lần sinh milestone, module này chỉ
mớm đúng các beat thuộc ACT HIỆN TẠI và để model tự quyết milestone này có
"đạt" beat nào không (field trả về "story_beats_advanced": [id,...]), tự bịa
CÁCH nào để đạt chứ không bị ép kịch bản cứng. dungeon_master.py append các id
đã đạt vào story_state (tag "[Beats: ...]") khi milestone hoàn thành, để các
lần sinh milestone sau biết beat nào đã xong mà không target lại.

LƯU Ý: generate_milestone() gọi ollama.chat() ĐỒNG BỘ (blocking) — nơi gọi
(dungeon_master.py lúc milestone hoàn thành, hoặc main.py lúc setup campaign
mới) PHẢI chạy qua loop.run_in_executor(...), không được gọi trực tiếp trong
1 coroutine đang chạy trên event loop chính, nếu không sẽ chặn toàn bộ server
(kể cả các request không liên quan) trong suốt thời gian generate.
"""

import json

import ollama

from . import text_utils

MILESTONE_MODEL = "qwen3:14b"

# Schema milestone nhỏ hơn hẳn Campaign Bible -> ngân sách token thấp hơn,
# nhưng vẫn bật think=True để model tự self-check tính liền mạch với story_state.
MILESTONE_OPTIONS = {"num_ctx": 8192, "num_predict": 2200, "temperature": 0.9}


_DEFAULT_MILESTONE = {
    "title": "Follow the trail",
    "objective": "Find the next concrete lead toward the main goal.",
    "story_purpose": "Keeps the investigation moving without stalling.",
    "success_condition": "The character acts on any lead available in the current scene.",
    "failure_condition": "The character spends many turns avoiding the plot entirely with no progress.",
    "required_reveal": "",
    "forbidden_reveal": "",
    "npcs": [],
    "location": {"name": "", "description": "", "visual_prompt": ""},
    "possible_encounters": [],
    "possible_rewards": [],
    "next_milestone_hint": "",
    "act_complete": False,
    "story_beats_advanced": [],
}


def _s(d, key, default=""):
    return str(d.get(key) or default).strip() if isinstance(d, dict) else default


def _parse_milestone_json(reply: str):
    try:
        clean = reply.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(clean)
    except json.JSONDecodeError:
        return None
    return text_utils.strip_cjk_deep(parsed)  # xem agents/text_utils.py


def _normalize_milestone(obj: dict) -> dict:
    """Cùng triết lý với campaign._normalize_bible: field thiếu/hỏng rơi về
    default TƯƠNG ỨNG, không rơi về default nguyên khối."""
    if not isinstance(obj, dict) or not obj.get("title"):
        obj = {}

    npcs = []
    for i, n in enumerate(obj.get("npcs") or []):
        if isinstance(n, dict) and n.get("name"):
            gender_raw = _s(n, "gender").lower()
            npcs.append({
                "key": (n.get("key") or n.get("name") or f"npc_{i + 1}").strip().lower().replace(" ", "_"),
                "name": _s(n, "name"),
                "gender": gender_raw if gender_raw in ("male", "female") else "",
                "role": _s(n, "role", "npc"),
                "desc": _s(n, "desc"),
                "appearance": _s(n, "appearance"),
                "visual_prompt": _s(n, "visual_prompt"),
            })

    loc_raw = obj.get("location")
    if isinstance(loc_raw, dict) and loc_raw.get("name"):
        location = {
            "name": _s(loc_raw, "name"),
            "description": _s(loc_raw, "description"),
            "visual_prompt": _s(loc_raw, "visual_prompt"),
        }
    else:
        location = dict(_DEFAULT_MILESTONE["location"])

    encounters = []
    for e in obj.get("possible_encounters") or []:
        if isinstance(e, dict) and e.get("name"):
            encounters.append({
                "name": _s(e, "name"), "type": _s(e, "type", "monster"),
                "appearance": _s(e, "appearance"), "visual_prompt": _s(e, "visual_prompt"),
                "moveset": _s(e, "moveset"), "behavior": _s(e, "behavior"),
            })
        elif isinstance(e, str) and e.strip():
            encounters.append({"name": e.strip(), "type": "monster", "appearance": "", "visual_prompt": "", "moveset": "", "behavior": ""})

    rewards_raw = obj.get("possible_rewards")
    rewards = [str(r).strip() for r in rewards_raw if str(r).strip()] if isinstance(rewards_raw, list) else []

    beats_raw = obj.get("story_beats_advanced")
    beats_advanced = (
        [str(b).strip().lower() for b in beats_raw if str(b).strip()]
        if isinstance(beats_raw, list) else []
    )

    return {
        "title": _s(obj, "title", _DEFAULT_MILESTONE["title"]),
        "objective": _s(obj, "objective", _DEFAULT_MILESTONE["objective"]),
        "story_purpose": _s(obj, "story_purpose", _DEFAULT_MILESTONE["story_purpose"]),
        "success_condition": _s(obj, "success_condition", _DEFAULT_MILESTONE["success_condition"]),
        "failure_condition": _s(obj, "failure_condition", _DEFAULT_MILESTONE["failure_condition"]),
        "required_reveal": _s(obj, "required_reveal"),
        "forbidden_reveal": _s(obj, "forbidden_reveal"),
        "npcs": npcs,
        "location": location,
        "possible_encounters": encounters,
        "possible_rewards": rewards,
        "next_milestone_hint": _s(obj, "next_milestone_hint"),
        "act_complete": bool(obj.get("act_complete", False)),
        "story_beats_advanced": beats_advanced,
    }


def _build_milestone_prompt(bible: dict, story_state: str, act_index: int, milestone_number: int, target_total: int) -> str:
    from . import campaign  # tránh circular import ở module scope (campaign.py không import milestone.py, nhưng giữ thói quen an toàn)

    b = campaign._normalize_bible(bible)
    acts = b["acts"]
    idx = max(0, min(act_index, len(acts) - 1))
    act = acts[idx]

    hidden_truths = "\n".join(f"- [{t['id']}] {t['description']}" for t in b["story"]["hidden_truths"]) or "None"
    act_beats = [bt for bt in b.get("story_beats", []) if bt.get("act") == act["act"]]
    beats_lines = "\n".join(f"- [{bt['id']}] {bt['description']}" for bt in act_beats) or "None"
    npc_lines = "\n".join(
        f"- {n['name']} ({n['key']}) — wants: {n['motivation']} | secret: {n['secrets'] or '-'}"
        for n in b["characters"]["key_npcs"]
    ) or "None"
    monster_lines = "; ".join(m["name"] for m in b["content"]["major_monsters"] if m.get("name")) or "None"
    location_lines = "; ".join(l["name"] for l in b["world"]["major_locations"] if l.get("name")) or "None"

    story_state_block = story_state.strip() if story_state and story_state.strip() else "(Nothing has happened yet — this is the very first milestone, the opening of the campaign.)"

    remaining = max(1, target_total - milestone_number + 1)

    return f"""You are the MILESTONE DESIGNER for a solo D&D 5e dark-fantasy RPG. A separate
Dungeon Master AI narrates turn-by-turn; you generate ONE milestone at a time — a short,
playable stretch of story — based on the fixed Campaign Bible below AND what the player has
ACTUALLY done so far. Unlike the Bible, everything you invent here (NPCs, a location, possible
enemies) is DISPOSABLE — it can vanish forever once this milestone ends and never needs to be
referenced again. Do NOT invent anything that would need to persist beyond this milestone; if
it feels campaign-critical, it should already be in the Bible, not something you add here.

## CAMPAIGN BIBLE (canon — stay consistent with this, never contradict it)
Title: {b['campaign']['title']} | Genre: {b['campaign']['genre']} | Tone: {b['campaign']['tone']}
Main goal: {b['story']['main_goal']}
Main conflict: {b['story']['main_conflict']}
Narrative constraints (never violate): {'; '.join(b['story']['narrative_constraints']) or 'None'}
Hidden truths (secret — only reveal through required_reveal when it's actually time, never all at once):
{hidden_truths}
Key NPCs available (reuse these if it fits — do not always invent new ones): {npc_lines}
Recurring monsters available (reuse if it fits): {monster_lines}
Major locations available (reuse if it fits, or invent a minor new one for this milestone only): {location_lines}
Main antagonist: {b['characters']['main_antagonist']['name']} — {b['characters']['main_antagonist']['desc']}

## CURRENT ACT: {act['act']}/3 — {act['purpose']}
This act ends when: {act['exit_condition']}

## STORY BEATS DUE THIS ACT (mandatory plot points fixed by the Bible — you decide IF and HOW
this specific milestone realizes any of them; invent your own concrete scene/method, never copy a
beat's wording into objective/success_condition verbatim). Check STORY STATE below for "[Beats:
...]" tags first — any id listed there is ALREADY done, never re-target it:
{beats_lines}

## STORY STATE (what the player has ACTUALLY done so far, most recent last — the single most
important input here, this milestone MUST react to it, not ignore it):
{story_state_block}

## PACING BUDGET
This will be milestone #{milestone_number} of an estimated {target_total} for the whole
campaign (~{remaining} remaining, including this one). If few remain and this act's
exit_condition still isn't close, make THIS milestone push decisively toward it — do not add a
leisurely detour this late. If this is early, you have room to breathe.

## WHAT TO DESIGN
1. title, objective (what the player is trying to accomplish right now, concretely).
2. story_purpose: why this matters to the main plot (ties back to Bible, one sentence).
3. success_condition: a CONCRETE state that ends this milestone successfully. Must be reachable
   through MULTIPLE possible player approaches (combat, stealth, social, avoidance, sacrifice)
   — never assume the player will pick one specific method.
4. failure_condition: a concrete ACCUMULATED narrative state (never a single failed dice roll)
   that would end this milestone in failure — e.g. a specific NPC dies, a specific window of
   opportunity closes, the character is captured. Failure should still let the story continue
   (a setback, not a game over) unless it's meant to be truly severe.
5. required_reveal: ONE hidden_truth (or partial piece of one) from the Bible that should
   surface by the end of this milestone, if any is due yet — leave empty string if none should
   be revealed this milestone. Do not reveal truths out of order or too early.
6. forbidden_reveal: what must NOT be revealed yet (usually the remaining hidden_truths, or
   specifically the final twist) — a short reminder for the DM.
7. npcs: 0-3 DISPOSABLE npcs for this milestone specifically (name, gender, role, desc, one
   English sentence appearance, visual_prompt = short comma-separated visual traits for image
   gen). May reuse a Bible key_npc instead of inventing one if that fits better — then you can
   omit npcs or list them lightly.
8. location: ONE disposable location for this milestone (name, description, visual_prompt) —
   may reuse a Bible major_location, or invent a minor new one scoped to just this milestone.
9. possible_encounters: 0-3 SUGGESTED (not mandatory) enemies the DM may use if the player's
   actions lead to a fight here — name, type ("monster"|"npc"), appearance, visual_prompt,
   moveset, behavior. These do not need to persist after this milestone.
10. possible_rewards: 0-3 suggested rewards (items/info/allies) if the player succeeds — light
    suggestions, the DM has final say.
11. next_milestone_hint: one sentence pointing toward what might plausibly come after, GIVEN
    multiple possible outcomes of this one — not a guarantee, just a seed for continuity.
12. act_complete: true ONLY if resolving THIS milestone would also satisfy the CURRENT ACT's
    exit_condition above. Most milestones are NOT the last one in an act — default to false
    unless the story_state + this milestone's scope genuinely closes out the act's condition.
13. story_beats_advanced: 0+ ids from STORY BEATS DUE THIS ACT above that resolving THIS
    milestone's success_condition would genuinely realize (the underlying story fact becomes
    true, not just mentioned in passing). Leave [] if this milestone doesn't land any beat yet —
    that's normal, not every milestone has to. Never include an id already marked done via a
    "[Beats: ...]" tag in STORY STATE.

## SELF-CHECK before writing final JSON (silently fix problems, never narrate the check):
- Does this milestone actually react to STORY STATE, not ignore what already happened?
- Is success_condition reachable multiple ways (not forcing one playstyle)?
- Is failure_condition a real accumulated state, not "if the player fails one roll"?
- Does required_reveal (if any) make sense given what's already been revealed in story_state?
- Is everything you invented here genuinely disposable (nothing that secretly needs Bible-level
  permanence)?
- Does story_beats_advanced (if any) actually get realized by success_condition, not just
  name-dropped? Does it avoid any id already done per STORY STATE's "[Beats: ...]" tags?
- LANGUAGE: every string value below must be English only — re-check field by field.

## OUTPUT — ONLY this JSON object, no markdown, no commentary:
{{
  "title": "...", "objective": "...", "story_purpose": "...",
  "success_condition": "...", "failure_condition": "...",
  "required_reveal": "...", "forbidden_reveal": "...",
  "npcs": [{{"key": "snake_case_id", "name": "...", "gender": "male|female", "role": "...", "desc": "...", "appearance": "...", "visual_prompt": "..."}}],
  "location": {{"name": "...", "description": "...", "visual_prompt": "..."}},
  "possible_encounters": [{{"name": "...", "type": "monster|npc", "appearance": "...", "visual_prompt": "...", "moveset": "...", "behavior": "..."}}],
  "possible_rewards": ["..."],
  "next_milestone_hint": "...",
  "act_complete": false,
  "story_beats_advanced": []
}}"""


def generate_milestone(bible: dict, story_state: str, act_index: int, milestone_number: int, target_total: int, model: str = None, options: dict = None) -> dict:
    """1 LLM call (think=True, tự self-check) sinh milestone kế tiếp. Lỗi/
    parse hỏng -> fallback về _DEFAULT_MILESTONE (đủ dùng để campaign không
    bị kẹt cứng, dù kém sinh động hơn)."""
    model = model or MILESTONE_MODEL
    options = options or MILESTONE_OPTIONS
    parsed = None

    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "system", "content": _build_milestone_prompt(bible, story_state, act_index, milestone_number, target_total)}],
            format="json",
            options=options,
            think=True,
        )
        content = response["message"]["content"]
        parsed = _parse_milestone_json(content)
        if parsed is None:
            thinking_len = len(response["message"].get("thinking") or "")
            print(
                f"[DEBUG] generate_milestone: content không parse được (content={content[:200]!r}, "
                f"thinking_len={thinking_len}) -> fallback default."
            )
    except Exception as e:
        print(f"[DEBUG] generate_milestone lỗi ({e}) -> fallback default")

    return _normalize_milestone(parsed)
