"""
campaign.py — Sinh "Campaign Bible": tài liệu tham chiếu DUY NHẤT (single
source of truth) mà AI DM (model nhỏ, ~14B, không có trí nhớ nào ngoài những
gì được nhét vào context mỗi lượt) dựa vào để dẫn dắt toàn bộ ván chơi solo
dài ~50-80 lượt, thay vì bịa tùy hứng từng turn rồi dần lạc đề/đi vòng vòng.

Người chơi chỉ thấy đúng 1 câu "theme" (hook, hiển thị lúc tạo nhân vật) —
toàn bộ phần còn lại bị giấu khỏi UI, chỉ được nhét vào system prompt cho DM
đọc (xem format_campaign_context). Campaign Bible được lưu ra 1 FILE JSON
riêng (backend/campaign_saves/current_campaign.json — xem save/load bên
dưới) thay vì chỉ nằm trong 1 cột TEXT của SQLite, để dễ mở lên xem/theo dõi
lúc dev/debug.

Có 3 điểm vào:
- generate_campaign_hooks(): AI chỉ bịa 5 câu "theme" (hook) ngắn, mỗi câu một
  "vị" khác nhau — KHÔNG sinh phần còn lại ở bước này, để trả kết quả nhanh
  cho người chơi chọn trước khi tốn thời gian khai triển đầy đủ.
- expand_campaign_hook(theme): sau khi người chơi chọn 1 trong 5 hook ở trên,
  khai triển ĐÚNG hook đó (giữ nguyên câu theme) thành đầy đủ Campaign Bible.
- expand_custom_seed(text): người chơi tự viết 1 ý tưởng/theme ngắn, AI khai
  triển ra Campaign Bible bám sát ý tưởng gốc thay vì bịa lạc đề.

generate_campaign_hooks() tắt "think" để lấy tốc độ (chỉ sinh vài câu ngắn).
Cả 2 hàm expand đều bật "think" — model được yêu cầu tự SELF-CHECK (tính nhất
quán timeline, động cơ NPC, thứ tự tiết lộ bí mật, pacing) trong lúc "suy
nghĩ" rồi tự sửa TRƯỚC KHI xuất JSON cuối, thay vì chỉ viết 1 lần và hy vọng
đúng — xem _build_campaign_bible_prompt().
"""

import json
import os
import re

import ollama

CAMPAIGN_MODEL = "qwen3:14b"
SEED_COUNT = 5

# Chỉ sinh vài câu ngắn -> ngân sách token nhỏ hơn nhiều, và think=False vì
# không cần suy luận sâu cho việc bịa 1 câu hook.
HOOKS_OPTIONS = {"num_ctx": 4096, "num_predict": 1200, "temperature": 0.95}

# Campaign Bible có schema lớn hơn hẳn seed cũ (milestones/secrets/factions/
# npcs chi tiết/locations/boss plan_stages...) -> cần num_predict lớn hơn để
# chứa CẢ phần "thinking" (self-check, think=True) lẫn JSON cuối cùng mà
# không bị cắt ngang giữa chừng. Vẫn phải giữ vừa phải: máy chạy ollama ở đây
# chủ yếu CPU-bound (qwen3:14b ~84% CPU/16% GPU theo `ollama ps`), num_predict
# 9000 + num_ctx 16384 từng khiến 1 lần generate mất >10 phút — hạ xuống mức
# này để vẫn đủ chỗ cho schema lớn mà không quá chậm để dùng thực tế.
BIBLE_OPTIONS = {"num_ctx": 8192, "num_predict": 5500, "temperature": 0.9}

# File lưu Campaign Bible ra ngoài (không chỉ nằm trong cột DB) — game này
# chỉ có 1 save-slot (xem main.py: DELETE FROM character trước khi tạo mới),
# nên dùng 1 đường dẫn cố định, ghi đè mỗi lần tạo nhân vật mới.
_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAMPAIGN_SAVE_DIR = os.path.join(_BACKEND_DIR, "game-data", "campaign_saves")
CAMPAIGN_SAVE_PATH = os.path.join(CAMPAIGN_SAVE_DIR, "current_campaign.json")


_DEFAULT_BIBLE = {
    "theme": "Bạn tỉnh dậy giữa tàn tích của một buổi lễ đã thất bại, không nhớ nổi mình đã hứa điều gì với thứ đang chờ dưới lòng đất.",
    "premise": "The character was meant to be the vessel for an interrupted binding ritual. They survived with no memory of that night, but the ritual's true architect — someone they once trusted — is still working to finish what was started.",
    "tone": "grim, paranoid, quietly desperate",
    "setting": {
        "name": "Hollow's Reach",
        "description": "A fog-bound trade town built atop a sealed ritual site, where the old faith never fully died and half the town still leaves offerings at a shrine no one admits to visiting.",
    },
    "starting_location": {
        "name": "The Sealed Shrine",
        "description": "The half-collapsed ritual site where the character wakes, ash and broken chalk circles underfoot, the air still faintly metallic with old magic.",
        "opening_hook": "A single scorched token bearing the character's own name lies within arm's reach — proof someone knew they would be here before they ever arrived.",
    },
    "main_goal": "Uncover what ritual was interrupted, and stop it from completing before it's too late.",
    "plot_twist": "The one orchestrating the ritual's completion is someone the character has trusted since before the story began — their survival that night was not luck, it was arranged.",
    "moral_dilemma": "Expose the orchestrator publicly (justice, but the town descends into chaos and witch-hunts) or handle it quietly (order preserved, but the character becomes complicit in the cover-up).",
    "story_milestones": [
        {"id": "m1", "title": "Wake with no memory", "description": "Character wakes at the ritual site with fragmentary memories and a mark they can't explain.", "npc_involved": None},
        {"id": "m2", "title": "Meet the informant", "description": "Character finds someone with pieces of the truth, willing to trade information for a favor.", "npc_involved": "unknown_informant"},
        {"id": "m3", "title": "Identify the ritual's true purpose", "description": "Character learns the ritual was meant to bind something, not summon it, and why they were chosen as vessel.", "npc_involved": None},
        {"id": "m4", "title": "First confrontation with the cult", "description": "The cultists notice the character asking questions and move against them directly.", "npc_involved": None},
        {"id": "m5", "title": "Discover the orchestrator's identity", "description": "Character finds undeniable proof of who arranged the ritual — someone close to them.", "npc_involved": "unknown_informant"},
        {"id": "m6", "title": "The orchestrator's counter-move", "description": "The orchestrator acts to silence the character or complete the ritual early.", "npc_involved": None},
        {"id": "m7", "title": "Final confrontation", "description": "Character faces the orchestrator and the near-completed ritual, forcing the moral dilemma.", "npc_involved": None},
    ],
    "secrets": [
        {"id": "s1", "description": "The character bears a ritual mark that reacts to cult symbols.", "reveal_tier": "public"},
        {"id": "s2", "description": "The ritual was meant to BIND an ancient entity, not summon it — the character's survival was the failure state, not an accident.", "reveal_tier": "mid_game"},
        {"id": "s3", "description": "The informant already knows the orchestrator's identity but is too afraid to say it outright.", "reveal_tier": "late_game"},
        {"id": "s4", "description": "The orchestrator is someone the character has trusted since before the story began.", "reveal_tier": "finale"},
    ],
    "revelation_order": {"public": ["s1"], "mid_game": ["s2"], "late_game": ["s3"], "finale": ["s4"]},
    "factions": [
        {"id": "f1", "name": "The Hollow Circle", "goal": "Complete the interrupted ritual by any means.", "methods": "Infiltration, blackmail, ritual murder disguised as accidents.", "relationship_to_player": "Hostile once the character starts investigating."},
    ],
    "npcs": [
        {
            "key": "unknown_informant", "name": "Unknown Informant", "gender": "male", "role": "ally",
            "desc": "A cautious contact with fragments of the truth.",
            "personality": "Speaks only in half-truths and riddles, terrified of being overheard.",
            "appearance": "Gaunt, cloaked figure with ink-stained fingers and a nervous twitch.",
            "motivation": "Wants the ritual stopped but is too afraid to act directly.",
            "trusts": "No one fully, but leans on the character out of shared danger.",
            "distrusts": "Anyone connected to the Hollow Circle, including old friends.",
            "secret": "Knows the orchestrator's identity but hasn't said it outright.",
            "autonomous_behavior": "Gathers scraps of information and hides them in dead drops, growing more paranoid and harder to reach as the cult closes in.",
        },
    ],
    "key_locations": [
        {"name": "The Sealed Shrine", "description": "The ritual site itself, half-collapsed and still watched by the cult.", "npc_keys_present": []},
        {"name": "The Informant's Attic", "description": "A cramped hideout above a shuttered shop.", "npc_keys_present": ["unknown_informant"]},
    ],
    "monsters": [
        {
            "name": "Cultist", "species": "Corrupted human",
            "appearance": "Robed figure with ritual scars and a hollow, unblinking stare.",
            "moveset": "Sacrificial dagger stabs, chants that inflict fear at range.",
            "behavior": "Fanatical, fights to the death, flees only to warn others.",
        },
        {
            "name": "Shadow Beast", "species": "Aberration",
            "appearance": "A hound-sized mass of writhing black smoke with glowing red eyes.",
            "moveset": "Lunging bite, brief invisibility, claw swipes that drain vigor.",
            "behavior": "Stalks silently before ambushing; retreats into darkness when badly hurt.",
        },
        {
            "name": "Corrupted Guardian", "species": "Animated construct",
            "appearance": "A cracked stone statue fused with pulsing dark veins of corruption.",
            "moveset": "Slow heavy slams, ground-shaking stomps, brief defensive stone-skin.",
            "behavior": "Relentless and unthinking, never flees, guards a fixed location.",
        },
    ],
    "boss": {
        "name": "The Awakened One", "desc": "The ancient evil at the heart of the plot.",
        "appearance": "A towering, half-formed silhouette of shifting shadow and cracked bone.",
        "moveset": "Reality-warping strikes, summons minor shadow beasts, a devastating area pulse at low health.",
        "behavior": "Calm and taunting at first, grows increasingly violent and desperate as it weakens.",
        "ultimate_goal": "Fully manifest in the mortal world by completing the interrupted ritual.",
        "plan_stages": [
            {"stage": 1, "title": "Recover the lost vessel", "description": "Track down the character, the ritual's original intended vessel."},
            {"stage": 2, "title": "Reassemble the ritual circle", "description": "The cult repairs the damaged shrine and gathers scattered relics."},
            {"stage": 3, "title": "Silence witnesses", "description": "Eliminate or discredit anyone who could expose the ritual, including the informant."},
            {"stage": 4, "title": "Complete the binding", "description": "Force the ritual to completion, with or without the character's consent."},
        ],
    },
    "failure_state": {
        "idle_turn_threshold": 5,
        "description": "The Hollow Circle advances to the next plan_stage regardless — a location the player knows falls under cult control, or the informant goes missing, forcing the player back into the plot with higher stakes.",
    },
    "pacing_guidance": "Milestones 1-2 in turns 1-15 (setup, establish the mystery); milestones 3-4 in turns 16-40 (investigation, first real danger); milestones 5-6 in turns 41-65 (escalation, the twist lands); milestone 7 (finale) in the last 10-15 turns.",
    "dm_directives": {
        "never_reveal": ["The orchestrator's identity before milestone 5", "The true purpose of the ritual (binding, not summoning) before milestone 3"],
        "npc_encounter_rule": "Proactively place NPCs into scenes at their key_locations — do not make the player hunt blindly for someone to talk to.",
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _slugify(text, fallback):
    text = str(text or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text or fallback


def _parse_campaign_json(reply: str):
    try:
        clean = reply.replace("```json", "").replace("```", "").strip()
        return json.loads(clean)
    except json.JSONDecodeError:
        return None


def _normalize_bible(obj: dict) -> dict:
    """Đảm bảo đủ field/đúng kiểu cho Campaign Bible — schema này lớn, model
    có thể bỏ sót/sai bất kỳ field nào; field nào thiếu/hỏng rơi về ĐÚNG phần
    tương ứng của _DEFAULT_BIBLE (không rơi về default nguyên khối), để 1 lỗi
    nhỏ ở 1 field không xoá sạch phần model đã làm đúng ở các field khác."""
    if not isinstance(obj, dict):
        obj = {}

    def s(d, key, default=""):
        return str(d.get(key) or default).strip() if isinstance(d, dict) else default

    theme = s(obj, "theme", _DEFAULT_BIBLE["theme"])
    premise = s(obj, "premise", _DEFAULT_BIBLE["premise"])
    tone = s(obj, "tone", _DEFAULT_BIBLE["tone"])

    setting_raw = obj.get("setting")
    if isinstance(setting_raw, dict) and setting_raw.get("name"):
        setting = {"name": s(setting_raw, "name"), "description": s(setting_raw, "description")}
    else:
        setting = dict(_DEFAULT_BIBLE["setting"])

    start_raw = obj.get("starting_location")
    if isinstance(start_raw, dict) and start_raw.get("name") and start_raw.get("opening_hook"):
        starting_location = {
            "name": s(start_raw, "name"),
            "description": s(start_raw, "description"),
            "opening_hook": s(start_raw, "opening_hook"),
        }
    else:
        starting_location = dict(_DEFAULT_BIBLE["starting_location"])

    main_goal = s(obj, "main_goal", _DEFAULT_BIBLE["main_goal"])
    plot_twist = s(obj, "plot_twist", _DEFAULT_BIBLE["plot_twist"])
    moral_dilemma = s(obj, "moral_dilemma", _DEFAULT_BIBLE["moral_dilemma"])

    milestones_raw = obj.get("story_milestones")
    milestones = []
    if isinstance(milestones_raw, list):
        for i, m in enumerate(milestones_raw):
            if isinstance(m, dict) and m.get("title"):
                npc_involved = s(m, "npc_involved")
                milestones.append({
                    "id": _slugify(m.get("id") or m.get("title"), f"m{i + 1}"),
                    "title": s(m, "title"),
                    "description": s(m, "description"),
                    "npc_involved": npc_involved or None,
                })
    if not milestones:
        milestones = [dict(m) for m in _DEFAULT_BIBLE["story_milestones"]]

    secrets_raw = obj.get("secrets")
    secrets = []
    valid_tiers = {"public", "mid_game", "late_game", "finale"}
    if isinstance(secrets_raw, list):
        for i, sec in enumerate(secrets_raw):
            if isinstance(sec, dict) and sec.get("description"):
                tier = s(sec, "reveal_tier", "mid_game").lower()
                if tier not in valid_tiers:
                    tier = "mid_game"
                secrets.append({
                    "id": _slugify(sec.get("id"), f"s{i + 1}"),
                    "description": s(sec, "description"),
                    "reveal_tier": tier,
                })
    if not secrets:
        secrets = [dict(sec) for sec in _DEFAULT_BIBLE["secrets"]]

    secret_ids = {sec["id"] for sec in secrets}
    revelation_raw = obj.get("revelation_order")
    revelation = {"public": [], "mid_game": [], "late_game": [], "finale": []}
    if isinstance(revelation_raw, dict):
        for tier in revelation:
            ids = revelation_raw.get(tier)
            if isinstance(ids, list):
                revelation[tier] = [i for i in ids if i in secret_ids]
    # Đảm bảo MỌI secret đều xuất hiện đâu đó trong revelation_order (theo
    # đúng reveal_tier của nó) — phòng model khai báo secret nhưng quên liệt
    # kê nó ra trong revelation_order, khiến DM không biết khi nào được tiết lộ.
    already_placed = {i for ids in revelation.values() for i in ids}
    for sec in secrets:
        if sec["id"] not in already_placed:
            revelation[sec["reveal_tier"]].append(sec["id"])

    factions_raw = obj.get("factions")
    factions = []
    if isinstance(factions_raw, list):
        for i, f in enumerate(factions_raw):
            if isinstance(f, dict) and f.get("name"):
                factions.append({
                    "id": _slugify(f.get("id") or f.get("name"), f"f{i + 1}"),
                    "name": s(f, "name"),
                    "goal": s(f, "goal"),
                    "methods": s(f, "methods"),
                    "relationship_to_player": s(f, "relationship_to_player"),
                })
    if not factions:
        factions = [dict(f) for f in _DEFAULT_BIBLE["factions"]]

    npcs_raw = obj.get("npcs")
    npcs = []
    if isinstance(npcs_raw, list):
        for i, n in enumerate(npcs_raw):
            if isinstance(n, dict) and n.get("name"):
                gender_raw = s(n, "gender").lower()
                npcs.append({
                    "key": _slugify(n.get("key") or n.get("name"), f"npc_{i + 1}"),
                    "name": s(n, "name"),
                    "gender": gender_raw if gender_raw in ("male", "female") else "",
                    "role": s(n, "role", "npc"),
                    "desc": s(n, "desc"),
                    "personality": s(n, "personality"),
                    "appearance": s(n, "appearance"),
                    "motivation": s(n, "motivation"),
                    "trusts": s(n, "trusts"),
                    "distrusts": s(n, "distrusts"),
                    "secret": s(n, "secret"),
                    "autonomous_behavior": s(n, "autonomous_behavior"),
                })
    if not npcs:
        npcs = [dict(n) for n in _DEFAULT_BIBLE["npcs"]]

    npc_keys = {n["key"] for n in npcs}
    # Model hay điền npc_involved bằng placeholder string ("nil", "none",
    # "null"...) thay vì JSON null thật, hoặc trỏ tới 1 key không tồn tại
    # trong npcs -> nếu để lọt, DM sẽ đọc thấy "involves NPC: nil" như thể đó
    # là 1 NPC thật. Validate lại: chỉ giữ nếu khớp đúng 1 key có thật.
    for m in milestones:
        if m["npc_involved"] and _slugify(m["npc_involved"], "") not in npc_keys and m["npc_involved"] not in npc_keys:
            m["npc_involved"] = None

    locations_raw = obj.get("key_locations")
    locations = []
    if isinstance(locations_raw, list):
        for l in locations_raw:
            if isinstance(l, dict) and l.get("name"):
                present = l.get("npc_keys_present")
                present = [p for p in present if p in npc_keys] if isinstance(present, list) else []
                locations.append({"name": s(l, "name"), "description": s(l, "description"), "npc_keys_present": present})
    if not locations:
        locations = [dict(l) for l in _DEFAULT_BIBLE["key_locations"]]
    # starting_location PHẢI xuất hiện trong key_locations (DM cần thấy nó ở
    # cả 2 chỗ: bản thân nó + trong danh sách địa điểm chung) — model đôi khi
    # quên lặp lại, tự thêm vào nếu thiếu thay vì bắt DM tự suy luận.
    if not any(l["name"].strip().lower() == starting_location["name"].strip().lower() for l in locations):
        locations.insert(0, {
            "name": starting_location["name"],
            "description": starting_location["description"],
            "npc_keys_present": [],
        })

    monsters_raw = obj.get("monsters")
    monsters = []
    if isinstance(monsters_raw, list):
        for m in monsters_raw:
            if isinstance(m, dict) and m.get("name"):
                monsters.append({
                    "name": s(m, "name"), "species": s(m, "species"),
                    "appearance": s(m, "appearance"), "moveset": s(m, "moveset"),
                    "behavior": s(m, "behavior"),
                })
            elif isinstance(m, str) and m.strip():
                monsters.append({"name": m.strip(), "species": "", "appearance": "", "moveset": "", "behavior": ""})
    if not monsters:
        monsters = [dict(m) for m in _DEFAULT_BIBLE["monsters"]]

    boss_raw = obj.get("boss")
    if isinstance(boss_raw, dict) and boss_raw.get("name"):
        stages_raw = boss_raw.get("plan_stages")
        stages = []
        if isinstance(stages_raw, list):
            for i, st in enumerate(stages_raw):
                if isinstance(st, dict) and st.get("title"):
                    stages.append({
                        "stage": _safe_int(st.get("stage"), i + 1) or (i + 1),
                        "title": s(st, "title"),
                        "description": s(st, "description"),
                    })
        if not stages:
            stages = [dict(st) for st in _DEFAULT_BIBLE["boss"]["plan_stages"]]
        boss = {
            "name": s(boss_raw, "name"), "desc": s(boss_raw, "desc"),
            "appearance": s(boss_raw, "appearance"), "moveset": s(boss_raw, "moveset"),
            "behavior": s(boss_raw, "behavior"),
            "ultimate_goal": s(boss_raw, "ultimate_goal"),
            "plan_stages": stages,
        }
    else:
        boss = json.loads(json.dumps(_DEFAULT_BIBLE["boss"]))

    failure_raw = obj.get("failure_state")
    if isinstance(failure_raw, dict) and failure_raw.get("description"):
        failure = {
            "idle_turn_threshold": max(1, _safe_int(failure_raw.get("idle_turn_threshold"), 5)),
            "description": s(failure_raw, "description"),
        }
    else:
        failure = dict(_DEFAULT_BIBLE["failure_state"])

    pacing_guidance = s(obj, "pacing_guidance", _DEFAULT_BIBLE["pacing_guidance"])

    directives_raw = obj.get("dm_directives")
    never_reveal, npc_encounter_rule = [], ""
    if isinstance(directives_raw, dict):
        nr = directives_raw.get("never_reveal")
        never_reveal = [str(x).strip() for x in nr if str(x).strip()] if isinstance(nr, list) else []
        npc_encounter_rule = s(directives_raw, "npc_encounter_rule")
    if not never_reveal:
        never_reveal = list(_DEFAULT_BIBLE["dm_directives"]["never_reveal"])
    if not npc_encounter_rule:
        npc_encounter_rule = _DEFAULT_BIBLE["dm_directives"]["npc_encounter_rule"]

    return {
        "theme": theme,
        "premise": premise,
        "tone": tone,
        "setting": setting,
        "starting_location": starting_location,
        "main_goal": main_goal,
        "plot_twist": plot_twist,
        "moral_dilemma": moral_dilemma,
        "story_milestones": milestones,
        "secrets": secrets,
        "revelation_order": revelation,
        "factions": factions,
        "npcs": npcs,
        "key_locations": locations,
        "monsters": monsters,
        "boss": boss,
        "failure_state": failure,
        "pacing_guidance": pacing_guidance,
        "dm_directives": {"never_reveal": never_reveal, "npc_encounter_rule": npc_encounter_rule},
    }


# ---------------------------------------------------------------------------
# 5 hook (chỉ câu "theme") do AI tự bịa — nhanh, để người chơi chọn trước khi
# tốn thời gian khai triển đầy đủ
# ---------------------------------------------------------------------------

def _build_hooks_prompt() -> str:
    return f"""You are a D&D 5e campaign designer. Output ONLY a JSON object (no markdown,
no thinking, be fast) with a single key "hooks": an array of EXACTLY {SEED_COUNT} short hook
sentences for a solo dark-fantasy campaign.

Each of the {SEED_COUNT} hooks MUST be a genuinely different FLAVOR/GENRE of story — not
{SEED_COUNT} variations of the same "ancient evil awakens" plot. Spread across distinct vibes
such as (pick {SEED_COUNT} different ones, your choice): political intrigue/betrayal,
heist/theft, revenge/personal vendetta, cosmic horror/eldritch dread, survival/wilderness
disaster, mystery/investigation, war/siege, cursed bloodline/tragedy, rescue/hostage, forbidden
knowledge/cult. No two hooks may share the same core vibe, and each of the {SEED_COUNT} hooks
MUST imply a DIFFERENT WORLD/PLACE from the others — do not reuse the same kingdom/city/region
across hooks, even implicitly.

SETTING: D&D 5e MEDIEVAL DARK-FANTASY (swords, bows, magic, curses, gods, undead, fey,
aberrations). You are NOT limited to the Forgotten Realms specifically — feel free to invent
an original world/region/culture for each hook (its own geography, faction, myth, or curse)
rather than defaulting to generic "the kingdom"/"the realm" phrasing. NEVER introduce sci-fi/
technological/modern elements: no robots, holograms, simulations, computers/code, lasers/
plasma, drones, cyberpunk, or "reality is a simulation/matrix" framing. A hook about illusion/
false reality/stolen identity must be framed through MAGIC (a trickster god's curse, a fey
mirror-realm, a mad archmage's dream-prison, a doppelganger plot) — never through technology.

Each hook: ONE sentence in Vietnamese, SECOND PERSON ("Bạn là...", "Bạn bị...", "Bạn vừa...",
etc.), max ~35 words. It must do BOTH of:
(a) tell the player WHAT ROLE/SITUATION they are in, with one concrete, specific WRONGNESS or
    UNANSWERED QUESTION that makes them want to know "why/how/who" — not just a mood or a job
    description. A flat statement like "Bạn là một thợ săn tiền thưởng trong một thành phố đầy
    tội phạm" is BAD (no mystery, just a job).
(b) sketch a glimpse of the WORLD itself — name or hint at a specific place, faction, myth,
    law, or phenomenon that makes this world feel strange/distinct (not just "a dangerous
    land"). The mystery should feel bigger than just the player's personal situation.
Stay intriguing and do NOT spoil any plot twist (there is no plot behind it yet — that gets
written later only for the hook the player picks).

VARY THE DEVICE across the {SEED_COUNT} hooks — do not lean on the same trick twice. In
particular, DO NOT use "nhận được một lá thư/bức thư" (receiving a letter/note) more than
ONCE across the set; prefer other devices such as:
- an impossible/contradictory fact about the player ("...nhưng không ai nhớ bạn từng ở đó")
- a physical mark/wound/object on the player they can't explain
- a place, law, or ritual specific to this world that the player is entangled in
- someone/something reacting to the player in a way that doesn't make sense yet
- a countdown or consequence already in motion ("...trước khi [cụ thể] xảy ra lần nữa")
- a betrayal or reveal already half-visible but not yet named
- the player waking into/being thrust into an already-strange location or event
Avoid generic fantasy flavor text with no hook (no plain "bạn là chiến binh/pháp sư trong một
vùng đất nguy hiểm" with nothing unresolved attached).

Model the SPECIFICITY, world-glimpse, and dangling-question quality of these examples (do not
reuse them verbatim, and do not copy their exact phrasing/structure either):
"Bạn là người duy nhất nhận ra danh tính của mình đã bị đánh cắp trong thành phố Vaelthorn, nơi
không ai còn nhớ bạn từng tồn tại.", "Bạn tỉnh dậy trong quan tài của chính mình giữa nghĩa
địa của một dòng tu đã bị giải tán 200 năm trước, với ngày mất khắc sẵn trên bia — chỉ thiếu
ngày, tháng.", "Ba người trước bạn từng đeo chiếc nhẫn của gia tộc Draumeir, và cả ba đều biến
mất đúng đêm trăng tròn thứ bảy sau khi nhận nó.", "Bạn là sứ giả duy nhất sống sót trở về từ
hội nghị hòa bình giữa ba vương triều, nơi mọi phái đoàn khác đều bị giết trong phòng khóa kín."

Output EXACTLY this shape:
{{"hooks": ["...", "...", "...", "...", "..."]}}"""


def generate_campaign_hooks(model: str = None, options: dict = None) -> list:
    """Gọi model 1 lần (think=False, ngân sách token nhỏ) để bịa nhanh
    SEED_COUNT câu hook. Không sinh phần còn lại — phần đó chỉ khai triển sau
    khi người chơi đã chọn (xem expand_campaign_hook). Lỗi/parse hỏng ->
    fallback về theme mặc định lặp lại (vẫn đủ SEED_COUNT phần tử để UI không
    vỡ)."""
    model = model or CAMPAIGN_MODEL
    options = options or HOOKS_OPTIONS

    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "system", "content": _build_hooks_prompt()}],
            format="json",
            options=options,
            think=False,
        )
        content = response["message"]["content"]
        parsed = _parse_campaign_json(content)
        hooks_raw = parsed.get("hooks") if isinstance(parsed, dict) else None
        if not isinstance(hooks_raw, list):
            print(
                f"[DEBUG] generate_campaign_hooks: content không parse được thành hooks "
                f"hợp lệ (content={content[:200]!r}) -> fallback {SEED_COUNT} default"
            )
            hooks_raw = []
    except Exception as e:
        print(f"[DEBUG] generate_campaign_hooks lỗi ({e}) -> fallback {SEED_COUNT} default")
        hooks_raw = []

    hooks = [str(h).strip() for h in hooks_raw[:SEED_COUNT] if str(h).strip()]
    while len(hooks) < SEED_COUNT:
        hooks.append(_DEFAULT_BIBLE["theme"])
    return hooks


# ---------------------------------------------------------------------------
# Campaign Bible — khai triển đầy đủ, dùng chung cho cả 2 nguồn (hook do AI
# gợi ý / ý tưởng người chơi tự viết)
# ---------------------------------------------------------------------------

def _build_campaign_bible_prompt(theme: str = None, user_idea: str = None) -> str:
    if theme:
        seed_instruction = f"""CHOSEN HOOK — the player already picked this exact sentence and has seen it; you MUST
put it back VERBATIM as "theme" in your output, do not reword it even slightly:
\"\"\"{theme}\"\"\""""
    else:
        seed_instruction = f"""PLAYER'S OWN IDEA — the player wrote this themselves (may be short/rough, may be in
Vietnamese). Distill it into ONE polished Vietnamese hook sentence for "theme" (SECOND PERSON,
"Bạn là...”/"Bạn bị...”, max ~35 words) that stays faithful to their core premise — do not
replace it with something unrelated, but you may tidy the wording:
\"\"\"{user_idea}\"\"\""""

    return f"""You are the LEAD NARRATIVE DESIGNER for a solo D&D 5e dark-fantasy RPG. You are not
writing a short story — you are authoring a "CAMPAIGN BIBLE": a single source-of-truth reference
document that a SMALLER, WEAKER AI (running live as the Dungeon Master, ~14B parameters, no
memory beyond what you give it plus the raw chat log) will rely on for the ENTIRE campaign,
roughly 50-80 turns.

That DM AI cannot re-derive anything you leave vague. Every field below exists to prevent a
SPECIFIC, OBSERVED failure of a small DM model — you are patching these failures directly:
- vague NPC motivation -> NPCs just stand around waiting for the player, never act on their own
- no ordered boss plan -> the boss never progresses, the threat feels frozen for 50 turns
- no story checkpoints -> the DM wanders in circles, forgets the main quest exists
- no revelation ordering -> the DM spoils the twist turn 2, or never reveals it at all
- no idle/failure consequence -> if the player stalls (wandering, fishing, avoiding the plot),
  nothing happens, there is no pressure to engage
- no explicit "never reveal" list -> the DM improvises and accidentally confirms player guesses
  about the twist before it's earned
- no named locations tied to NPCs -> the player wanders empty generic scenery and never runs
  into anyone

{seed_instruction}

## SETTING
D&D 5e MEDIEVAL DARK-FANTASY (swords, bows, magic, curses, gods, undead, fey, aberrations). You
are NOT limited to the Forgotten Realms — invent an original world/region/culture/myth that fits
the hook. NEVER introduce sci-fi/modern elements (no robots, holograms, simulations, computers,
lasers, drones, cyberpunk, "it's all a simulation"). If the hook implies illusion/false reality,
reframe it through MAGIC (a god's curse, a fey mirror-realm, a mad archmage's dream-prison, a
doppelganger plot) — never through technology.

Every field below except "theme" must be PLAIN ENGLISH ONLY — no Chinese or other language
mixed in, not even a single stray word.

BE CONCISE EVERYWHERE (important — this whole document gets re-sent to the DM in full on EVERY
single turn for 50-80 turns, so bloated fields cost real latency budget every turn of the whole
campaign): unless a field explicitly asks for more, write ONE short, punchy, information-dense
sentence — not two or three. Prefer concrete nouns over flowing prose. Cut filler words. A
description like "A gaunt scholar with ink-stained fingers, terrified of being overheard" beats
a longer, softer paragraph saying the same thing.

## WHAT YOU MUST DESIGN (build these together so they reinforce each other — e.g. the plot
twist should be foreshadowed by an NPC's secret, the boss's plan_stages should be WHY the
story_milestones escalate when they do)

1. PREMISE & TONE — the real situation behind the hook (secret, only the DM reads it) and the
   emotional register of the whole campaign (2-5 words, e.g. "grim, mournful, quietly dreadful").

2. SETTING DETAIL — name the specific world/region/city this takes place in, plus 2-4 sentences
   of concrete texture (geography, ruling power, culture, what makes it strange).

3. STARTING LOCATION & OPENING HOOK — the EXACT place the character is in when the campaign
   begins (this MUST match/follow directly from the hook's situation — e.g. if the hook says
   "you wake in a tomb," the starting location IS that tomb, not somewhere else). Give it a
   name and 2-3 sentences of concrete description. Then give opening_hook: ONE concrete,
   physical thing already present at this exact location — an object, a body, a mark, a sound,
   a locked door, a stranger arriving — that the character can immediately notice/investigate
   and which starts pulling them toward story_milestones[0]. This is what stops the DM from
   opening on a vague empty scene with nothing to do: there must be a specific first thread to
   pull on, sitting right in front of the character from turn 1.

4. STORY MILESTONES — 6 to 10 ORDERED checkpoints carrying the campaign from the opening_hook
   to finale across ~50-80 turns. Milestone 1 should be the natural next step after the
   character acts on opening_hook. Each milestone is a concrete STATE CHANGE in what the player
   has learned/done/unlocked (not a vague vibe) — order matters, milestone 3 must only make
   sense after milestone 2. This is the DM's map: whenever it loses track of "what's happening,"
   it re-reads this list and finds the next uncompleted one instead of wandering.

5. SECRETS + REVELATION ORDER — every secret fact behind the hook (the twist, NPC hidden
   identities, the villain's true plan, etc.), each tagged with WHEN it's safe to reveal:
   public = fine to hint anytime, mid_game/late_game = only after enough milestones are done,
   finale = only in the climax. This is what stops the DM from blurting the twist turn 2 OR
   sitting on it forever.

6. FACTIONS — 1-3 organized groups with their own goal and methods, independent of the player.
   State their relationship to the player and (if relevant) to each other. Factions give the DM
   something to move even when the player isn't looking.

7. NPCs — 3-5 named NPCs. For EACH you must give: gender ("male"/"female" — fixed ground truth
   for pronouns so the small DM model never has to guess/switch mid-campaign), what they WANT
   (motivation), who they TRUST, who they DISTRUST/fear, a SECRET they're hiding (can be small),
   and — critically — autonomous_behavior: literally what this NPC is doing turn-to-turn if the
   player never interacts with them (their own agenda in motion). This is what stops NPCs from
   just standing in place waiting to be talked to.

8. KEY LOCATIONS — 3-6 concrete NAMED places tied to the story (never generic "a forest"),
   INCLUDING the starting_location itself as one of them (reuse the exact same name). For each,
   list which NPCs can plausibly be found/encountered there. This is what stops the DM from
   wandering the player through empty scenery with no one to meet — it now has a map of WHERE
   to put people.

9. MONSTERS — 3-5 objects, thematically fitting this world (name/species/appearance/moveset/
   behavior). Vary species/silhouette — not all the same creature type.

10. BOSS — full flavor block (name/desc/appearance/moveset/behavior) PLUS:
    - ultimate_goal: what they actually want, one sentence.
    - plan_stages: 3-5 ORDERED concrete steps the boss is executing toward that goal RIGHT NOW,
      independent of the player (e.g. stage 1 "gathering a relic", stage 2 "performing a partial
      ritual", stage 3 "eliminating a witness", stage 4 "breaking the seal"). This is what stops
      "the boss wants to revive a god" from being a static fact that never moves — the DM can
      advance the boss through concrete stages as turns pass, making the threat feel alive.

11. FAILURE STATE — if the player stalls, avoids the main plot, or wastes many turns on
    unrelated activity, the world must NOT wait patiently. Give idle_turn_threshold (an int —
    how many turns of avoiding the main plot before the world acts on its own) and a CONCRETE
    description of what happens autonomously (the boss advances a plan_stage, an NPC is
    captured/killed, a location falls, a faction makes a move) — always escalating stakes, never
    a passive "nothing happens."

12. PACING GUIDANCE — one short paragraph mapping story_milestones roughly onto a 50-80 turn
    campaign (e.g. "milestones 1-2 in turns 1-15 (setup), 3-6 in turns 16-55 (investigation/
    escalation), 7-8 in turns 56-75 (confrontation), finale in the last ~5-10 turns").

13. DM DIRECTIVES — never_reveal: a short list of specific facts/phrasings the DM must never
    state outright before their tier is reached (your last line of defense against spoiling).
    npc_encounter_rule: one sentence instructing the DM to proactively place NPCs into scenes at
    their key_locations rather than passively waiting for the player to seek them out.

## SELF-CHECK — do this BEFORE writing your final answer. Think it through fully, find every
problem, and SILENTLY FIX each one. Only the corrected final JSON should appear in your output —
never narrate the check itself or mention having done it.
- TIMELINE: do story_milestones and boss.plan_stages follow ONE consistent causal order with no
  contradictions (nothing references an event that hasn't logically happened yet)?
- MOTIVATION: does every NPC's autonomous_behavior actually follow from their stated motivation/
  trusts/distrusts? Would a reader ask "wait, why would they do that"? If so, fix the motivation
  or the behavior so they align.
- REVELATION SAFETY: does anything placed in "public" tier accidentally give away a "finale"
  tier secret? If so, move it to a later tier or soften its wording.
- PACING: is there enough content across story_milestones + boss.plan_stages to sustain 50-80
  turns without feeling thin, and not so much that it can't resolve by the finale?
- COHERENCE: do the NPCs, factions, and boss all connect to ONE coherent plot (not disjointed
  set-pieces)? Does plot_twist actually recontextualize main_goal in a way the player could not
  have guessed from the theme alone?
- COMPLETENESS: does every npc "key" referenced in story_milestones.npc_involved and
  key_locations.npc_keys_present actually exist in the npcs list? Does every secret "id"
  referenced in revelation_order actually exist in secrets? Does starting_location.name exactly
  match one of the entries in key_locations? Does opening_hook logically lead toward
  story_milestones[0] (not some unrelated later milestone)?
- LANGUAGE: re-read EVERY string value you are about to output EXCEPT "theme". Is any of it in
  Vietnamese, Chinese, or any language other than English? Fix any that are — rewrite them in
  English. This is a common mistake, check it carefully field by field (premise, tone, setting,
  main_goal, plot_twist, moral_dilemma, every milestone title/description, every secret
  description, every faction field, every npc field, every location field, every monster field,
  boss fields, failure_state description, pacing_guidance, dm_directives) — "theme" is the ONLY
  field allowed to be Vietnamese.
If you find a problem, fix it now, silently, before writing the final JSON.

## FINAL LANGUAGE RULE (read this last, right before you write the JSON): "theme" = Vietnamese.
LITERALLY EVERY OTHER STRING in the JSON below = ENGLISH ONLY, no exceptions, no Vietnamese
words or phrases anywhere else, not even one. Double-check each field as you write it.

## OUTPUT — ONLY this JSON object, no markdown, no commentary, no explanation of your check:
{{
  "theme": "...",
  "premise": "...",
  "tone": "...",
  "setting": {{"name": "...", "description": "..."}},
  "starting_location": {{"name": "...", "description": "...", "opening_hook": "..."}},
  "main_goal": "...",
  "plot_twist": "...",
  "moral_dilemma": "...",
  "story_milestones": [
    {{"id": "m1", "title": "...", "description": "...", "npc_involved": "npc_key_or_null"}}
  ],
  "secrets": [
    {{"id": "s1", "description": "...", "reveal_tier": "public|mid_game|late_game|finale"}}
  ],
  "revelation_order": {{"public": ["s1"], "mid_game": [], "late_game": [], "finale": []}},
  "factions": [
    {{"id": "f1", "name": "...", "goal": "...", "methods": "...", "relationship_to_player": "..."}}
  ],
  "npcs": [
    {{"key": "snake_case_id", "name": "...", "gender": "male|female", "role": "ally|rival|neutral|antagonist",
      "desc": "...", "personality": "...", "appearance": "...", "motivation": "...", "trusts": "...",
      "distrusts": "...", "secret": "...", "autonomous_behavior": "..."}}
  ],
  "key_locations": [
    {{"name": "...", "description": "...", "npc_keys_present": ["snake_case_id"]}}
  ],
  "monsters": [
    {{"name": "...", "species": "...", "appearance": "...", "moveset": "...", "behavior": "..."}}
  ],
  "boss": {{
    "name": "...", "desc": "...", "appearance": "...", "moveset": "...", "behavior": "...",
    "ultimate_goal": "...",
    "plan_stages": [{{"stage": 1, "title": "...", "description": "..."}}]
  }},
  "failure_state": {{"idle_turn_threshold": 5, "description": "..."}},
  "pacing_guidance": "...",
  "dm_directives": {{"never_reveal": ["..."], "npc_encounter_rule": "..."}}
}}"""


def _generate_campaign_bible(theme: str = None, user_idea: str = None, model: str = None, options: dict = None) -> dict:
    model = model or CAMPAIGN_MODEL
    options = options or BIBLE_OPTIONS
    parsed = None

    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "system", "content": _build_campaign_bible_prompt(theme=theme, user_idea=user_idea)}],
            format="json",
            options=options,
            think=True,
        )
        content = response["message"]["content"]
        parsed = _parse_campaign_json(content)
        if parsed is None:
            thinking_len = len(response["message"].get("thinking") or "")
            print(
                f"[DEBUG] _generate_campaign_bible: content không parse được (content={content[:200]!r}, "
                f"thinking_len={thinking_len}) -> fallback default. Nếu content rỗng mà thinking_len lớn, "
                f"tăng num_predict trong BIBLE_OPTIONS (thinking/self-check đã ăn hết ngân sách token)."
            )
    except Exception as e:
        print(f"[DEBUG] _generate_campaign_bible lỗi ({e}) -> fallback default")

    bible = _normalize_bible(parsed)
    fixed_theme = (theme or "").strip()
    if fixed_theme:
        bible["theme"] = fixed_theme
    elif user_idea and (parsed is None or not isinstance(parsed, dict) or not parsed.get("theme")):
        bible["theme"] = user_idea.strip()
    return bible


def expand_campaign_hook(theme: str, model: str = None, options: dict = None) -> dict:
    """Khai triển 1 hook (đã chọn từ generate_campaign_hooks) thành đầy đủ
    Campaign Bible. Giữ nguyên theme gốc bất kể model trả về gì (không để
    model viết lại câu hook người chơi đã thấy và chọn)."""
    return _generate_campaign_bible(theme=(theme or "").strip(), model=model, options=options)


def expand_custom_seed(user_text: str, model: str = None, options: dict = None) -> dict:
    """Khai triển 1 ý tưởng người chơi tự viết thành đầy đủ Campaign Bible,
    bám sát ý tưởng gốc thay vì bịa lạc đề."""
    return _generate_campaign_bible(user_idea=(user_text or "").strip(), model=model, options=options)


# ---------------------------------------------------------------------------
# Lưu/đọc Campaign Bible ra 1 file JSON riêng (ngoài DB) — dễ mở lên xem/theo
# dõi lúc dev/debug thay vì phải đọc 1 cột TEXT lớn trong SQLite. Game chỉ có
# 1 save-slot (main.py: DELETE FROM character trước khi tạo mới) nên 1 file
# cố định là đủ, ghi đè mỗi lần tạo nhân vật mới.
# ---------------------------------------------------------------------------

def save_campaign_bible(bible: dict) -> str:
    os.makedirs(CAMPAIGN_SAVE_DIR, exist_ok=True)
    with open(CAMPAIGN_SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(bible, f, ensure_ascii=False, indent=2)
    return CAMPAIGN_SAVE_PATH


def load_campaign_bible() -> dict:
    if not os.path.exists(CAMPAIGN_SAVE_PATH):
        return None
    try:
        with open(CAMPAIGN_SAVE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


# ---------------------------------------------------------------------------
# Context cho DM (system prompt) — bị giấu khỏi UI, chỉ DM đọc
# ---------------------------------------------------------------------------

def monster_roster_names(bible: dict) -> list:
    """Danh sách tên quái trong roster campaign — dùng để nhắc lại ngắn gọn ở
    turn_note mỗi lượt /chat (xem main.py), vì rule đầy đủ trong CAMPAIGN
    BIBLE section nằm quá xa lúc model thực sự quyết định entity mới trong 1
    system prompt rất dài, dễ bị lãng quên."""
    if not bible:
        return []
    bible = _normalize_bible(bible)
    return [m["name"] for m in bible["monsters"] if m.get("name")]


def _milestone_tier_unlocked(milestone_index: int, total: int) -> set:
    """Tier nào đã 'mở khoá' để lộ cho DM, dựa theo milestone_index hiện tại
    (0-based) so với tổng số milestone — cùng ngưỡng ~1/3, ~2/3, cuối cùng mà
    prompt sinh Campaign Bible đã dùng để tự thiết kế revelation_order, nên
    nhất quán với ý đồ ban đầu của chính campaign đó."""
    total = max(1, total)
    unlocked = {"public"}
    if milestone_index >= max(1, total // 3):
        unlocked.add("mid_game")
    if milestone_index >= max(2, (total * 2) // 3):
        unlocked.add("late_game")
    if milestone_index >= total - 1:
        unlocked.add("finale")
    return unlocked


def format_campaign_context(bible: dict, turn_number: int = 0, milestone_index: int = 0) -> str:
    """CHỈ mớm milestone HIỆN TẠI cho DM (không phải toàn bộ danh sách) —
    milestone_index (0-based, do main.py lưu/tăng dần khi DM báo hoàn thành
    qua mechanics.milestone_complete) quyết định:
    - milestone nào đang active (chi tiết đầy đủ) vs đã qua (chỉ còn tiêu đề,
      cho continuity) vs CHƯA đọc tới (ẩn hoàn toàn — DM không được biết
      trước, tránh tự spoil/lên kế hoạch quá xa).
    - tier bí mật nào đã "mở khoá" (public/mid_game/late_game/finale) — bí
      mật tier chưa mở khoá bị ẩn hoàn toàn khỏi context, không chỉ dựa vào
      lời dặn "đừng tiết lộ" (an toàn hơn, cũng nhẹ context hơn).
    - boss đang ở stage kế hoạch nào — chỉ hiện stage hiện tại, không hiện
      trước các stage tương lai.
    turn_number=0 (scene mở đầu chưa diễn ra) -> gồm cả block TURN 1 MUST
    OPEN AT; turn_number>0 -> bỏ (đã qua rồi, tốn token vô ích)."""
    if not bible:
        return ""
    bible = _normalize_bible(bible)

    milestones = bible["story_milestones"]
    total_m = len(milestones)
    idx = max(0, min(milestone_index, total_m - 1))

    past_titles = "; ".join(m["title"] for m in milestones[:idx]) or "None yet"
    current = milestones[idx]
    current_line = (
        f"[{current['id']}] {current['title']} — {current['description']}"
        + (f" (involves NPC: {current['npc_involved']})" if current.get("npc_involved") else "")
    )
    is_finale = idx >= total_m - 1

    unlocked_tiers = _milestone_tier_unlocked(idx, total_m)
    secrets_by_id = {sec["id"]: sec for sec in bible["secrets"]}

    def _tier_lines(tier):
        if tier not in unlocked_tiers:
            return "(locked — not reached yet)"
        ids = bible["revelation_order"].get(tier) or []
        lines = [f"[{sid}] {secrets_by_id[sid]['description']}" for sid in ids if sid in secrets_by_id]
        return "; ".join(lines) or "None"

    faction_lines = "; ".join(
        f"{f['name']} ({f['id']}) — goal: {f['goal']} | methods: {f['methods']} | "
        f"toward player: {f['relationship_to_player']}"
        for f in bible["factions"]
    ) or "None"

    npc_lines = "\n".join(
        f"- {n['name']} ({n['key']}, {n['role']}{', ' + n['gender'] if n.get('gender') else ''}) — "
        f"{n['desc']} | personality: {n['personality']} | "
        f"appearance: {n['appearance']} | wants: {n['motivation']} | trusts: {n['trusts'] or '-'} | "
        f"distrusts: {n['distrusts'] or '-'} | secret: {n['secret'] or '-'} | "
        f"off-screen right now: {n['autonomous_behavior']}"
        for n in bible["npcs"]
    ) or "None"

    location_lines = "; ".join(
        f"{l['name']} — {l['description']}"
        + (f" [NPCs found here: {', '.join(l['npc_keys_present'])}]" if l.get("npc_keys_present") else "")
        for l in bible["key_locations"]
    ) or "None"

    monster_lines = "; ".join(
        f"{m['name']}"
        + (f" ({m['species']})" if m.get("species") else "")
        + (f" — appearance: {m['appearance']}" if m.get("appearance") else "")
        + (f" | moveset: {m['moveset']}" if m.get("moveset") else "")
        + (f" | behavior: {m['behavior']}" if m.get("behavior") else "")
        for m in bible["monsters"]
    ) or "None"

    boss = bible["boss"]
    boss_line = f"{boss['name']} — {boss['desc']}"
    if boss.get("appearance"):
        boss_line += f" | appearance: {boss['appearance']}"
    if boss.get("moveset"):
        boss_line += f" | moveset: {boss['moveset']}"
    if boss.get("behavior"):
        boss_line += f" | behavior: {boss['behavior']}"

    stages = boss.get("plan_stages") or []
    if stages:
        # Boss stage hiện tại tỉ lệ thuận với milestone hiện tại (không cần
        # 1 field self-report riêng cho boss — suy ra từ tiến độ milestone,
        # đủ chính xác cho mục đích "boss đang làm gì" mà không thêm state).
        stage_pos = min(len(stages) - 1, (idx * len(stages)) // max(1, total_m))
        current_stage = stages[stage_pos]
        stage_line = f"Stage {current_stage['stage']}: {current_stage['title']} — {current_stage['description']}"
        if stage_pos < len(stages) - 1:
            stage_line += f" ({len(stages) - 1 - stage_pos} further stage(s) beyond this, not yet revealed)"
    else:
        stage_line = "None"

    failure = bible["failure_state"]
    never_reveal = "; ".join(bible["dm_directives"].get("never_reveal") or []) or "None"
    npc_rule = bible["dm_directives"].get("npc_encounter_rule") or "place NPCs proactively at their key_locations."

    opening_block = (
        f"\nTURN 1 MUST OPEN AT: {bible['starting_location']['name']} — {bible['starting_location']['description']}\n"
        f"Weave this in immediately: {bible['starting_location']['opening_hook']}\n"
        if turn_number <= 0 else ""
    )
    twist_line = (
        bible['plot_twist'] if is_finale
        else "(locked — a deeper truth exists but is not yet yours to know; do not invent or hint at specifics beyond what current secrets/NPCs already imply)"
    )

    return f"""## CAMPAIGN BIBLE — source of truth, overrides generic improvisation. Never dump verbatim/spoil early; unfold through play.

PREMISE ({bible['tone'] or 'dark fantasy'}): {bible['premise']}
World: {bible['setting']['name']} — {bible['setting']['description']}
Main goal: {bible['main_goal']}
Twist (secret, only unlocked at finale, never state outright before then): {twist_line}
Moral dilemma: {bible['moral_dilemma']}
{opening_block}
STORY PROGRESS — completed so far (titles only): {past_titles}
CURRENT MILESTONE (aim every scene at completing this — do not skip ahead, do not reveal what comes after it):
{current_line}
{"This is the FINAL milestone — bring the story to its climax and resolution." if is_finale else "When this milestone's condition is clearly met by the story, set mechanics.milestone_complete=true so the next one unlocks."}

SECRETS (only tiers reached so far are unlocked; locked ones are fully hidden from you — if the player guesses one, deflect, don't confirm):
public: {_tier_lines('public')}
mid_game: {_tier_lines('mid_game')}
late_game: {_tier_lines('late_game')}
finale: {_tier_lines('finale')}
Never state outright before unlocked: {never_reveal}

FACTIONS (act on their own agenda even off-screen): {faction_lines}

NPCs ({npc_rule} Roleplay consistently; off-screen behavior should surface as rumors/consequences even when unseen.)
{npc_lines}

LOCATIONS (use to decide where scenes happen / who's encountered — don't invent disconnected generic scenery): {location_lines}

MONSTER ROSTER — MANDATORY source for any NEW hostile creature (reuse exact name/species/appearance/moveset/behavior); only invent outside it if truly nothing here fits: {monster_lines}

BOSS: {boss_line}
Goal: {boss.get('ultimate_goal', '')}
Current plan stage (surface indirectly via rumors/minions/omens until the arc is ready to climax): {stage_line}

FAILURE STATE — if player avoids main plot ~{failure.get('idle_turn_threshold', 5)}+ turns straight, world acts anyway (never passive): {failure.get('description', '')}"""
