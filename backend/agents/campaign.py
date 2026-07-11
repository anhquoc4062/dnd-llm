"""
campaign.py — Sinh "Campaign Bible": tài liệu tham chiếu CỐ ĐỊNH (canon) mà AI
DM (model nhỏ, ~14B, không có trí nhớ nào ngoài những gì được nhét vào context
mỗi lượt) dựa vào để dẫn dắt toàn bộ ván chơi solo, thay vì bịa tùy hứng từng
turn rồi dần lạc đề/đi vòng vòng.

KIẾN TRÚC 2 TẦNG (Bible cố định + Milestone sinh dần — xem agents/milestone.py):
- Bible chỉ chứa những gì mà NẾU MẤT ĐI sẽ làm hỏng cốt truyện chính: thế
  giới/timeline, main antagonist, key NPCs, quái/vật phẩm quan trọng, 3 act
  (mỗi act có purpose/entry/exit condition), story_beats (các "điểm cốt truyện
  bắt buộc" phải xảy ra trong từng act — xem field "story_beats" bên dưới),
  các ending khả dĩ. Sinh 1 LẦN DUY NHẤT lúc tạo campaign, không đổi trong
  suốt ván chơi.
- Milestone (module milestone.py) sinh TỪNG CÁI MỘT, dựa trên Bible + story
  state thật (người chơi đã làm gì), disposable — NPC/quái/location của 1
  milestone có thể biến mất vĩnh viễn sau khi qua, không cần nhồi vào Bible.
  story_beats là mảnh ghép nối Bible với Milestone: Bible cố định CÁI GÌ phải
  xảy ra trong act nào, Milestone Designer tự do sáng tạo CÁCH NÀO để nó xảy
  ra (xem milestone._build_milestone_prompt) — vừa giữ campaign không lạc đề,
  vừa không cứng nhắc như kịch bản viết sẵn.

Người chơi chỉ thấy đúng 1 câu "theme" (hook, hiển thị lúc tạo nhân vật) —
toàn bộ phần còn lại bị giấu khỏi UI, chỉ được nhét vào system prompt cho DM
đọc (xem format_campaign_context). Campaign Bible được lưu ra 1 FILE JSON
riêng (backend/game-data/campaign_saves/current_campaign.json — xem save/load
bên dưới) thay vì chỉ nằm trong 1 cột TEXT của SQLite, để dễ mở lên xem/theo
dõi lúc dev/debug.

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
quán timeline, động cơ NPC, pacing 3 act) trong lúc "suy nghĩ" rồi tự sửa
TRƯỚC KHI xuất JSON cuối, thay vì chỉ viết 1 lần và hy vọng đúng — xem
_build_campaign_bible_prompt().

LƯU Ý: cả 2 hàm expand đều gọi ollama.chat() ĐỒNG BỘ (blocking) — nơi gọi
(main.py orchestrator lúc tạo nhân vật) PHẢI chạy qua run_in_executor, không
được await trực tiếp trong 1 coroutine, nếu không sẽ chặn cả event loop.
"""

import json
import os
import re

import ollama

from . import text_utils

CAMPAIGN_MODEL = "qwen3:14b"
SEED_COUNT = 5

# Chỉ sinh vài câu ngắn -> ngân sách token nhỏ hơn nhiều, và think=False vì
# không cần suy luận sâu cho việc bịa 1 câu hook.
HOOKS_OPTIONS = {"num_ctx": 4096, "num_predict": 1200, "temperature": 0.95}

# Campaign Bible giờ GỌN hơn bản cũ (bỏ story_milestones/secrets/revelation_
# order — những thứ đó chuyển sang milestone.py sinh dần), nhưng vẫn đủ lớn
# (world/characters/content/acts/endings) để cần ngân sách rộng rãi cho cả
# phần "thinking" (self-check, think=True) lẫn JSON cuối cùng.
BIBLE_OPTIONS = {"num_ctx": 8192, "num_predict": 4500, "temperature": 0.9}

# File lưu Campaign Bible ra ngoài (không chỉ nằm trong cột DB) — game này
# chỉ có 1 save-slot (xem main.py: DELETE FROM character trước khi tạo mới),
# nên dùng 1 đường dẫn cố định, ghi đè mỗi lần tạo nhân vật mới.
_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAMPAIGN_SAVE_DIR = os.path.join(_BACKEND_DIR, "game-data", "campaign_saves")
CAMPAIGN_SAVE_PATH = os.path.join(CAMPAIGN_SAVE_DIR, "current_campaign.json")


_DEFAULT_BIBLE = {
    "campaign": {
        "title": "The Hollow Reach",
        "theme": "Bạn tỉnh dậy giữa tàn tích của một buổi lễ đã thất bại, không nhớ nổi mình đã hứa điều gì với thứ đang chờ dưới lòng đất.",
        "genre": "dark fantasy mystery",
        "tone": "grim, paranoid, quietly desperate",
        "estimated_length": {"target_milestones": 10},
    },
    "world": {
        "overview": "A fog-bound trade town built atop a sealed ritual site, where the old faith never fully died.",
        "timeline": "The ritual was interrupted years ago; the cult has spent that time quietly rebuilding toward finishing it.",
        "major_locations": [
            {"name": "Hollow's Reach", "description": "A fog-bound trade town built atop a sealed ritual site."},
            {"name": "The Sealed Shrine", "description": "The half-collapsed ritual site itself, still watched by the cult."},
        ],
        "major_factions": [
            {"name": "The Hollow Circle", "goal": "Complete the interrupted ritual by any means.", "methods": "Infiltration, blackmail, ritual murder disguised as accidents.", "relationship_to_player": "Hostile once the character starts investigating."},
        ],
    },
    "story": {
        "main_goal": "Uncover what ritual was interrupted, and stop it from completing before it's too late.",
        "main_conflict": "The character was meant to be the ritual's vessel and must stop whoever orchestrated it before it finishes what it started.",
        "hidden_truths": [
            {"id": "t1", "description": "The ritual was meant to BIND an ancient entity, not summon it — the character's survival was the failure state, not an accident."},
            {"id": "t2", "description": "The orchestrator is someone the character has trusted since before the story began."},
        ],
        "main_plot": "The character survived a ritual meant to bind an ancient evil, and must uncover who orchestrated it before they finish the job.",
        "narrative_constraints": [
            "The orchestrator's identity must never be confirmed before Act 3.",
            "The character cannot permanently leave Hollow's Reach before the ritual threat is resolved.",
        ],
    },
    "characters": {
        "main_antagonist": {
            "name": "The Awakened One", "desc": "The ancient evil at the heart of the plot.",
            "appearance": "A towering, half-formed silhouette of shifting shadow and cracked bone.",
            "visual_prompt": "towering shadow entity, cracked bone fragments, shifting silhouette, ominous glow",
            "moveset": "Reality-warping strikes, summons minor shadow beasts, a devastating area pulse at low health.",
            "behavior": "Calm and taunting at first, grows increasingly violent and desperate as it weakens.",
            "ultimate_goal": "Fully manifest in the mortal world by completing the interrupted ritual.",
            "plan_stages": [
                {"stage": 1, "title": "Recover the lost vessel", "description": "Track down the character, the ritual's original intended vessel."},
                {"stage": 2, "title": "Reassemble the ritual circle", "description": "The cult repairs the damaged shrine and gathers scattered relics."},
                {"stage": 3, "title": "Silence witnesses", "description": "Eliminate or discredit anyone who could expose the ritual."},
                {"stage": 4, "title": "Complete the binding", "description": "Force the ritual to completion, with or without the character's consent."},
            ],
        },
        "key_npcs": [
            {
                "key": "unknown_informant", "name": "Unknown Informant", "gender": "male", "role": "ally",
                "motivation": "Wants the ritual stopped but is too afraid to act directly.",
                "personality": "Speaks only in half-truths and riddles, terrified of being overheard.",
                "relationships": "Distrusts anyone connected to the Hollow Circle, leans on the character out of shared danger.",
                "secrets": "Knows the orchestrator's identity but hasn't said it outright.",
                "appearance": "Gaunt, cloaked figure with ink-stained fingers and a nervous twitch.",
                "visual_prompt": "gaunt cloaked man, ink-stained fingers, nervous expression, dim candlelight",
            },
        ],
    },
    "content": {
        "major_monsters": [
            {
                "name": "Shadow Wraith", "species": "Aberration",
                "appearance": "A hound-sized mass of writhing black smoke with glowing red eyes.",
                "visual_prompt": "writhing black smoke creature, glowing red eyes, hound-sized, aberration",
                "moveset": "Lunging bite, brief invisibility, claw swipes that drain vigor.",
                "behavior": "Stalks silently before ambushing; retreats into darkness when badly hurt.",
            },
        ],
        "important_items": [
            {
                "name": "Scorched Ritual Token",
                "description": "Bears the character's own name — proof someone knew they would be here before they ever arrived.",
                "significance": "Key evidence tying the character to the ritual's true purpose.",
            },
        ],
    },
    "acts": [
        {"act": 1, "purpose": "Establish the mystery and the character's stake in it.", "entry_condition": "Campaign start.", "exit_condition": "The character learns the ritual was meant to bind something, not summon it."},
        {"act": 2, "purpose": "Investigate the cult and identify the orchestrator.", "entry_condition": "Act 1 exit condition met.", "exit_condition": "The character has undeniable proof of the orchestrator's identity."},
        {"act": 3, "purpose": "Confront the orchestrator and resolve the ritual threat.", "entry_condition": "Act 2 exit condition met.", "exit_condition": "The ritual is stopped or completed, and the moral dilemma is resolved."},
    ],
    "possible_endings": [
        {"name": "Public Exposure", "description": "The character exposes the orchestrator publicly — justice, but the town descends into chaos and witch-hunts.", "trigger_condition": "The character prioritized public truth over order throughout Act 3."},
        {"name": "Quiet Containment", "description": "The character handles it quietly — order preserved, but they become complicit in the cover-up.", "trigger_condition": "The character prioritized stability/secrecy throughout Act 3."},
    ],
    "story_beats": [
        {"id": "sb1", "act": 1, "description": "The character learns the ritual was meant to bind something, not summon it."},
        {"id": "sb2", "act": 1, "description": "The character finds physical proof they were the ritual's intended vessel."},
        {"id": "sb3", "act": 2, "description": "The character uncovers evidence pointing toward the orchestrator's identity."},
        {"id": "sb4", "act": 3, "description": "The character confronts the orchestrator with the truth."},
    ],
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


def _s(d, key, default=""):
    return str(d.get(key) or default).strip() if isinstance(d, dict) else default


def _parse_campaign_json(reply: str):
    try:
        clean = reply.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(clean)
    except json.JSONDecodeError:
        return None
    # Hậu kỳ ngay từ nguồn (Bible chỉ sinh 1 lần, sai là sai suốt campaign) —
    # xem agents/text_utils.py. Vô hại với field tiếng Anh, chỉ cắt được gì
    # có ký tự CJK thật.
    return text_utils.strip_cjk_deep(parsed)


def _normalize_bible(obj: dict) -> dict:
    """Đảm bảo đủ field/đúng kiểu cho Campaign Bible — schema lớn, model có
    thể bỏ sót/sai bất kỳ field nào; field nào thiếu/hỏng rơi về ĐÚNG phần
    tương ứng của _DEFAULT_BIBLE (không rơi về default nguyên khối), để 1 lỗi
    nhỏ ở 1 field không xoá sạch phần model đã làm đúng ở các field khác."""
    if not isinstance(obj, dict):
        obj = {}

    # --- campaign ---
    campaign_raw = obj.get("campaign")
    if isinstance(campaign_raw, dict) and campaign_raw.get("theme"):
        est_raw = campaign_raw.get("estimated_length")
        target_ms = _safe_int(
            est_raw.get("target_milestones") if isinstance(est_raw, dict) else est_raw,
            _DEFAULT_BIBLE["campaign"]["estimated_length"]["target_milestones"],
        )
        target_ms = max(6, min(target_ms, 18))  # chặn số vô lý, giữ trong khung ~1-2h chơi
        campaign_block = {
            "title": _s(campaign_raw, "title", _DEFAULT_BIBLE["campaign"]["title"]),
            "theme": _s(campaign_raw, "theme", _DEFAULT_BIBLE["campaign"]["theme"]),
            "genre": _s(campaign_raw, "genre", _DEFAULT_BIBLE["campaign"]["genre"]),
            "tone": _s(campaign_raw, "tone", _DEFAULT_BIBLE["campaign"]["tone"]),
            "estimated_length": {"target_milestones": target_ms},
        }
    else:
        campaign_block = json.loads(json.dumps(_DEFAULT_BIBLE["campaign"]))

    # --- world ---
    world_raw = obj.get("world")
    if isinstance(world_raw, dict) and world_raw.get("overview"):
        locs = []
        for l in (world_raw.get("major_locations") or []):
            if isinstance(l, dict) and l.get("name"):
                locs.append({"name": _s(l, "name"), "description": _s(l, "description")})
        if not locs:
            locs = [dict(l) for l in _DEFAULT_BIBLE["world"]["major_locations"]]

        facs = []
        for f in (world_raw.get("major_factions") or []):
            if isinstance(f, dict) and f.get("name"):
                facs.append({
                    "name": _s(f, "name"), "goal": _s(f, "goal"),
                    "methods": _s(f, "methods"), "relationship_to_player": _s(f, "relationship_to_player"),
                })
        if not facs:
            facs = [dict(f) for f in _DEFAULT_BIBLE["world"]["major_factions"]]

        world_block = {
            "overview": _s(world_raw, "overview"),
            "timeline": _s(world_raw, "timeline"),
            "major_locations": locs,
            "major_factions": facs,
        }
    else:
        world_block = json.loads(json.dumps(_DEFAULT_BIBLE["world"]))

    # --- story ---
    story_raw = obj.get("story")
    if isinstance(story_raw, dict) and story_raw.get("main_goal"):
        truths = []
        for i, t in enumerate(story_raw.get("hidden_truths") or []):
            if isinstance(t, dict) and t.get("description"):
                truths.append({"id": _slugify(t.get("id"), f"t{i + 1}"), "description": _s(t, "description")})
        if not truths:
            truths = [dict(t) for t in _DEFAULT_BIBLE["story"]["hidden_truths"]]

        constraints_raw = story_raw.get("narrative_constraints")
        constraints = [str(x).strip() for x in constraints_raw if str(x).strip()] if isinstance(constraints_raw, list) else []
        if not constraints:
            constraints = list(_DEFAULT_BIBLE["story"]["narrative_constraints"])

        story_block = {
            "main_goal": _s(story_raw, "main_goal"),
            "main_conflict": _s(story_raw, "main_conflict"),
            "hidden_truths": truths,
            "main_plot": _s(story_raw, "main_plot"),
            "narrative_constraints": constraints,
        }
    else:
        story_block = json.loads(json.dumps(_DEFAULT_BIBLE["story"]))

    # --- characters ---
    characters_raw = obj.get("characters")
    antagonist_raw = characters_raw.get("main_antagonist") if isinstance(characters_raw, dict) else None
    if isinstance(antagonist_raw, dict) and antagonist_raw.get("name"):
        stages = []
        for i, st in enumerate(antagonist_raw.get("plan_stages") or []):
            if isinstance(st, dict) and st.get("title"):
                stages.append({
                    "stage": _safe_int(st.get("stage"), i + 1) or (i + 1),
                    "title": _s(st, "title"), "description": _s(st, "description"),
                })
        if not stages:
            stages = [dict(st) for st in _DEFAULT_BIBLE["characters"]["main_antagonist"]["plan_stages"]]
        antagonist = {
            "name": _s(antagonist_raw, "name"), "desc": _s(antagonist_raw, "desc"),
            "appearance": _s(antagonist_raw, "appearance"), "visual_prompt": _s(antagonist_raw, "visual_prompt"),
            "moveset": _s(antagonist_raw, "moveset"), "behavior": _s(antagonist_raw, "behavior"),
            "ultimate_goal": _s(antagonist_raw, "ultimate_goal"), "plan_stages": stages,
        }
    else:
        antagonist = json.loads(json.dumps(_DEFAULT_BIBLE["characters"]["main_antagonist"]))

    npcs_raw = characters_raw.get("key_npcs") if isinstance(characters_raw, dict) else None
    npcs = []
    if isinstance(npcs_raw, list):
        for i, n in enumerate(npcs_raw):
            if isinstance(n, dict) and n.get("name"):
                gender_raw = _s(n, "gender").lower()
                npcs.append({
                    "key": _slugify(n.get("key") or n.get("name"), f"npc_{i + 1}"),
                    "name": _s(n, "name"),
                    "gender": gender_raw if gender_raw in ("male", "female") else "",
                    "role": _s(n, "role", "npc"),
                    "motivation": _s(n, "motivation"),
                    "personality": _s(n, "personality"),
                    "relationships": _s(n, "relationships"),
                    "secrets": _s(n, "secrets"),
                    "appearance": _s(n, "appearance"),
                    "visual_prompt": _s(n, "visual_prompt"),
                })
    if not npcs:
        npcs = [dict(n) for n in _DEFAULT_BIBLE["characters"]["key_npcs"]]

    characters_block = {"main_antagonist": antagonist, "key_npcs": npcs}

    # --- content ---
    content_raw = obj.get("content")
    monsters = []
    items = []
    if isinstance(content_raw, dict):
        for m in (content_raw.get("major_monsters") or []):
            if isinstance(m, dict) and m.get("name"):
                monsters.append({
                    "name": _s(m, "name"), "species": _s(m, "species"),
                    "appearance": _s(m, "appearance"), "visual_prompt": _s(m, "visual_prompt"),
                    "moveset": _s(m, "moveset"), "behavior": _s(m, "behavior"),
                })
        for it in (content_raw.get("important_items") or []):
            if isinstance(it, dict) and it.get("name"):
                items.append({"name": _s(it, "name"), "description": _s(it, "description"), "significance": _s(it, "significance")})
    if not monsters:
        monsters = [dict(m) for m in _DEFAULT_BIBLE["content"]["major_monsters"]]
    if not items:
        items = [dict(it) for it in _DEFAULT_BIBLE["content"]["important_items"]]
    content_block = {"major_monsters": monsters, "important_items": items}

    # --- acts (đúng 3, đánh số 1-3) ---
    acts_raw = obj.get("acts")
    acts = []
    if isinstance(acts_raw, list):
        for i, a in enumerate(acts_raw[:3]):
            if isinstance(a, dict) and a.get("purpose"):
                acts.append({
                    "act": i + 1, "purpose": _s(a, "purpose"),
                    "entry_condition": _s(a, "entry_condition"), "exit_condition": _s(a, "exit_condition"),
                })
    if len(acts) != 3:
        acts = [dict(a) for a in _DEFAULT_BIBLE["acts"]]

    # --- possible_endings ---
    endings_raw = obj.get("possible_endings")
    endings = []
    if isinstance(endings_raw, list):
        for e in endings_raw:
            if isinstance(e, dict) and e.get("name"):
                endings.append({"name": _s(e, "name"), "description": _s(e, "description"), "trigger_condition": _s(e, "trigger_condition")})
    if not endings:
        endings = [dict(e) for e in _DEFAULT_BIBLE["possible_endings"]]

    # --- story_beats (mandatory plot points per act — links Bible to Milestone
    # Generator: fixed WHAT, but the milestone designer stays free to invent
    # HOW each beat actually plays out) ---
    beats_raw = obj.get("story_beats")
    beats = []
    if isinstance(beats_raw, list):
        for i, bt in enumerate(beats_raw):
            if isinstance(bt, dict) and bt.get("description"):
                beats.append({
                    "id": _slugify(bt.get("id"), f"sb{i + 1}"),
                    "act": max(1, min(_safe_int(bt.get("act"), 1), len(acts))),
                    "description": _s(bt, "description"),
                })
    if not beats:
        beats = [dict(bt) for bt in _DEFAULT_BIBLE["story_beats"]]

    return {
        "campaign": campaign_block,
        "world": world_block,
        "story": story_block,
        "characters": characters_block,
        "content": content_block,
        "acts": acts,
        "possible_endings": endings,
        "story_beats": beats,
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
        hooks.append(_DEFAULT_BIBLE["campaign"]["theme"])
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

    return f"""You are the LEAD NARRATIVE DESIGNER for a solo D&D 5e dark-fantasy RPG. You are
authoring a "CAMPAIGN BIBLE": a CANON reference document — only things that would BREAK THE
MAIN PLOT if removed. Everything else (minor NPCs, throwaway monsters, specific locations for
a single scene) is generated LATER, per-milestone, by a separate system as the story unfolds —
you do NOT invent those here. Ask yourself for every field: "if this were deleted, does the
main plot break? If yes, it belongs here. If no, leave it out."

A SMALLER, WEAKER AI (running live as the Dungeon Master, ~14B parameters, no memory beyond
what it's given each turn) will rely on this Bible for the entire campaign. It cannot re-derive
anything you leave vague:
- vague antagonist plan -> the threat never progresses, feels frozen for the whole campaign
- no act structure with exit conditions -> the DM wanders in circles, forgets what act it's in
- no narrative constraints -> the DM improvises contradictions turn to turn
- no named factions/major NPCs -> nothing moves in the world except the player's own actions

{seed_instruction}

## SETTING
D&D 5e MEDIEVAL DARK-FANTASY (swords, bows, magic, curses, gods, undead, fey, aberrations). You
are NOT limited to the Forgotten Realms — invent an original world/region/culture/myth that fits
the hook. NEVER introduce sci-fi/modern elements (no robots, holograms, simulations, computers,
lasers, drones, cyberpunk, "it's all a simulation"). If the hook implies illusion/false reality,
reframe it through MAGIC (a god's curse, a fey mirror-realm, a mad archmage's dream-prison, a
doppelganger plot) — never through technology.

Every field below except "campaign.theme" must be PLAIN ENGLISH ONLY — no Chinese or other
language mixed in, not even a single stray word.

BE CONCISE EVERYWHERE (this whole document gets re-sent to the DM in full on EVERY turn of the
whole campaign, so bloated fields cost real latency budget every single turn): unless a field
explicitly asks for more, write ONE short, punchy, information-dense sentence — not two or
three. Prefer concrete nouns over flowing prose.

Every character/monster gets an "appearance" (English sentence, for narration) AND a
"visual_prompt" (English, short comma-separated visual traits only — species/build, clothing/
gear, notable features, mood/lighting — for an image generator, NOT shown to the player, no
need for full sentences).

## WHAT YOU MUST DESIGN

1. CAMPAIGN — title, theme (see instruction above), genre (2-4 words), tone (2-5 words, e.g.
   "grim, mournful, quietly dreadful"), estimated_length.target_milestones: an INTEGER 8-14 —
   your estimate of how many milestones (across all 3 acts) this story needs to reach a
   satisfying finale in roughly 1-2 hours of solo play. This is a hard budget a later system
   will pace against, so pick a number that actually matches the story's scope.

2. WORLD — overview (2-4 sentences: geography, ruling power, culture, what makes it strange),
   timeline (1-2 sentences of relevant backstory/history leading to now), major_locations
   (2-4 concrete NAMED places that matter to the PREMISE itself, not to any single scene —
   e.g. the capital, the cult's stronghold — never a place that only matters for one moment),
   major_factions (1-3 organized groups with their own goal/methods/relationship_to_player,
   independent of the player — they act off-screen even when unseen).

3. STORY — main_goal (what the character is ultimately trying to achieve), main_conflict (the
   central tension driving the whole campaign, one sentence), hidden_truths (2-4 SECRET facts
   behind the hook — the twist, a hidden identity, the villain's true plan — each with a short
   id and description; these are NEVER shown to the DM directly turn-to-turn, only surfaced
   later through milestones, so don't worry about revelation order here), main_plot (2-3
   sentences: the throughline connecting opening to finale), narrative_constraints (2-4 hard
   rules the DM must never violate, e.g. "X's identity must never be confirmed before Act 3",
   "the character cannot permanently leave [place] before Y").

4. CHARACTERS — main_antagonist: name/desc/appearance/visual_prompt/moveset/behavior PLUS
   ultimate_goal (one sentence) and plan_stages (3-5 ORDERED concrete steps they're executing
   toward that goal RIGHT NOW, independent of the player — this is what stops the threat from
   being a static fact that never moves). key_npcs: 2-4 named NPCs who matter to the MAIN plot
   across the whole campaign (not a single milestone) — for each: gender (fixed ground truth
   for pronouns), role (ally/rival/neutral/antagonist-adjacent), motivation, personality,
   relationships (who they trust/distrust and why), secrets (what they're hiding), appearance,
   visual_prompt.

5. CONTENT — major_monsters: 1-3 LEGENDARY/recurring-threat creatures tied to the main plot
   (not generic fodder — those get invented per-milestone later) — name/species/appearance/
   visual_prompt/moveset/behavior. important_items: 1-3 PLOT-CRITICAL items (a relic, a key,
   evidence) that persist across the whole campaign regardless of which milestone is active —
   name/description/significance (why it matters to the main plot).

6. ACTS — exactly 3 acts. For each: purpose (1-2 sentences: what this act is about),
   entry_condition (what must be true to begin it — Act 1 is always "Campaign start"),
   exit_condition (the CONCRETE state change that ends this act and starts the next — this is
   what a later per-milestone system checks against to know when to advance). Acts should
   escalate: Act 1 establishes the mystery/stakes, Act 2 is investigation/complication, Act 3
   is confrontation/resolution.

7. POSSIBLE_ENDINGS — 2-3 candidate endings for the finale, each with name, description (what
   happens), and trigger_condition (what player behavior/choices across the campaign would make
   THIS ending the fitting one — e.g. "player prioritized public justice over secrecy"). A later
   system picks whichever fits based on how the player actually played, so make the trigger
   conditions genuinely distinguishable from each other, not near-identical.

8. STORY_BEATS — 4-8 MANDATORY plot beats spread across the 3 acts (at least 1 per act, weighted
   toward whichever acts carry more narrative weight) — each a concrete, story-critical event that
   MUST occur at some point during its act for the main plot to hold together (e.g. "the character
   learns X", "the character obtains Y", "the character confronts Z"). A beat is NOT the same as
   its act's exit_condition (that's the single gate that ends the act) — a beat is one required
   story POINT inside the act, and an act can (and usually should) contain 2+ beats before its
   exit_condition is met. A later system (the MILESTONE DESIGNER, generating ONE milestone at a
   time as the story actually unfolds) will invent a DIFFERENT concrete scene/method to realize
   each beat depending on how the player plays — you are fixing WHAT must happen, never HOW.
   Each: id (short snake_case, e.g. "sb1"), act (integer 1-3, matching the acts array), description
   (ONE concrete sentence, plain English, phrased as something that happens/is learned/is obtained
   — never a vague theme like "tension rises").

## SELF-CHECK — do this BEFORE writing your final answer. Think it through fully, find every
problem, and SILENTLY FIX each one. Only the corrected final JSON should appear in your output —
never narrate the check itself or mention having done it.
- SCOPE DISCIPLINE: does everything you wrote actually matter to the MAIN plot across the whole
  campaign? If something feels like it only matters for one scene/moment, CUT it — it belongs
  in a later milestone, not here.
- COHERENCE: do world, characters, content, and acts all connect to ONE coherent plot? Does
  main_conflict actually follow from the hook?
- ACT PROGRESSION: does each act's exit_condition logically lead into the next act's purpose?
  Is Act 3's exit_condition an actual satisfying resolution, not a vague non-ending?
- ENDING DISTINCTNESS: are the possible_endings' trigger_conditions genuinely different axes
  (not two near-identical "good vs slightly-less-good" outcomes)?
- STORY BEATS: does every beat's "act" genuinely fit BEFORE that act's exit_condition (not after
  it, not contradicting it)? Is each beat concrete enough that a later milestone designer could
  recognize when it's been hit, without being so specific it forces one exact scene/method?
- BUDGET SANITY: is estimated_length.target_milestones large enough to cover 3 acts without
  feeling rushed, and small enough to resolve in ~1-2 hours (not 8+)?
- LANGUAGE: re-read EVERY string value you are about to output EXCEPT "campaign.theme". Is any
  of it in Vietnamese, Chinese, or any language other than English? Fix any that are.
If you find a problem, fix it now, silently, before writing the final JSON.

## FINAL LANGUAGE RULE (read this last, right before you write the JSON): "campaign.theme" =
Vietnamese. LITERALLY EVERY OTHER STRING in the JSON below = ENGLISH ONLY, no exceptions.

## OUTPUT — ONLY this JSON object, no markdown, no commentary, no explanation of your check:
{{
  "campaign": {{
    "title": "...", "theme": "...", "genre": "...", "tone": "...",
    "estimated_length": {{"target_milestones": 10}}
  }},
  "world": {{
    "overview": "...", "timeline": "...",
    "major_locations": [{{"name": "...", "description": "..."}}],
    "major_factions": [{{"name": "...", "goal": "...", "methods": "...", "relationship_to_player": "..."}}]
  }},
  "story": {{
    "main_goal": "...", "main_conflict": "...",
    "hidden_truths": [{{"id": "t1", "description": "..."}}],
    "main_plot": "...", "narrative_constraints": ["..."]
  }},
  "characters": {{
    "main_antagonist": {{
      "name": "...", "desc": "...", "appearance": "...", "visual_prompt": "...",
      "moveset": "...", "behavior": "...", "ultimate_goal": "...",
      "plan_stages": [{{"stage": 1, "title": "...", "description": "..."}}]
    }},
    "key_npcs": [
      {{"key": "snake_case_id", "name": "...", "gender": "male|female", "role": "ally|rival|neutral|antagonist",
        "motivation": "...", "personality": "...", "relationships": "...", "secrets": "...",
        "appearance": "...", "visual_prompt": "..."}}
    ]
  }},
  "content": {{
    "major_monsters": [{{"name": "...", "species": "...", "appearance": "...", "visual_prompt": "...", "moveset": "...", "behavior": "..."}}],
    "important_items": [{{"name": "...", "description": "...", "significance": "..."}}]
  }},
  "acts": [
    {{"act": 1, "purpose": "...", "entry_condition": "Campaign start.", "exit_condition": "..."}},
    {{"act": 2, "purpose": "...", "entry_condition": "...", "exit_condition": "..."}},
    {{"act": 3, "purpose": "...", "entry_condition": "...", "exit_condition": "..."}}
  ],
  "possible_endings": [
    {{"name": "...", "description": "...", "trigger_condition": "..."}}
  ],
  "story_beats": [
    {{"id": "sb1", "act": 1, "description": "..."}}
  ]
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
        bible["campaign"]["theme"] = fixed_theme
    elif user_idea and (parsed is None or not isinstance(parsed, dict) or not (parsed.get("campaign") or {}).get("theme")):
        bible["campaign"]["theme"] = user_idea.strip()
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

def monster_roster_names(bible: dict, current_milestone: dict = None) -> list:
    """Danh sách tên quái gợi ý — gộp major_monsters (Bible, cố định) với
    possible_encounters (milestone hiện tại, disposable) — dùng để nhắc lại
    ngắn gọn ở turn_note mỗi lượt /chat (xem dungeon_master.py), vì rule đầy
    đủ trong CAMPAIGN BIBLE section nằm quá xa lúc model thực sự quyết định
    entity mới trong 1 system prompt rất dài, dễ bị lãng quên."""
    names = []
    if bible:
        bible = _normalize_bible(bible)
        names.extend(m["name"] for m in bible["content"]["major_monsters"] if m.get("name"))
    if current_milestone:
        for e in (current_milestone.get("possible_encounters") or []):
            if isinstance(e, dict) and e.get("name"):
                names.append(e["name"])
    return names


def format_campaign_context(bible: dict, current_milestone: dict = None, act_index: int = 0, turn_number: int = 0) -> str:
    """Bible (cố định) + milestone HIỆN TẠI (đọc từ DB, sinh dần theo
    milestone.py — KHÔNG nằm trong bible nữa) ghép thành 1 block context cho
    DM. act_index (0-based, 0/1/2 = Act 1/2/3) quyết định act nào đang active
    + possible_endings chỉ lộ ra khi đã ở Act 3 (act_index==2), tránh DM tự
    kể ending quá sớm.
    turn_number=0 (scene mở đầu chưa diễn ra) -> không có ý nghĩa đặc biệt gì
    thêm ở đây nữa (opening_hook cũ đã chuyển thành milestone 1's objective/
    location — xem handle_start_game() ở dungeon_master.py), giữ tham số lại
    để tương thích chữ ký gọi cũ, không dùng bên trong."""
    if not bible:
        return ""
    bible = _normalize_bible(bible)

    c = bible["campaign"]
    w = bible["world"]
    st = bible["story"]
    ch = bible["characters"]
    content = bible["content"]
    acts = bible["acts"]
    endings = bible["possible_endings"]

    idx = max(0, min(act_index, len(acts) - 1))
    current_act = acts[idx]
    is_final_act = idx >= len(acts) - 1

    loc_lines = "; ".join(f"{l['name']} — {l['description']}" for l in w["major_locations"]) or "None"
    fac_lines = "; ".join(
        f"{f['name']} — goal: {f['goal']} | methods: {f['methods']} | toward player: {f['relationship_to_player']}"
        for f in w["major_factions"]
    ) or "None"

    npc_lines = "\n".join(
        f"- {n['name']} ({n['key']}, {n['role']}{', ' + n['gender'] if n.get('gender') else ''}) — "
        f"wants: {n['motivation']} | personality: {n['personality']} | relationships: {n['relationships']} | "
        f"secret: {n['secrets'] or '-'} | appearance: {n['appearance']}"
        for n in ch["key_npcs"]
    ) or "None"

    antagonist = ch["main_antagonist"]
    antagonist_line = f"{antagonist['name']} — {antagonist['desc']}"
    if antagonist.get("appearance"):
        antagonist_line += f" | appearance: {antagonist['appearance']}"
    if antagonist.get("moveset"):
        antagonist_line += f" | moveset: {antagonist['moveset']}"
    if antagonist.get("behavior"):
        antagonist_line += f" | behavior: {antagonist['behavior']}"
    stages = antagonist.get("plan_stages") or []
    stage_line = "None"
    if stages:
        # Stage hiện tại tỉ lệ thuận theo act đang ở — suy ra thay vì cần 1
        # field self-report riêng, đủ chính xác cho mục đích "đang làm gì".
        stage_pos = min(len(stages) - 1, (idx * len(stages)) // max(1, len(acts)))
        cs = stages[stage_pos]
        stage_line = f"Stage {cs['stage']}: {cs['title']} — {cs['description']}"
        if stage_pos < len(stages) - 1:
            stage_line += f" ({len(stages) - 1 - stage_pos} further stage(s) beyond this, not yet revealed)"

    monster_lines = "; ".join(
        f"{m['name']}" + (f" ({m['species']})" if m.get("species") else "")
        + (f" — appearance: {m['appearance']}" if m.get("appearance") else "")
        + (f" | moveset: {m['moveset']}" if m.get("moveset") else "")
        + (f" | behavior: {m['behavior']}" if m.get("behavior") else "")
        for m in content["major_monsters"]
    ) or "None"
    item_lines = "; ".join(f"{it['name']} — {it['description']}" for it in content["important_items"]) or "None"

    constraints = "; ".join(st["narrative_constraints"]) or "None"

    endings_block = ""
    if is_final_act and endings:
        endings_lines = "\n".join(f"- {e['name']}: {e['description']} (fits when: {e['trigger_condition']})" for e in endings)
        endings_block = f"\nPOSSIBLE ENDINGS (Act 3 — pick whichever fits how the story actually played out, weave toward it):\n{endings_lines}\n"

    # --- Milestone hiện tại (sinh dần, KHÔNG nằm trong bible) ---
    if current_milestone:
        m = current_milestone
        ms_npcs = "; ".join(
            f"{n.get('name')} ({n.get('role', 'npc')})" for n in (m.get("npcs") or []) if isinstance(n, dict) and n.get("name")
        ) or "None"
        ms_loc = m.get("location") or {}
        ms_loc_line = f"{ms_loc.get('name', '')} — {ms_loc.get('description', '')}" if ms_loc.get("name") else "Same as current scene"
        ms_encounters = "; ".join(
            f"{e.get('name')}" for e in (m.get("possible_encounters") or []) if isinstance(e, dict) and e.get("name")
        ) or "None"
        milestone_block = f"""
CURRENT MILESTONE (this is your map for right now — aim scenes at this, not the whole campaign):
{m.get('title', '')} — objective: {m.get('objective', '')}
Why this matters to the story: {m.get('story_purpose', '')}
Succeeds when: {m.get('success_condition', '')}
Fails when (a real, accumulated narrative state — NOT a single failed roll): {m.get('failure_condition', '')}
Must reveal to the player before this milestone ends: {m.get('required_reveal') or 'Nothing specific.'}
Must NOT reveal yet: {m.get('forbidden_reveal') or 'Nothing specific.'}
Suggested (not mandatory) NPCs for this stretch: {ms_npcs}
Suggested (not mandatory) location: {ms_loc_line}
Suggested (not mandatory) encounters: {ms_encounters}
When this milestone's success OR failure condition is clearly met by the story, set
mechanics.milestone_complete=true, mechanics.milestone_outcome_summary (English, one sentence:
what actually happened), and mechanics.act_complete (true only if this also satisfies the
CURRENT ACT's exit_condition below)."""
    else:
        milestone_block = """
CURRENT MILESTONE: (transitioning — the next one is still being prepared) use your own judgment
based on ACT purpose/exit_condition and the story so far; do not stall, keep the scene moving."""

    return f"""## CAMPAIGN BIBLE — canon, overrides generic improvisation. Never dump verbatim/spoil early; unfold through play.

CAMPAIGN: {c['title']} ({c['genre']}, tone: {c['tone']})
World: {w['overview']}
Timeline: {w['timeline']}
Major locations: {loc_lines}
Major factions (act on their own agenda even off-screen): {fac_lines}

Main goal: {st['main_goal']}
Main conflict: {st['main_conflict']}
Narrative constraints (never violate): {constraints}

KEY NPCs (proactively place them into scenes rather than passively waiting for the player to seek them out; roleplay consistently; off-screen behavior should surface as rumors/consequences even when unseen.)
{npc_lines}

MAIN ANTAGONIST: {antagonist_line}
Goal: {antagonist.get('ultimate_goal', '')}
Current plan stage (surface indirectly via rumors/minions/omens until ready to climax): {stage_line}

RECURRING MONSTERS (reuse exact name/species/appearance/moveset/behavior for these specifically; other minor creatures may be invented freely per scene): {monster_lines}

IMPORTANT ITEMS (persist across the whole campaign, not tied to any one milestone): {item_lines}

ACT {current_act['act']}/3 (current): {current_act['purpose']}
This act ends when: {current_act['exit_condition']}
{endings_block}{milestone_block}"""
