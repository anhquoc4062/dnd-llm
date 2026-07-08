"""
campaign.py — Sinh "campaign seed": khung truyện tổng (main goal, plot, NPC,
loại quái, boss) để DM bám theo xuyên suốt ván chơi, thay vì bịa tùy hứng
từng turn.

Người chơi chỉ thấy đúng 1 câu "theme" (hiển thị trong dropdown lúc tạo nhân
vật) — toàn bộ phần còn lại (main_goal/plot/npcs/monsters/boss) bị giấu khỏi
UI, chỉ được nhét vào system prompt cho DM đọc (xem format_campaign_context).

Có 2 nguồn:
- generate_campaign_seeds(): AI tự bịa 5 seed, mỗi seed một "vị" khác nhau
  (thể loại/tông truyện khác nhau hẳn, không phải 5 biến thể na ná nhau).
- expand_custom_seed(text): người chơi tự viết 1 ý tưởng/theme ngắn, AI khai
  triển ra ĐÚNG cấu trúc như trên, bám sát ý tưởng gốc thay vì bịa lạc đề.

Cả 2 đều bật "think" (qwen3 hỗ trợ chế độ suy luận sâu hơn trước khi trả JSON
cuối) để kịch bản có chiều sâu/logic hơn — khác với các lệnh gọi khác trong
game (classify/summarize/narrate) vốn tắt think để lấy tốc độ.
"""

import json

import ollama

CAMPAIGN_MODEL = "qwen3:14b"
# num_predict phải đủ lớn để CHỨA CẢ phần "thinking" (think=True) lẫn JSON cuối
# cùng — với qwen3, token thinking bị trừ vào cùng ngân sách num_predict; nếu
# đặt quá thấp, model dùng hết ngân sách để suy luận rồi bị cắt ngang trước khi
# kịp in JSON, khiến content trả về rỗng dù không có lỗi nào được raise.
CAMPAIGN_OPTIONS = {"num_ctx": 8192, "num_predict": 6000, "temperature": 0.95}
EXPAND_OPTIONS = {"num_ctx": 8192, "num_predict": 3500, "temperature": 0.9}

SEED_COUNT = 5

_DEFAULT_CAMPAIGN = {
    "theme": "Bạn tỉnh dậy giữa tàn tích của một buổi lễ đã thất bại, không nhớ nổi mình đã hứa điều gì với thứ đang chờ dưới lòng đất.",
    "main_goal": "Uncover what ritual was interrupted, and stop it from completing before it's too late.",
    "plot": "The character wakes with no memory of the past night, near the wreckage of an occult "
            "ritual site. Following fragments of memory and physical clues, they realize they "
            "themselves were meant to be the ritual's vessel — the twist is that the one who orchestrated "
            "it was someone they trusted. Confronting this person forces a choice between exposing them "
            "publicly (justice, but chaos) or handling it quietly (order, but complicity).",
    "npcs": [
        {
            "name": "Unknown Informant", "role": "ally",
            "desc": "A cautious contact with fragments of the truth.",
            "personality": "Speaks only in half-truths and riddles, terrified of being overheard.",
        },
    ],
    "monsters": ["Cultist", "Shadow Beast", "Corrupted Guardian"],
    "boss": {"name": "The Awakened One", "desc": "The ancient evil at the heart of the plot."},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_campaign(obj: dict) -> dict:
    """Đảm bảo đủ field, kiểu dữ liệu đúng — model có thể bỏ sót/sai schema,
    không để lỗi đó lan ra tới lúc build system prompt."""
    if not isinstance(obj, dict):
        obj = {}

    theme = str(obj.get("theme") or _DEFAULT_CAMPAIGN["theme"]).strip()
    main_goal = str(obj.get("main_goal") or _DEFAULT_CAMPAIGN["main_goal"]).strip()
    plot = str(obj.get("plot") or _DEFAULT_CAMPAIGN["plot"]).strip()

    npcs_raw = obj.get("npcs")
    npcs = []
    if isinstance(npcs_raw, list):
        for n in npcs_raw:
            if isinstance(n, dict) and n.get("name"):
                npcs.append({
                    "name": str(n.get("name")).strip(),
                    "role": str(n.get("role") or "npc").strip(),
                    "desc": str(n.get("desc") or "").strip(),
                    "personality": str(n.get("personality") or "").strip(),
                })
    if not npcs:
        npcs = _DEFAULT_CAMPAIGN["npcs"]

    monsters_raw = obj.get("monsters")
    monsters = [str(m).strip() for m in monsters_raw if str(m).strip()] if isinstance(monsters_raw, list) else []
    if not monsters:
        monsters = _DEFAULT_CAMPAIGN["monsters"]

    boss_raw = obj.get("boss")
    if isinstance(boss_raw, dict) and boss_raw.get("name"):
        boss = {"name": str(boss_raw.get("name")).strip(), "desc": str(boss_raw.get("desc") or "").strip()}
    else:
        boss = _DEFAULT_CAMPAIGN["boss"]

    return {
        "theme": theme,
        "main_goal": main_goal,
        "plot": plot,
        "npcs": npcs,
        "monsters": monsters,
        "boss": boss,
    }


def _parse_campaign_json(reply: str):
    try:
        clean = reply.replace("```json", "").replace("```", "").strip()
        return json.loads(clean)
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# 5 seed do AI tự bịa
# ---------------------------------------------------------------------------

def _build_seeds_prompt() -> str:
    return f"""You are a D&D 5e campaign designer. Think carefully, then output ONLY a JSON
object (no markdown) with a single key "seeds": an array of EXACTLY {SEED_COUNT} campaign
seeds for a solo dark-fantasy Forgotten Realms campaign.

Each of the {SEED_COUNT} seeds MUST be a genuinely different FLAVOR/GENRE of story — not 5
variations of the same "ancient evil awakens" plot. Spread across distinct vibes such as
(pick {SEED_COUNT} different ones, your choice): political intrigue/betrayal, heist/theft,
revenge/personal vendetta, cosmic horror/eldritch dread, survival/wilderness disaster,
mystery/investigation, war/siege, cursed bloodline/tragedy, rescue/hostage, forbidden
knowledge/cult. No two seeds may share the same core vibe.

Every field except "theme" must be written in PLAIN ENGLISH ONLY — no Chinese, no other
language mixed in, even a single stray word or character.

Each seed is a JSON object with:

- "theme": ONE hook sentence in Vietnamese, SECOND PERSON ("Bạn là...", "Bạn bị...", "Bạn
  vừa...", etc.), max ~30 words. This is the ONLY field the player will ever see (shown in a
  selection dropdown) — it must simultaneously (a) tell the player WHAT ROLE/SITUATION they
  are in, and (b) hint at the shape of the world/setting, while staying intriguing and NOT
  spoiling the plot twist. Model your tone after these examples (do not reuse them verbatim):
  "Bạn là người duy nhất nhận ra danh tính của mình đã bị đánh cắp trong một thế giới nơi không
  ai còn nhớ bạn từng tồn tại.", "Bạn lạc vào một tấm gương ma thuật, nơi cả thế giới bị đảo
  ngược và luật lệ không còn như cũ.", "Bạn là một thành viên phi hành đoàn trên chuyến thám
  hiểm tới ngôi mộ cổ mà không ai từng quay về."
- "main_goal": the character's ultimate objective this campaign, in English, 1-2 sentences.
- "plot": the overall story arc/outline, in English, 4-6 sentences. MUST explicitly include
  all three of:
  1. a central MYSTERY the player has to investigate/uncover through play (a hidden truth,
     an unexplained event, a secret identity — something unresolved at the start),
  2. a PLOT TWIST behind that mystery (a betrayal, a hidden culprit, a reveal that recontextualizes
     the goal — something the player would not guess from the theme alone),
  3. at least one MORAL DILEMMA the character will face — a choice with no clean "correct"
     answer, where every option costs something (justice vs mercy, truth vs order, self vs
     others, etc.).
  This is secret, only the DM reads it, and must give enough shape to guide many turns.
- "npcs": 2-4 objects {{"name": "English Name", "role": "ally|rival|neutral|antagonist",
  "desc": "1 sentence in English — their stake in the plot", "personality": "1 sentence in
  English describing a DISTINCT, specific personality/voice/quirk (not generic) — how they
  actually talk and behave, so the DM can roleplay them consistently"}}.
- "monsters": 3-5 English monster/enemy type names thematically fitting this seed's genre
  (not generic — fit the vibe, e.g. a heist seed leans human guards/constructs/traps, a
  cosmic horror seed leans aberrations/cultists).
- "boss": {{"name": "English Name", "desc": "1-2 sentences in English describing the final
  confrontation and why they matter to the plot"}}.

Output EXACTLY this shape:
{{"seeds": [
  {{"theme": "...", "main_goal": "...", "plot": "...", "npcs": [...], "monsters": [...], "boss": {{...}}}},
  ... ({SEED_COUNT} total)
]}}"""


def generate_campaign_seeds(model: str = None, options: dict = None) -> list:
    """Gọi model 1 lần (think=True) để bịa ra SEED_COUNT campaign seed khác vị
    nhau. Lỗi/parse hỏng ở seed nào -> fallback default riêng seed đó (không
    làm hỏng cả mẻ)."""
    model = model or CAMPAIGN_MODEL
    options = options or CAMPAIGN_OPTIONS

    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "system", "content": _build_seeds_prompt()}],
            format="json",
            options=options,
            think=True,
        )
        content = response["message"]["content"]
        parsed = _parse_campaign_json(content)
        seeds_raw = parsed.get("seeds") if isinstance(parsed, dict) else None
        if not isinstance(seeds_raw, list):
            thinking_len = len(response["message"].get("thinking") or "")
            print(
                f"[DEBUG] generate_campaign_seeds: content không parse được thành seeds "
                f"hợp lệ (content={content[:200]!r}, thinking_len={thinking_len}) -> fallback "
                f"{SEED_COUNT} default. Nếu content rỗng mà thinking_len lớn, tăng num_predict "
                f"trong CAMPAIGN_OPTIONS (thinking đã ăn hết ngân sách token)."
            )
            seeds_raw = []
    except Exception as e:
        print(f"[DEBUG] generate_campaign_seeds lỗi ({e}) -> fallback {SEED_COUNT} default giống nhau")
        seeds_raw = []

    seeds = [_normalize_campaign(s) for s in seeds_raw[:SEED_COUNT]]
    while len(seeds) < SEED_COUNT:
        seeds.append(_normalize_campaign(None))
    return seeds


# ---------------------------------------------------------------------------
# Khai triển seed do người chơi tự viết
# ---------------------------------------------------------------------------

def _build_expand_prompt(user_text: str) -> str:
    return f"""You are a D&D 5e campaign designer. The player wrote their own campaign idea
below — think carefully, then expand it into the SAME structured JSON shape used for
AI-generated campaign seeds, staying faithful to the player's idea (do not replace it with
something unrelated; fill gaps creatively but keep their core premise intact).

Every field except "theme" must be written in PLAIN ENGLISH ONLY — no Chinese, no other
language mixed in, even a single stray word or character.

PLAYER'S IDEA (may be short/rough, may be in Vietnamese):
\"\"\"{user_text}\"\"\"

Output ONLY this JSON object (no markdown):
{{"theme": "ONE hook sentence in Vietnamese, SECOND PERSON ('Bạn là...', 'Bạn bị...', etc.),
max ~30 words, summarizing/tidying the player's idea — must convey the player's ROLE and a
glimpse of the world/setting, intriguing, not spoiling the twist. This is the ONLY field the
player sees again, so make it read well.",
"main_goal": "English, 1-2 sentences",
"plot": "English, 4-6 sentences. MUST include: (1) a central MYSTERY the player investigates
through play, (2) a PLOT TWIST behind it the player wouldn't guess from the theme alone, (3)
at least one MORAL DILEMMA with no clean right answer. Secret, only the DM reads this.",
"npcs": [{{"name": "English Name", "role": "ally|rival|neutral|antagonist", "desc": "1
sentence English — their stake in the plot", "personality": "1 sentence English, a distinct
specific personality/voice/quirk, not generic"}}],
"monsters": ["3-5 English monster/enemy type names fitting the idea's genre"],
"boss": {{"name": "English Name", "desc": "1-2 sentences English"}}}}"""


def expand_custom_seed(user_text: str, model: str = None, options: dict = None) -> dict:
    """Khai triển 1 câu/đoạn ý tưởng người chơi tự viết thành đủ cấu trúc
    campaign. Lỗi -> fallback default nhưng vẫn giữ nguyên theme = ý tưởng
    gốc của người chơi (không đánh mất input của họ ngay cả khi model hỏng)."""
    model = model or CAMPAIGN_MODEL
    options = options or EXPAND_OPTIONS
    user_text = (user_text or "").strip()

    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "system", "content": _build_expand_prompt(user_text)}],
            format="json",
            options=options,
            think=True,
        )
        content = response["message"]["content"]
        parsed = _parse_campaign_json(content)
        if parsed is None:
            thinking_len = len(response["message"].get("thinking") or "")
            print(
                f"[DEBUG] expand_custom_seed: content không parse được (content={content[:200]!r}, "
                f"thinking_len={thinking_len}) -> fallback default, giữ theme gốc người chơi"
            )
        campaign = _normalize_campaign(parsed)
        if parsed is None and user_text:
            campaign["theme"] = user_text
    except Exception as e:
        print(f"[DEBUG] expand_custom_seed lỗi ({e}) -> fallback default, giữ theme gốc người chơi")
        campaign = _normalize_campaign(None)
        if user_text:
            campaign["theme"] = user_text

    return campaign


# ---------------------------------------------------------------------------
# Context cho DM (system prompt) — bị giấu khỏi UI, chỉ DM đọc
# ---------------------------------------------------------------------------

def format_campaign_context(campaign: dict) -> str:
    if not campaign:
        return ""
    campaign = _normalize_campaign(campaign)

    npc_lines = "; ".join(
        f"{n['name']} ({n['role']}) — {n['desc']}"
        + (f" [personality: {n['personality']}]" if n.get("personality") else "")
        for n in campaign["npcs"]
    ) or "None"
    monster_line = ", ".join(campaign["monsters"]) or "None"
    boss = campaign["boss"]

    return f"""## CAMPAIGN — SECRET MASTER PLAN (never reveal these exact terms/twists directly;
unfold them naturally through play. This overrides generic scene-by-scene improvisation —
every scene should serve this arc.)
Main goal: {campaign['main_goal']}
Plot — includes a mystery to investigate, a twist, and a moral dilemma; reveal gradually
through clues/events, never dump as exposition: {campaign['plot']}
Key NPCs — roleplay each with their given personality CONSISTENTLY whenever they appear, not
as a generic voice: {npc_lines}
Preferred monster/enemy types for this campaign's encounters: {monster_line}
Final boss (only when the story arc is ready to climax): {boss['name']} — {boss['desc']}"""
