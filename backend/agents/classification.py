"""
classification.py — Tách riêng phần "phân loại hành động người chơi" (gọi model
lần 1 để xác định attribute/DC/advantage) và "tung xúc xắc + quyết định thành
bại" khỏi main.py.

Module này KHÔNG đụng DB — chỉ nhận char_dict (đã load sẵn) và trả về kết quả
thuần. Việc ghi DB (tiêu hao item, đặt cooldown skill, trừ mana...) vẫn do
main.py thực hiện dựa trên kết quả trả về ở đây, vì đó là tầng persistence,
không phải logic phân loại/roll.
"""

import random

import ollama

CLASSIFICATION_MODEL = "qwen3:14b"
CLASSIFICATION_OPTIONS = {"num_ctx": 4096, "num_predict": 200}


# ---------------------------------------------------------------------------
# Helpers nội bộ (bản sao gọn nhẹ, tránh import ngược từ main.py -> circular import)
# ---------------------------------------------------------------------------

def _safe_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _item_matches(item, name):
    """Giống hệt logic _item_matches trong main.py — cố ý duplicate (nhỏ, ổn
    định) thay vì import ngược để giữ module này độc lập, dễ test/tái sử dụng."""
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


def _parse_classify_json(reply: str) -> dict:
    import json
    try:
        clean_reply = reply.replace("```json", "").replace("```", "").strip()
        return json.loads(clean_reply)
    except json.JSONDecodeError:
        # Fallback an toàn: mặc định cần roll bình thường, không ưu ái/thiệt gì
        return {
            "needs_roll": True,
            "attribute": "wis",
            "advantage_state": "normal",
            "advantage_reason": None,
            "dc": 12,
            "item_or_skill_used": None,
            "item_or_skill_owned": True,
            "contest_type": "none",
            "target_key": None,
        }


# ---------------------------------------------------------------------------
# Prompt cho lượt classify
# ---------------------------------------------------------------------------

def build_classify_prompt(char_dict: dict, active_entities: list = None) -> str:
    strengths = ", ".join(
        (t.get("en") or t.get("name") or "") for t in char_dict.get("strengths", [])
    ) or "None"
    weaknesses = ", ".join(
        (t.get("en") or t.get("name") or "") for t in char_dict.get("weaknesses", [])
    ) or "None"
    skills_list = ", ".join(f"{s.get('vi', s['en'])} ({s['en']})" for s in char_dict["skills"]) or "None"
    items_list = ", ".join(f"{i.get('vi', i['en'])} ({i['en']})" for i in char_dict["items"]) or "None"
    attrs = char_dict["attrs"]
    entities_list = ", ".join(
        f"{e['key']}({e['name']}{'|hostile' if e.get('hostile') else ''})" for e in (active_entities or [])
    ) or "None"

    return f"""You are a D&D 5e rules referee. Do NOT narrate. Output ONLY JSON.

CHARACTER SHEET
Attributes: {", ".join(f"{k.upper()}:{v}" for k, v in attrs.items())}
Strengths: {strengths}
Weaknesses: {weaknesses}
Skills: {skills_list}
Items: {items_list}
Active entities in scene (key(name|hostile)): {entities_list}

TASK: Read the player's action (may be freely typed in Vietnamese, or a suggested option
already in Vietnamese). The player will usually refer to skills/items by their Vietnamese
name (shown before the parentheses above) — match it against that name, but always output
item_or_skill_used using the ENGLISH name (in parentheses), since that's what the backend
sheet uses for matching.

ATTRIBUTE SELECTION (pick the ONE that actually governs this action — do not default to a
habitual choice):
- str: melee force — swinging/thrusting a heavy weapon, breaking/lifting/grappling.
- dex: finesse/speed — ranged attacks, light-weapon precision strikes, dodging, stealth, acrobatics.
- con: enduring — resisting poison/fatigue/pain, holding on, surviving harsh conditions.
- int: knowledge — recalling lore, deciphering, investigating clues, tactical planning.
- wis: perception/instinct — noticing danger, reading intent, animal handling, willpower.
- cha: social force — persuading, deceiving, intimidating, performing.
A plain melee weapon attack (sword/axe/spear thrust or swing) defaults to str unless the
action explicitly emphasizes speed/precision/finesse (then dex). Never pick an attribute
just because it was used last turn.

Determine advantage_state using this EXACT procedure, in order:
1. Does any WEAKNESS directly apply to this action? -> disadvantage, cite it.
2. Else does any STRENGTH directly apply? -> advantage, cite it.
3. Else is the relevant attribute notably high (16+) or low (8-) for this action?
   -> advantage/disadvantage citing the attribute.
4. Otherwise -> normal. Do not force advantage/disadvantage without a real match above.

DC CALIBRATION (pick realistically, never default to 12) — this only matters when the
player's action does NOT match one of the DM's previously suggested choices (those already
carry their own DC, set by the DM with full scene context, and the backend reuses that instead
of your guess here):
- 8-9 trivial | 10-12 easy | 13-15 moderate (real failure risk) | 16-18 hard
  (dangerous, against alert/skilled opposition) | 19-20 near-impossible
RULE: if advantage_state = "disadvantage", DC must be at least 14.
RULE: combat attacks against an alert hostile target default to DC 14-16.

Only cite ONE reason. Never invent a strength/weakness/skill not on the sheet.

CONTEST TYPE (do NOT decide damage numbers yourself — the backend rolls real dice for these):
- "attack": the player is directly striking/shooting/casting offensive damage at ONE specific
  entity from the Active entities list above. Set target_key to that entity's exact key.
- "hazard": the player is trying to AVOID/RESIST/SURVIVE danger where failure means THEY take
  damage (dodging a trap, resisting a hit, surviving an ambush/environmental threat). If the
  danger is a specific HOSTILE entity from the Active entities list directly attacking/striking
  the player, set target_key to that entity's key (the backend will roll the monster's real
  attack for you — you only narrate hit/miss afterward). If the danger has no specific entity
  (a trap, falling, poison gas, generic environmental hazard), target_key is null.
- "none": no physical damage is at stake either way (social, exploration, pure flavor, non-combat
  skill use). target_key is always null.
Never pick "attack" without a valid target_key from the Active entities list; if no matching
entity exists, use "hazard" or "none" instead.

Output:
{{"needs_roll": true or false based on player action, "attribute": "dex", "advantage_state": "normal", "advantage_reason": null, "dc": 12, "item_or_skill_used": "Item or Skill used, if not is null", "item_or_skill_owned": true, "contest_type": "attack"|"hazard"|"none", "target_key": "exact key from Active entities list, or null"}}

advantage_reason format when not null: {{"type": "strength"|"weakness"|"class"|"race"|"item", "name": "exact name from sheet"}}"""


def _reason_is_valid(reason, char_dict):
    if not reason:
        return True
    name = (reason.get("name") or "").strip().lower()
    t = reason.get("type")
    if t == "attribute":
        return name in {k.lower() for k in char_dict["attrs"]}
    if t == "strength":
        return any((s.get("en") or s.get("name") or "").lower() == name for s in char_dict.get("strengths", []))
    if t == "weakness":
        return any((w.get("en") or w.get("name") or "").lower() == name for w in char_dict.get("weaknesses", []))
    if t == "skill":
        return any((s.get("en") or "").strip().lower() == name for s in char_dict.get("skills", []))
    if t == "item":
        return any((i.get("en") or "").strip().lower() == name for i in char_dict.get("items", []))
    if t == "race":
        return name == (char_dict.get("race_en") or "").strip().lower()
    if t == "class":
        return name == (char_dict.get("character_class_en") or "").strip().lower()
    return False


# ---------------------------------------------------------------------------
# Known choices (đã được DM phân loại sẵn ở lượt trước — tái sử dụng, không
# gọi lại LLM để quyết adv/dis/needs_roll cho 4 lựa chọn đã biết)
# ---------------------------------------------------------------------------

def _match_known_choice(user_input: str, known_choices: list):
    """So khớp user_input với 1 trong các choices mà chính DM đã trả về ở
    lượt trước (kèm sẵn roll/needs_roll/reason). Match tuyệt đối trước; nếu
    không khớp, fallback substring 2 chiều để vẫn bắt được trường hợp
    frontend/người chơi gõ hơi khác chữ so với text gốc."""
    if not known_choices:
        return None
    user_norm = (user_input or "").strip().lower()
    if not user_norm:
        return None

    for ch in known_choices:
        if not isinstance(ch, dict):
            continue
        text = (ch.get("text") or "").strip().lower()
        if text and text == user_norm:
            return ch

    for ch in known_choices:
        if not isinstance(ch, dict):
            continue
        text = (ch.get("text") or "").strip().lower()
        if len(text) >= 8 and len(user_norm) >= 8 and (text in user_norm or user_norm in text):
            return ch

    return None


# ---------------------------------------------------------------------------
# Dice
# ---------------------------------------------------------------------------

def attr_modifier(score: int) -> int:
    return (score - 10) // 2


def proficiency_bonus(level: int) -> int:
    """5e: +2 ở level 1-4, +3 ở 5-8, +4 ở 9-12... Dùng cho attack bonus,
    tách biệt khỏi modifier thuần của attribute check."""
    level = max(1, _safe_int(level, 1))
    return 2 + (level - 1) // 4


DEFAULT_MONSTER_AC = 12
DEFAULT_MONSTER_ATTACK_BONUS = 3
DEFAULT_MONSTER_DAMAGE_DICE = "1d6"


def flip_advantage(state: str) -> str:
    """Player disadvantage khi né/chống đỡ = quái CÓ LỢI hơn khi ra đòn (và
    ngược lại) — dùng khi roll xúc xắc chuyển từ góc nhìn player sang quái."""
    return {"advantage": "disadvantage", "disadvantage": "advantage"}.get(state, "normal")


def sanitize_damage_dice(notation: str) -> str:
    global _DICE_RE
    if _DICE_RE is None:
        import re
        _DICE_RE = re.compile(r"(\d+)d(\d+)([+-]\d+)?")
    if notation and _DICE_RE.match(notation.strip()):
        return notation.strip()
    return DEFAULT_MONSTER_DAMAGE_DICE


def roll_d20(modifier: int, advantage_state: str = "normal") -> dict:
    r1 = random.randint(1, 20)
    if advantage_state == "advantage":
        r2 = random.randint(1, 20)
        taken = max(r1, r2)
    elif advantage_state == "disadvantage":
        r2 = random.randint(1, 20)
        taken = min(r1, r2)
    else:
        r2 = None
        taken = r1
    return {
        "rolls": [r1] + ([r2] if r2 is not None else []),
        "taken": taken,
        "modifier": modifier,
        "total": taken + modifier,
    }


# ---------------------------------------------------------------------------
# Damage dice — code tự roll sát thương thật (5e-style), không để model tự
# bịa số HP mất/còn. Vũ khí/kỹ năng được suy ra loại dice qua khớp từ khoá
# trong tên (EN), không hỏi lại model để tránh model tự chọn dice có lợi cho nó.
# ---------------------------------------------------------------------------

DAMAGE_DICE_TABLE = [
    (("dagger", "knife"), "1d4"),
    (("bow", "arrow", "crossbow", "sling"), "1d8"),
    (("sword", "blade", "saber", "rapier", "scimitar", "longsword"), "1d8"),
    (("axe", "hammer", "mace", "club", "warhammer", "maul"), "1d8"),
    (("spear", "lance", "polearm", "halberd", "glaive"), "1d6"),
    (("staff", "wand"), "1d6"),
    (("fire", "flame", "burn", "inferno", "fireball"), "2d6"),
    (("ice", "frost", "lightning", "shock", "thunder", "arcane", "spell", "magic", "bolt"), "1d10"),
    (("fist", "punch", "kick", "unarmed", "claw", "bite"), "1d4"),
]
DEFAULT_MELEE_DICE = "1d6"
DEFAULT_SPELL_DICE = "1d8"

# DC tier -> dice sát thương phản đòn khi player THẤT BẠI trong 1 tình huống
# nguy hiểm (hazard: bẫy, phục kích, đòn phản công của kẻ địch...). DC càng
# cao (đối thủ/hiểm hoạ càng mạnh) thì phản đòn càng đau — không có stat block
# quái đầy đủ nên dùng DC làm proxy độ nguy hiểm, vẫn roll dice thật.
HAZARD_DICE_BY_DC = [
    (12, "1d4"),
    (15, "1d6"),
    (18, "2d6"),
    (999, "3d6"),
]


def lookup_damage_dice(used_name: str, attribute: str) -> str:
    if used_name:
        name_l = used_name.strip().lower()
        for keywords, dice in DAMAGE_DICE_TABLE:
            if any(k in name_l for k in keywords):
                return dice
    return DEFAULT_SPELL_DICE if attribute in ("int", "wis", "cha") else DEFAULT_MELEE_DICE


def hazard_dice_for_dc(dc: int) -> str:
    for threshold, dice in HAZARD_DICE_BY_DC:
        if dc <= threshold:
            return dice
    return "3d6"


_DICE_RE = None


def roll_dice(notation: str) -> dict:
    """Parse '<n>d<sides>[+/-k]' (vd '2d6+1') và roll thật bằng random."""
    global _DICE_RE
    if _DICE_RE is None:
        import re
        _DICE_RE = re.compile(r"(\d+)d(\d+)([+-]\d+)?")
    m = _DICE_RE.match((notation or "").strip())
    if not m:
        return {"notation": notation, "rolls": [], "bonus": 0, "total": 0}
    n, sides, bonus = int(m.group(1)), int(m.group(2)), int(m.group(3) or 0)
    n = max(1, min(n, 10))
    sides = max(2, min(sides, 20))
    rolls = [random.randint(1, sides) for _ in range(n)]
    return {"notation": notation, "rolls": rolls, "bonus": bonus, "total": sum(rolls) + bonus}


# ---------------------------------------------------------------------------
# API cấp cao — main.py chỉ cần gọi 2 hàm này
# ---------------------------------------------------------------------------

def classify_action(model: str, options: dict, user_input: str, char_dict: dict, known_choices: list = None, active_entities: list = None) -> dict:
    """Gọi model lần 1 để phân loại hành động. Trả về dict đã được làm sạch
    (reason bịa -> ép về normal), sẵn sàng cho resolve_action().

    known_choices (optional): 4 lựa chọn mà chính DM đã trả về ở lượt trước,
    mỗi lựa chọn đã kèm sẵn roll (advantage/disadvantage/normal), needs_roll,
    reason. Nếu user_input khớp với 1 trong số đó, TÁI SỬ DỤNG needs_roll/
    roll/reason đã có sẵn từ DM thay vì để classify tự phân tích lại từ đầu
    — tránh trường hợp choice hiển thị "disadvantage" nhưng lúc thực thi lại
    ra "advantage" do 2 lần gọi LLM độc lập không đồng thuận với nhau.
    dc/attribute/item_or_skill_used vẫn luôn được tính mới bình thường vì
    choices của DM không có các trường này."""
    classify_resp = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": build_classify_prompt(char_dict, active_entities)},
            {"role": "user", "content": user_input},
        ],
        format="json",
        options=options,
        think=False,
    )
    classification = _parse_classify_json(classify_resp["message"]["content"])

    # Validate contest_type/target_key thật kỹ — model hay bịa key không tồn
    # tại hoặc chọn "attack" mà không có mục tiêu hợp lệ. Code tự sửa lại
    # thay vì tin tưởng, vì đây là cơ sở để tính damage thật ở resolve_action.
    valid_entities = {(e.get("key") or "").strip().lower(): e for e in (active_entities or [])}
    contest_type = classification.get("contest_type")
    if contest_type not in ("attack", "hazard", "none"):
        contest_type = "none"
    target_key = (classification.get("target_key") or "").strip().lower() or None

    if contest_type == "attack":
        if target_key not in valid_entities:
            contest_type, target_key = ("hazard", None) if target_key is None else ("none", None)
    elif contest_type == "hazard":
        # target_key ở hazard chỉ có nghĩa khi trỏ tới 1 entity hostile đang
        # sống thật (quái đang chủ động tấn công) — nếu không, đây là hazard
        # chung chung (bẫy/môi trường) và target_key phải bỏ trống.
        entity = valid_entities.get(target_key)
        if not entity or not entity.get("hostile"):
            target_key = None
    else:
        target_key = None

    classification["contest_type"] = contest_type
    classification["target_key"] = target_key

    matched_choice = _match_known_choice(user_input, known_choices)

    def _use_known_choice(choice):
        """Trả về (adv_state, adv_reason) tái sử dụng từ known choice, hoặc
        None nếu reason của nó không còn đáng tin (viện dẫn skill/item không
        khớp với item_or_skill_used mà classify lượt này vừa xác định)."""
        adv_state = choice.get("roll") if choice.get("roll") in (
            "advantage", "disadvantage", "normal"
        ) else "normal"
        adv_reason = choice.get("reason")
        if not _reason_is_valid(adv_reason, char_dict):
            return "normal", None

        # DM chỉ VIẾT ra choice text lúc trước, không thực sự kiểm tra sở hữu/
        # cooldown/mana — item_or_skill_used thật sự chỉ được xác định MỚI ở
        # lượt classify này. Nếu reason của choice cũ viện dẫn skill/item mà
        # không khớp với những gì classify vừa xác định là được dùng, không
        # thể tin adv/dis đó nữa -> báo hiệu rơi về kết quả classify tự phân tích.
        if adv_reason and adv_reason.get("type") in ("skill", "item"):
            used_name = (classification.get("item_or_skill_used") or "").strip().lower()
            reason_name = (adv_reason.get("name") or "").strip().lower()
            if not used_name or used_name != reason_name:
                print(
                    f"[DEBUG] known choice viện dẫn {adv_reason.get('type')}='{reason_name}' nhưng "
                    f"classify lượt này xác định item_or_skill_used='{used_name or None}' -> không khớp, "
                    f"bỏ qua adv/dis từ choice cũ, dùng kết quả classify tự phân tích"
                )
                return None
        return adv_state, adv_reason

    reused = _use_known_choice(matched_choice) if matched_choice is not None else None

    if reused is not None:
        adv_state, adv_reason = reused
        classification["needs_roll"] = matched_choice.get("needs_roll", classification.get("needs_roll", True))
        print(
            f"[DEBUG] classify_action: input khớp choice đã biết ('{matched_choice.get('text')}') "
            f"-> dùng lại roll={adv_state}, needs_roll={classification['needs_roll']}, reason={adv_reason} từ DM thay vì tính lại"
        )
    else:
        adv_state = classification.get("advantage_state", "normal")
        adv_reason = classification.get("advantage_reason")
        if not _reason_is_valid(adv_reason, char_dict):
            print(f"[DEBUG] classify bịa reason không có thật: {adv_reason} -> ép về normal")
            adv_state = "normal"
            adv_reason = None

    # DC: nếu action khớp 1 trong 4 choice mà chính DM đã đưa ra ở lượt trước,
    # DM đã tự tính sẵn DC cho choice đó NGAY LÚC còn thấy toàn bộ ngữ cảnh
    # scene vừa viết ra -> đáng tin hơn hẳn DC mà classify tự đoán ở đây chỉ
    # từ 1 câu hành động rời rạc, không có ngữ cảnh. Chỉ rơi về DC tự tính khi
    # người chơi gõ tự do, không khớp với choice nào DM từng gợi ý.
    known_dc = None
    if matched_choice is not None and matched_choice.get("needs_roll"):
        known_dc = _safe_int(matched_choice.get("dc"), None)

    if reused is not None and known_dc is not None:
        dc = known_dc
        print(f"[DEBUG] classify_action: dùng lại dc={dc} do DM tự tính sẵn cho choice này thay vì tính lại")
    else:
        dc = _safe_int(classification.get("dc", 12), 12)

    if adv_state == "disadvantage" and dc < 14:
        dc = 14  # ép cứng rule DC tối thiểu, phòng model quên áp dụng (bất kể dc đến từ đâu)

    classification["advantage_state"] = adv_state
    classification["advantage_reason"] = adv_reason
    classification["dc"] = dc

    print(f"[DEBUG] classify result: {classification}")
    return classification


def resolve_action(classification: dict, char_dict: dict, active_entities: list = None) -> dict:
    """Kiểm tra tài nguyên thật (item/skill có sở hữu không, đủ mana không,
    còn cooldown không) rồi tung xúc xắc thật. KHÔNG ghi DB — chỉ trả về
    quyết định để main.py áp dụng (tiêu hao item/mana/cooldown) và build
    dice_fact cho prompt của lượt kể chuyện chính.

    ATTACK ROLL vs ABILITY CHECK/SAVE — tách biệt theo đúng luật 5e:
    - contest_type == "attack": d20 + (attribute modifier + proficiency bonus)
      so với AC THẬT của entity mục tiêu (lấy từ active_entities, không phải
      DC do model tự bịa).
    - contest_type == "hazard"/"none": d20 + attribute modifier so với DC
      (ability check / saving throw, giữ nguyên cơ chế cũ).

    Trả về:
    {
        "success": bool,
        "roll_type": "normal"|"advantage"|"disadvantage",
        "dice": dict|None,
        "dc": int,
        "target_ac": int|None,
        "attribute": str,
        "used_name": str|None,
        "consumed_kind": "item"|"skill"|None,
        "mana_cost": int,
        "resource_note": str,
        "adv_reason": dict|None,
    }
    """
    attribute = classification.get("attribute", "wis")
    needs_roll = classification.get("needs_roll", True)
    adv_state = classification.get("advantage_state", "normal")
    adv_reason = classification.get("advantage_reason")
    dc = classification.get("dc", 12)
    used_name = classification.get("item_or_skill_used")
    owned = classification.get("item_or_skill_owned", True)
    contest_type = classification.get("contest_type", "none")
    target_key = classification.get("target_key")

    resource_note = ""
    forced_fail = False
    consumed_kind = None  # "item" | "skill" | None
    mana_cost = 0

    if used_name:
        matched_item = next((i for i in char_dict["items"] if _item_matches(i, used_name)), None)
        matched_skill = next((s for s in char_dict["skills"] if _item_matches(s, used_name)), None)

        if matched_item:
            if matched_item.get("quantity", 1) <= 0:
                forced_fail = True
                resource_note = f"{used_name} is out of stock — the character fumbles reaching for nothing."
            else:
                consumed_kind = "item"
        elif matched_skill:
            mana_cost = _safe_int(matched_skill.get("manaCost", 0), 0)
            if matched_skill.get("cooldown_current", 0) > 0:
                forced_fail = True
                resource_note = (
                    f"{used_name} is still on cooldown "
                    f"({matched_skill.get('cooldown_current')} turn(s) left) — the character hesitates, it isn't ready."
                )
            elif char_dict["mana"] < mana_cost:
                forced_fail = True
                resource_note = (
                    f"{used_name} costs {mana_cost} mana but the character only has "
                    f"{char_dict['mana']} left — the power fizzles out before it can be unleashed."
                )
            else:
                consumed_kind = "skill"
        elif not owned:
            forced_fail = True
            resource_note = f"{used_name} does not exist on the character sheet."

    modifier = attr_modifier(char_dict["attrs"].get(attribute, 8))

    target_ac = None
    if contest_type == "attack" and target_key:
        target_entity = next(
            (e for e in (active_entities or [])
             if (e.get("key") or "").strip().lower() == target_key),
            None,
        )
        target_ac = _safe_int((target_entity or {}).get("ac"), DEFAULT_MONSTER_AC)

    # Nếu "hazard" mà target_key trỏ tới 1 entity hostile đang sống -> đây là
    # đòn tấn công CHỦ ĐỘNG của quái nhắm vào player. Code tự roll THAY quái
    # (d20 + attack_bonus của quái vs AC player) thay vì dùng DC do model chọn
    # — DM chỉ còn việc mô tả trúng/trượt, không còn quyết định kết quả.
    attacker_entity = None
    if contest_type == "hazard" and target_key:
        attacker_entity = next(
            (e for e in (active_entities or [])
             if (e.get("key") or "").strip().lower() == target_key and e.get("hostile")),
            None,
        )

    if forced_fail or not needs_roll:
        dice = None
        success = False if forced_fail else True
        roll_type = adv_state if needs_roll else "normal"
    elif target_ac is not None:
        # Attack roll thật (player tấn công): d20 + (modifier + proficiency)
        # vs AC của mục tiêu, không dùng DC — tách biệt hẳn khỏi ability check.
        attack_bonus = modifier + proficiency_bonus(char_dict.get("level", 1))
        dice = roll_d20(attack_bonus, adv_state)
        success = dice["total"] >= target_ac
        roll_type = adv_state
    elif attacker_entity is not None:
        # Attack roll thật (quái tấn công player): d20 + attack_bonus của quái
        # vs AC của player. success=True nghĩa là quái TRƯỢT (tốt cho player).
        monster_adv = flip_advantage(adv_state)
        monster_attack_bonus = _safe_int(attacker_entity.get("attack_bonus"), DEFAULT_MONSTER_ATTACK_BONUS)
        player_ac = _safe_int(char_dict.get("ac"), 10)
        dice = roll_d20(monster_attack_bonus, monster_adv)
        success = dice["total"] < player_ac
        roll_type = monster_adv
    else:
        dice = roll_d20(modifier, adv_state)
        success = dice["total"] >= dc
        roll_type = adv_state

    # --- Damage thật, roll bằng code — không để model tự bịa số HP mất/còn ---
    damage = None
    if not forced_fail:
        if contest_type == "attack" and success and target_key:
            dice_notation = lookup_damage_dice(used_name, attribute)
            dmg_roll = roll_dice(dice_notation)
            total = max(1, dmg_roll["total"] + modifier)
            damage = {"target": "entity", "target_key": target_key, "dice": dmg_roll,
                      "modifier": modifier, "total": total}
        elif contest_type == "hazard" and not success and attacker_entity is not None:
            dice_notation = sanitize_damage_dice(attacker_entity.get("damage_dice"))
            dmg_roll = roll_dice(dice_notation)
            total = max(1, dmg_roll["total"])
            damage = {"target": "player", "target_key": None, "dice": dmg_roll,
                      "modifier": 0, "total": total}
        elif contest_type == "hazard" and not success:
            dice_notation = hazard_dice_for_dc(dc)
            dmg_roll = roll_dice(dice_notation)
            total = max(1, dmg_roll["total"])
            damage = {"target": "player", "target_key": None, "dice": dmg_roll,
                      "modifier": 0, "total": total}

    return {
        "success": success,
        "roll_type": roll_type,
        "dice": dice,
        "dc": dc,
        "target_ac": target_ac,
        "attribute": attribute,
        "used_name": used_name,
        "consumed_kind": consumed_kind,
        "mana_cost": mana_cost,
        "resource_note": resource_note,
        "adv_reason": adv_reason,
        "contest_type": contest_type,
        "target_key": target_key,
        "damage": damage,
        "forced_fail": forced_fail,
    }