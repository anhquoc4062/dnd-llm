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
        }


# ---------------------------------------------------------------------------
# Prompt cho lượt classify
# ---------------------------------------------------------------------------

def build_classify_prompt(char_dict: dict) -> str:
    strengths = ", ".join(
        (t.get("en") or t.get("name") or "") for t in char_dict.get("strengths", [])
    ) or "None"
    weaknesses = ", ".join(
        (t.get("en") or t.get("name") or "") for t in char_dict.get("weaknesses", [])
    ) or "None"
    skills_list = ", ".join(f"{s.get('vi', s['en'])} ({s['en']})" for s in char_dict["skills"]) or "None"
    items_list = ", ".join(f"{i.get('vi', i['en'])} ({i['en']})" for i in char_dict["items"]) or "None"
    attrs = char_dict["attrs"]

    return f"""You are a D&D 5e rules referee. Do NOT narrate. Output ONLY JSON.

CHARACTER SHEET
Attributes: {", ".join(f"{k.upper()}:{v}" for k, v in attrs.items())}
Strengths: {strengths}
Weaknesses: {weaknesses}
Skills: {skills_list}
Items: {items_list}

TASK: Read the player's action (may be freely typed in Vietnamese, or a suggested option
already in Vietnamese). The player will usually refer to skills/items by their Vietnamese
name (shown before the parentheses above) — match it against that name, but always output
item_or_skill_used using the ENGLISH name (in parentheses), since that's what the backend
sheet uses for matching. Determine advantage_state using this EXACT procedure, in order:
1. Does any WEAKNESS directly apply to this action? -> disadvantage, cite it.
2. Else does any STRENGTH directly apply? -> advantage, cite it.
3. Else is the relevant attribute notably high (16+) or low (8-) for this action?
   -> advantage/disadvantage citing the attribute.
4. Otherwise -> normal. Do not force advantage/disadvantage without a real match above.

DC CALIBRATION (pick realistically, never default to 12):
- 8-9 trivial | 10-12 easy | 13-15 moderate (real failure risk) | 16-18 hard
  (dangerous, against alert/skilled opposition) | 19-20 near-impossible
RULE: if advantage_state = "disadvantage", DC must be at least 14.
RULE: combat attacks against an alert hostile target default to DC 14-16.

Only cite ONE reason. Never invent a strength/weakness/skill not on the sheet.

Output:
{{"needs_roll": true or false based on player action, "attribute": "dex", "advantage_state": "normal", "advantage_reason": null, "dc": 12, "item_or_skill_used": "Item or Skill used, if not is null", "item_or_skill_owned": true}}

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
    return False


# ---------------------------------------------------------------------------
# Dice
# ---------------------------------------------------------------------------

def attr_modifier(score: int) -> int:
    return (score - 10) // 2


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
# API cấp cao — main.py chỉ cần gọi 2 hàm này
# ---------------------------------------------------------------------------

def classify_action(model: str, options: dict, user_input: str, char_dict: dict) -> dict:
    """Gọi model lần 1 để phân loại hành động. Trả về dict đã được làm sạch
    (reason bịa -> ép về normal), sẵn sàng cho resolve_action()."""
    classify_resp = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": build_classify_prompt(char_dict)},
            {"role": "user", "content": user_input},
        ],
        format="json",
        options=options,
        think=False,
    )
    classification = _parse_classify_json(classify_resp["message"]["content"])

    adv_state = classification.get("advantage_state", "normal")
    adv_reason = classification.get("advantage_reason")

    if not _reason_is_valid(adv_reason, char_dict):
        print(f"[DEBUG] classify bịa reason không có thật: {adv_reason} -> ép về normal")
        adv_state = "normal"
        adv_reason = None

    dc = _safe_int(classification.get("dc", 12), 12)
    if adv_state == "disadvantage" and dc < 14:
        dc = 14  # ép cứng rule DC tối thiểu, phòng model quên áp dụng

    classification["advantage_state"] = adv_state
    classification["advantage_reason"] = adv_reason
    classification["dc"] = dc

    print(f"[DEBUG] classify result: {classification}")
    return classification


def resolve_action(classification: dict, char_dict: dict) -> dict:
    """Kiểm tra tài nguyên thật (item/skill có sở hữu không, đủ mana không,
    còn cooldown không) rồi tung xúc xắc thật. KHÔNG ghi DB — chỉ trả về
    quyết định để main.py áp dụng (tiêu hao item/mana/cooldown) và build
    dice_fact cho prompt của lượt kể chuyện chính.

    Trả về:
    {
        "success": bool,
        "roll_type": "normal"|"advantage"|"disadvantage",
        "dice": dict|None,
        "dc": int,
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

    if forced_fail or not needs_roll:
        dice = None
        success = False if forced_fail else True
        roll_type = adv_state if needs_roll else "normal"
    else:
        modifier = attr_modifier(char_dict["attrs"].get(attribute, 10))
        dice = roll_d20(modifier, adv_state)
        success = dice["total"] >= dc
        roll_type = adv_state

    return {
        "success": success,
        "roll_type": roll_type,
        "dice": dice,
        "dc": dc,
        "attribute": attribute,
        "used_name": used_name,
        "consumed_kind": consumed_kind,
        "mana_cost": mana_cost,
        "resource_note": resource_note,
        "adv_reason": adv_reason,
    }