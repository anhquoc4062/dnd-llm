"""
social.py — Cách NPC nhìn nhận/đối xử với nhân vật dựa trên race/class.

Dữ liệu tĩnh (không đổi theo turn), load 1 lần từ lore_data/race_class_reactions.json.
Khác với lore.py (thay đổi theo region/session), block này là ĐẶC ĐIỂM CỐ ĐỊNH của
nhân vật nên được nhét vào system prompt (build_system_prompt), không phải nhét mỗi
turn như lore/entities/world_state.
"""

import json
import os

_DATA_PATH = os.path.join(os.path.dirname(__file__), "lore_data", "race_class_reactions.json")
_DATA_CACHE = None


def _load_data():
    global _DATA_CACHE
    if _DATA_CACHE is None:
        with open(_DATA_PATH, "r", encoding="utf-8") as f:
            _DATA_CACHE = json.load(f)
    return _DATA_CACHE


def get_race_reaction(race_en: str):
    data = _load_data()
    return data["race_reactions"].get(race_en) or data["default_reaction"]


def get_class_reaction(class_en: str):
    data = _load_data()
    return data["class_reactions"].get(class_en) or data["default_reaction"]


def format_social_context(race_en: str, class_en: str) -> str:
    """Block text nhét vào system prompt (1 lần, không đổi theo turn) mô tả
    cách NPC thường phản ứng với race/class của nhân vật — để DM lồng ghép
    tự nhiên vào thái độ NPC, không phải áp dụng máy móc mọi lúc."""
    race_info = get_race_reaction(race_en)
    class_info = get_class_reaction(class_en)

    return f"""## HOW NPCS PERCEIVE THIS CHARACTER (weave into NPC attitude/dialogue naturally — not every NPC needs to react to this, use as flavor when it fits)
Race ({race_en}): {race_info['reputation']}
  - Tends to be favorable with: {race_info['favorable_with']}
  - Tends to be unfavorable with: {race_info['unfavorable_with']}
Class ({class_en}): {class_info['reputation']}
  - Tends to be favorable with: {class_info['favorable_with']}
  - Tends to be unfavorable with: {class_info['unfavorable_with']}

RULES: This shapes ATTITUDE and dialogue tone only — never changes mechanics (no
free advantage/disadvantage from this alone; that still only comes from
attribute/strength/weakness per the ACTION VALIDATION rules). Don't force a reaction
onto every NPC; use it when an NPC would plausibly notice or care (e.g. a devout
temple-goer reacting to a Warlock, a guard eyeing a Tiefling warily). Never let this
override an NPC's established personality or a relationship already built in this
conversation — consistency with what's already happened always wins."""