import json
import sys

# Console Windows mặc định dùng cp1252, không encode được ký tự tiếng Việt
# (ế, ị, ư...) trong các print(f"[DEBUG] ...") rải khắp codebase -> crash cả
# request. Ép stdout/stderr sang UTF-8 ngay từ đầu để tránh treo server.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import db
from agents import entities, campaign, assistant
from agents import dungeon_master as dm

db.init_db()

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory="../"), name="static")


_CACHEABLE_STATIC_EXT = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".woff", ".woff2", ".otf", ".ttf")


@app.middleware("http")
async def no_cache_for_api(request: Request, call_next):
    """Chặn browser/proxy cache cho toàn bộ API JSON (GET /start_game,
    /character_info, POST /chat...). Đây là nguyên nhân phổ biến nhất khiến
    'choices bị cache' — GET request không có Cache-Control sẽ bị trình
    duyệt tự cache và trả lại y hệt response cũ dù server đã có state mới,
    đặc biệt rõ với /start_game vì URL không đổi giữa các lần gọi.

    /static cũng KHÔNG được loại trừ hoàn toàn nữa: .js/.css/.html sửa xong
    là phải thấy ngay (không cache), chỉ ảnh/font mới thật sự nên cache lâu
    (nội dung không đổi theo code)."""
    response = await call_next(request)
    is_long_lived_static = (
        request.url.path.startswith("/static")
        and request.url.path.lower().endswith(_CACHEABLE_STATIC_EXT)
    )
    if not is_long_lived_static:
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


# ---------------------------------------------------------------------------
# Static pages
# ---------------------------------------------------------------------------

@app.get("/")
async def get_index():
    return FileResponse("../index.html")


@app.get("/game")
async def get_game():
    return FileResponse("../screen/game/game.html")


# ---------------------------------------------------------------------------
# Campaign seed (khung truyện tổng — main goal/plot/npc/monster/boss cho DM
# bám theo xuyên suốt ván chơi). Người chơi chỉ thấy "theme"; phần còn lại bị
# giấu khỏi UI, chỉ gắn vào system prompt cho DM đọc (xem agents/campaign.py).
# ---------------------------------------------------------------------------

@app.get("/campaign_hooks")
async def campaign_hooks():
    """AI chỉ bịa nhanh 5 câu hook (theme), KHÔNG khai triển main_goal/plot/
    npcs/monsters/boss — phần đó tốn thời gian nên chỉ làm sau khi người chơi
    đã chọn 1 trong 5 (xem /campaign_seed/expand_hook)."""
    hooks = campaign.generate_campaign_hooks()
    return {"hooks": hooks}


@app.post("/campaign_seed/expand_hook")
async def campaign_seed_expand_hook(data: dict):
    """Người chơi đã chọn 1 trong 5 hook do AI gợi ý (bấm 'Xác nhận') -> khai
    triển ĐÚNG hook đó thành đầy đủ cấu trúc campaign seed."""
    theme = (data.get("theme") or "").strip()
    if not theme:
        return {"error": "Thiếu kịch bản đã chọn."}
    return campaign.expand_campaign_hook(theme)


@app.post("/campaign_seed/expand")
async def campaign_seed_expand(data: dict):
    """Người chơi tự viết 1 ý tưởng/theme ngắn -> khai triển thành đúng cấu
    trúc campaign seed (main_goal/plot/npcs/monsters/boss), bám sát ý tưởng
    gốc thay vì bịa lạc đề."""
    text = (data.get("text") or "").strip()
    if not text:
        return {"error": "Thiếu nội dung kịch bản."}
    return campaign.expand_custom_seed(text)


# ---------------------------------------------------------------------------
# Character creation
# ---------------------------------------------------------------------------

@app.post("/create_character")
async def create_character(data: dict):
    attrs = data.get("attrs", {}) or {}
    hp = db.safe_int(data.get("hp", 100), 100)
    mana = db.safe_int(data.get("mana", 50), 50)
    xp_target = db.safe_int(data.get("xpTarget", 10), 10)

    race = data.get("race", "")
    character_class = data.get("class", "")

    campaign_seed = data.get("campaignSeed")
    campaign_theme = None
    if isinstance(campaign_seed, dict):
        campaign_theme = campaign_seed.get("theme")
        # Campaign Bible được lưu ra file JSON riêng (backend/game-data/
        # campaign_saves/current_campaign.json) thay vì nhét nguyên khối vào
        # cột DB — dễ mở lên xem/theo dõi lúc dev/debug hơn hẳn so với đọc 1
        # cột TEXT lớn trong SQLite. Cột campaign_data trong DB chỉ còn giữ để
        # đọc ngược các save cũ tạo trước khi có cơ chế file này.
        campaign.save_campaign_bible(campaign_seed)

    conn = db.get_conn()
    c = conn.cursor()

    # Một save-slot duy nhất: xóa nhân vật & lịch sử cũ trước khi tạo mới
    c.execute("DELETE FROM character")
    c.execute("DELETE FROM history")
    conn.commit()

    # RAG: dọn sạch entity/loot của session cũ (single save-slot)
    entities.reset_session(conn)

    c.execute("""
        INSERT INTO character (
            name, gender, race, race_en, character_class, character_class_en,
            attr_str, attr_dex, attr_con, attr_int, attr_wis, attr_cha,
            hp, max_hp, mana, max_mana, level, xp, xp_target, gold,
            strengths, weaknesses, equipment, skills, items,
            campaign_theme, campaign_data
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        data.get("name", ""),
        data.get("gender", ""),
        race,
        data.get("raceEn") or race,
        character_class,
        data.get("classEn") or character_class,
        db.safe_int(attrs.get("str", 10), 10),
        db.safe_int(attrs.get("dex", 10), 10),
        db.safe_int(attrs.get("con", 10), 10),
        db.safe_int(attrs.get("int", 10), 10),
        db.safe_int(attrs.get("wis", 10), 10),
        db.safe_int(attrs.get("cha", 10), 10),
        hp, hp,        # hp = max_hp lúc mới tạo
        mana, mana,    # mana = max_mana lúc mới tạo
        1,             # level
        0,             # xp
        xp_target,
        0,             # gold
        db._list_json(data.get("strengths")),
        db._list_json(data.get("weaknesses")),
        db._list_json(data.get("equipment")),
        db._list_json(data.get("skills")),
        db._list_json(data.get("items")),
        campaign_theme,
        None,  # campaign_data: không còn ghi blob lớn vào DB, xem game-data/campaign_saves/
    ))

    conn.commit()
    conn.close()

    return {"status": "ok"}


@app.get("/character_info")
async def character_info():
    char = db.get_latest_character()
    if not char:
        return {}
    return db.character_row_to_dict(char)


@app.get("/game_state")
async def game_state():
    """Trả về TOÀN BỘ trạng thái để frontend dựng lại UI khi load/reload
    trang — khác với /start_game (chỉ trả lượt gần nhất để tiếp tục chơi).

    Dùng cái này khi cần vẽ lại scrollback đầy đủ; dùng /start_game khi chỉ
    cần "tiếp tục từ đây, tôi sẽ chọn hành động tiếp theo".
    """
    char = db.get_latest_character()
    if not char:
        return {"started": False, "history": [], "last_result": None, "character": None}

    region = char["region"] if "region" in char.keys() else None
    started = bool(region)

    conn = db.get_conn()
    c = conn.cursor()
    history_rows = c.execute(
        "SELECT id, role, content FROM history ORDER BY id ASC"
    ).fetchall()
    conn.close()

    history = []
    for i, row in enumerate(history_rows):
        content = row["content"]
        # Dòng user đầu tiên (nếu game đã start) là opening_instruction nội bộ
        # (tiếng Anh, hướng dẫn model — KHÔNG phải điều người chơi thực sự gõ).
        # Thay bằng nhãn thân thiện để frontend không hiển thị nhầm nó như
        # một tin nhắn của người chơi.
        if i == 0 and row["role"] == "user" and started:
            content = "[Bắt đầu cuộc phiêu lưu]"
        history.append({"role": row["role"], "content": content})

    last_result_raw = char["last_result"] if "last_result" in char.keys() else None
    last_result = None
    if last_result_raw:
        try:
            last_result = json.loads(last_result_raw)
        except (TypeError, json.JSONDecodeError):
            last_result = None

    return {
        "started": started,
        "region": region,
        "history": history,
        "last_result": last_result,
        "character": db.character_row_to_dict(char),
    }


# ---------------------------------------------------------------------------
# Trợ lý ngoài-truyện (assistant.py) / Gameplay (dm.py)
# ---------------------------------------------------------------------------

@app.post("/assistant_ask")
async def assistant_ask(data: dict):
    return await assistant.handle_ask(data)


@app.post("/chat")
async def chat(data: dict):
    return await dm.handle_chat(data)


@app.post("/chat/retry")
async def chat_retry():
    return await dm.handle_chat_retry()


@app.get("/start_game")
async def start_game():
    return await dm.handle_start_game()
