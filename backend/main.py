import asyncio
import json
import sys
from concurrent.futures import ThreadPoolExecutor

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
from agents import entities, campaign, milestone, assistant
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
# Campaign seed (Bible cố định + Milestone sinh dần — xem agents/campaign.py
# và agents/milestone.py). Người chơi chỉ thấy "theme"; phần còn lại bị giấu
# khỏi UI, chỉ gắn vào system prompt cho DM đọc.
# ---------------------------------------------------------------------------

@app.get("/campaign_hooks")
async def campaign_hooks():
    """AI chỉ bịa nhanh 5 câu hook (theme) để người chơi CHỌN — không khai
    triển gì thêm ở bước này nữa (không còn nút "Xác nhận kịch bản" riêng).
    Việc khai triển đầy đủ Campaign Bible + milestone đầu tiên xảy ra SAU khi
    bấm "Khởi tạo nhân vật", chạy nền qua run_campaign_setup() bên dưới."""
    hooks = campaign.generate_campaign_hooks()
    return {"hooks": hooks}


# ---------------------------------------------------------------------------
# Character creation — bấm 1 nút duy nhất, backend chạy NỀN qua 3 bước (sinh
# Bible -> sinh milestone 1 -> DM kể cảnh mở đầu + generate ảnh location song
# song) trong lúc frontend hiện modal tiến trình, poll /setup_status.
# ---------------------------------------------------------------------------

# ollama.chat() trong campaign.py/milestone.py là lời gọi ĐỒNG BỘ (blocking).
# run_campaign_setup() PHẢI chạy các bước đó qua executor riêng — nếu gọi
# trực tiếp trong task nền, sẽ chặn event loop, khiến /setup_status polling
# bị treo theo (modal đứng hình, phản tác dụng của việc chạy nền).
_SETUP_EXECUTOR = ThreadPoolExecutor(max_workers=1)


def _set_setup_stage(char_id: int, stage: str, error: str = None):
    conn = db.get_conn()
    conn.execute(
        "UPDATE character SET setup_stage = ?, setup_error = ? WHERE id = ?",
        (stage, error, char_id),
    )
    conn.commit()
    conn.close()


async def run_campaign_setup(char_id: int, theme: str, mode: str):
    """Chạy NỀN (asyncio.create_task, không await ở nơi gọi) ngay sau khi
    character row vừa được tạo. Cập nhật setup_stage từng bước để frontend
    poll thấy tiến độ thật."""
    loop = asyncio.get_running_loop()
    try:
        if mode == "custom":
            bible = await loop.run_in_executor(_SETUP_EXECUTOR, campaign.expand_custom_seed, theme)
        else:
            bible = await loop.run_in_executor(_SETUP_EXECUTOR, campaign.expand_campaign_hook, theme)
        campaign.save_campaign_bible(bible)
        _set_setup_stage(char_id, "milestone")

        target_total = bible["campaign"]["estimated_length"]["target_milestones"]
        first_ms = await loop.run_in_executor(
            _SETUP_EXECUTOR, milestone.generate_milestone, bible, "", 0, 1, target_total,
        )
        db.save_current_milestone(char_id, first_ms)
        _set_setup_stage(char_id, "opening")

        await dm.handle_start_game()
        _set_setup_stage(char_id, "ready")
    except Exception as e:
        print(f"[DEBUG] run_campaign_setup lỗi: {e}")
        _set_setup_stage(char_id, "error", str(e))


@app.get("/setup_status")
async def setup_status():
    char = db.get_latest_character()
    if not char:
        return {"stage": None, "error": None}
    return {
        "stage": char["setup_stage"] if "setup_stage" in char.keys() else None,
        "error": char["setup_error"] if "setup_error" in char.keys() else None,
    }


@app.post("/create_character")
async def create_character(data: dict):
    attrs = data.get("attrs", {}) or {}
    hp = db.safe_int(data.get("hp", 100), 100)
    mana = db.safe_int(data.get("mana", 50), 50)
    xp_target = db.safe_int(data.get("xpTarget", 10), 10)

    race = data.get("race", "")
    character_class = data.get("class", "")

    campaign_theme = (data.get("campaignTheme") or "").strip()
    campaign_mode = "custom" if data.get("campaignMode") == "custom" else "ai"
    if not campaign_theme:
        return {"error": "Thiếu kịch bản phiêu lưu."}

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
            campaign_theme, setup_stage
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
        "bible",
    ))
    char_id = c.lastrowid

    conn.commit()
    conn.close()

    # Bất đồng bộ hoàn toàn — trả response NGAY, frontend mở modal tiến trình
    # và tự poll /setup_status tới khi stage == "ready"/"error".
    asyncio.create_task(run_campaign_setup(char_id, campaign_theme, campaign_mode))

    return {"status": "started"}


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
        "scene_context": _scene_context_dict(char),
    }


def _scene_context_dict(char) -> dict:
    """Context panel hiện tại (location/quái/NPC mới nhất) — dùng chung giữa
    /game_state (resume sau reload) và /scene_context (poll ảnh sau khi
    imagegen.ensure_context_image chạy xong bất đồng bộ)."""
    return {
        "kind": char["context_kind"] if "context_kind" in char.keys() else None,
        "name": char["context_name"] if "context_name" in char.keys() else None,
        "description": char["context_desc"] if "context_desc" in char.keys() else None,
        "image_path": char["context_image_path"] if "context_image_path" in char.keys() else None,
    }


@app.get("/scene_context")
async def scene_context():
    char = db.get_latest_character()
    if not char:
        return {"kind": None, "name": None, "description": None, "image_path": None}
    return _scene_context_dict(char)


# ---------------------------------------------------------------------------
# Trợ lý ngoài-truyện (assistant.py) / Gameplay (dm.py)
# ---------------------------------------------------------------------------

@app.post("/assistant_ask")
async def assistant_ask(data: dict):
    return await assistant.handle_ask(data)


@app.post("/chat/classify")
async def chat_classify(data: dict):
    return await dm.handle_chat_classify(data)


@app.post("/chat/roll")
async def chat_roll():
    return await dm.handle_chat_roll()


@app.post("/chat/narrate")
async def chat_narrate():
    return await dm.handle_chat_narrate()


@app.post("/chat/retry")
async def chat_retry():
    return await dm.handle_chat_retry()


@app.get("/start_game")
async def start_game():
    return await dm.handle_start_game()
