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
from agents import entities, campaign, imagegen, milestone, assistant
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

@app.get("/campaign_seeds")
async def campaign_seeds():
    """AI chỉ bịa nhanh 5 câu campaign seed (theme) để người chơi CHỌN —
    không khai triển gì thêm ở bước này nữa (không còn nút "Xác nhận kịch
    bản" riêng). Việc khai triển đầy đủ Campaign Bible + milestone đầu tiên
    xảy ra SAU khi bấm "Khởi tạo nhân vật", chạy nền qua run_campaign_setup()
    bên dưới."""
    seeds = campaign.generate_campaign_seeds()
    return {"seeds": seeds}


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
    db.update_campaign_state(char_id, setup_stage=stage, setup_error=error)


async def run_campaign_setup(char_id: int, theme: str, mode: str, reuse_bible: bool = False):
    """Chạy NỀN (asyncio.create_task, không await ở nơi gọi) ngay sau khi
    character row vừa được tạo. Cập nhật setup_stage từng bước để frontend
    poll thấy tiến độ thật.

    reuse_bible=True: bỏ qua bước sinh Bible (dùng lại bible đã lưu trên đĩa)
    -- dùng khi "Chơi lại campaign với nhân vật mới" (giữ nguyên thế giới/cốt
    truyện, chỉ đổi nhân vật)."""
    loop = asyncio.get_running_loop()
    try:
        if reuse_bible:
            bible = campaign.load_campaign_bible()
            if bible is None:
                raise RuntimeError("Không tìm thấy campaign bible để tái sử dụng.")
        else:
            if mode == "custom":
                bible = await loop.run_in_executor(_SETUP_EXECUTOR, campaign.expand_custom_seed, theme)
            else:
                bible = await loop.run_in_executor(_SETUP_EXECUTOR, campaign.expand_campaign_seed, theme)
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


async def _restart_story_same_character(char_id: int):
    """Nền cho /replay_campaign mode=same_character: giữ nguyên bible VÀ nhân
    vật, chỉ sinh lại milestone 1 + cảnh mở đầu (giống run_campaign_setup
    nhưng bỏ qua cả bước sinh Bible lẫn tạo nhân vật)."""
    loop = asyncio.get_running_loop()
    try:
        bible = campaign.load_campaign_bible()
        if bible is None:
            raise RuntimeError("Không tìm thấy campaign bible.")
        target_total = bible["campaign"]["estimated_length"]["target_milestones"]
        first_ms = await loop.run_in_executor(
            _SETUP_EXECUTOR, milestone.generate_milestone, bible, "", 0, 1, target_total,
        )
        db.save_current_milestone(char_id, first_ms)
        _set_setup_stage(char_id, "opening")

        await dm.handle_start_game()
        _set_setup_stage(char_id, "ready")
    except Exception as e:
        print(f"[DEBUG] _restart_story_same_character lỗi: {e}")
        _set_setup_stage(char_id, "error", str(e))


# setup_stage chỉ là tên bước (đổi tuần tự bible -> milestone -> opening ->
# ready), không tự nói lên "còn bao xa" -> gán % ước lượng theo tỷ trọng thời
# gian THẬT của từng bước (đo thực tế: bible think=True + schema lớn nhất nên
# chiếm phần lớn, milestone think=True nhưng schema nhỏ hơn hẳn, opening
# think=False nên nhanh nhất) để frontend vẽ được 1 thanh loading % thay vì
# chỉ list tên bước. % gán cho 1 stage = mốc NGAY KHI bước đó bắt đầu chạy
# (chưa phải % của việc bước đó đã xong), nên percent chỉ nhảy bậc mỗi khi
# setup_stage đổi, không mượt liên tục trong 1 bước — chấp nhận được vì
# không có tín hiệu tiến độ thật bên trong 1 lần gọi ollama.chat().
_SETUP_STAGE_PERCENT = {
    None: 0,
    "bible": 10,
    "milestone": 60,
    "opening": 85,
    "ready": 100,
    "error": 0,
}


@app.get("/setup_status")
async def setup_status():
    char = db.get_latest_character()
    if not char:
        return {"stage": None, "error": None, "percent": 0}
    state = db.get_campaign_state(char["id"])
    stage = state["setup_stage"] if state else None
    return {
        "stage": stage,
        "error": state["setup_error"] if state else None,
        "percent": _SETUP_STAGE_PERCENT.get(stage, 0),
    }


@app.post("/create_character")
async def create_character(data: dict):
    attrs = data.get("attrs", {}) or {}
    hp = db.safe_int(data.get("hp", 100), 100)
    mana = db.safe_int(data.get("mana", 50), 50)
    xp_target = db.safe_int(data.get("xpTarget", 10), 10)

    race = data.get("race", "")
    character_class = data.get("class", "")

    reuse_bible = bool(data.get("reuseBible"))
    campaign_theme = (data.get("campaignTheme") or "").strip()
    campaign_mode = "custom" if data.get("campaignMode") == "custom" else "ai"
    if reuse_bible:
        existing_bible = campaign.load_campaign_bible()
        if existing_bible is None:
            return {"error": "Không tìm thấy campaign bible để tái sử dụng."}
        campaign_theme = existing_bible.get("campaign", {}).get("theme") or campaign_theme
    elif not campaign_theme:
        return {"error": "Thiếu kịch bản phiêu lưu."}

    conn = db.get_conn()
    c = conn.cursor()

    # Một save-slot duy nhất: xóa nhân vật & lịch sử cũ trước khi tạo mới.
    # campaign_state cũng phải xóa — trước đây bị bỏ sót nên row của các nhân
    # vật đã xóa còn tồn đọng (rác), gây khó debug và sai khi truy vấn không
    # lọc character_id.
    c.execute("DELETE FROM character")
    c.execute("DELETE FROM history")
    c.execute("DELETE FROM campaign_state")
    conn.commit()

    # RAG: dọn sạch entity/loot của session cũ (single save-slot)
    entities.reset_session(conn)

    # Dọn sạch ảnh generate của campaign cũ — cùng lý do (single save-slot),
    # tránh ảnh cache theo tên bị tái dùng nhầm cho campaign mới. Bỏ qua khi
    # reuse_bible=True: cùng 1 thế giới/roster nên ảnh cũ (location/monster/npc
    # trùng tên) vẫn dùng lại được, khỏi generate lại tốn thời gian.
    if not reuse_bible:
        imagegen.clear_generated_images()

    c.execute("""
        INSERT INTO character (
            name, gender, race, race_en, character_class, character_class_en,
            attr_str, attr_dex, attr_con, attr_int, attr_wis, attr_cha,
            hp, max_hp, mana, max_mana, level, xp, xp_target, gold,
            strengths, weaknesses, equipment, skills, items
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
    ))
    char_id = c.lastrowid

    conn.commit()
    conn.close()

    # campaign_state là bảng riêng (xem db.py) — tạo row rỗng cho nhân vật mới
    # rồi set campaign_theme/setup_stage ngay, trước khi orchestrator chạy nền.
    db.create_campaign_state(char_id)
    db.update_campaign_state(
        char_id, campaign_theme=campaign_theme,
        setup_stage=("milestone" if reuse_bible else "bible"),
    )

    # Bất đồng bộ hoàn toàn — trả response NGAY, frontend mở modal tiến trình
    # và tự poll /setup_status tới khi stage == "ready"/"error".
    asyncio.create_task(run_campaign_setup(char_id, campaign_theme, campaign_mode, reuse_bible=reuse_bible))

    return {"status": "started"}


# ---------------------------------------------------------------------------
# Chơi lại / Xoá campaign — dropdown ở góc trên-trái character panel (game.html)
# ---------------------------------------------------------------------------

@app.post("/replay_campaign")
async def replay_campaign(data: dict):
    """mode="same_character": giữ NGUYÊN bible + nhân vật, reset tiến trình
    (turn/milestone/act/history) về vạch xuất phát rồi sinh lại milestone 1 +
    cảnh mở đầu.
    mode="new_character": giữ nguyên bible (thế giới/cốt truyện), chỉ báo
    frontend chuyển sang màn tạo nhân vật với cờ reuse_bible=1 — việc xoá
    nhân vật cũ + tạo nhân vật mới do chính /create_character xử lý (đã có
    logic "single save-slot" sẵn), tránh trùng lặp."""
    mode = data.get("mode")
    if mode not in ("same_character", "new_character"):
        return {"error": "mode không hợp lệ."}

    if campaign.load_campaign_bible() is None:
        return {"error": "Không tìm thấy campaign bible."}

    if mode == "new_character":
        return {"status": "redirect", "redirect": "/?reuse_bible=1"}

    char = db.get_latest_character()
    if not char:
        return {"error": "Chưa có nhân vật."}
    char_id = char["id"]

    conn = db.get_conn()
    c = conn.cursor()
    c.execute("DELETE FROM history")
    # Reset nhân vật về trạng thái "mới toanh" nhưng giữ nguyên build (race/
    # class/chỉ số/điểm mạnh-yếu/trang bị) — hp/mana đầy lại, level/xp/vàng về 0.
    c.execute(
        "UPDATE character SET hp = max_hp, mana = max_mana, level = 1, xp = 0, gold = 0 "
        "WHERE id = ?",
        (char_id,),
    )
    conn.commit()
    entities.reset_session(conn)
    conn.close()
    imagegen.clear_generated_images()

    db.update_campaign_state(
        char_id,
        current_turn=0, history_summary=None, summarized_up_to_turn=0,
        campaign_act_index=0, campaign_milestone_number=0, milestone_advanced_turn=0,
        current_milestone=None, story_state=None, last_result=None,
        last_turn_resolution=None, pending_action=None, pre_turn_snapshot=None,
        context_kind=None, context_name=None, context_name_vi=None, context_desc=None,
        context_image_path=None, setup_stage="milestone", setup_error=None,
    )

    asyncio.create_task(_restart_story_same_character(char_id))
    return {"status": "started"}


@app.post("/delete_campaign")
async def delete_campaign():
    """Xoá TOÀN BỘ — nhân vật, lịch sử, campaign bible, ảnh generate — quay về
    y hệt lần đầu mở app (phải chọn/generate kịch bản mới từ đầu)."""
    conn = db.get_conn()
    c = conn.cursor()
    c.execute("DELETE FROM character")
    c.execute("DELETE FROM history")
    c.execute("DELETE FROM campaign_state")
    conn.commit()
    entities.reset_session(conn)
    conn.close()

    imagegen.clear_generated_images()
    campaign.delete_campaign_bible()

    return {"status": "deleted", "redirect": "/"}


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

    state = db.get_campaign_state(char["id"])
    region = state["region"] if state else None
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

    last_result_raw = state["last_result"] if state else None
    last_result = None
    if last_result_raw:
        try:
            last_result = json.loads(last_result_raw)
        except (TypeError, json.JSONDecodeError):
            last_result = None

    conn_e = db.get_conn()
    active_entities = [
        {
            "key": e["key"], "name": e["name"], "type": e["entity_type"],
            "hp": e["hp"], "max_hp": e["max_hp"], "hostile": bool(e["hostile"]),
            "image_path": imagegen.entity_image_url(e["entity_type"], e["name"]),
        }
        for e in entities.get_active_entities(conn_e, char["id"])
    ]
    conn_e.close()

    return {
        "started": started,
        "region": region,
        "history": history,
        "last_result": last_result,
        "character": db.character_row_to_dict(char),
        "scene_context": _scene_context_dict(state),
        "active_entities": active_entities,
        "act_index": db.safe_int(state["campaign_act_index"], 0) if state else 0,
    }


def _milestone_and_act(state) -> dict:
    """Milestone hiện tại + tên Act — hiển thị ở left panel (giữa ảnh cảnh và
    danh sách NPC/quái) để người chơi dễ track đang làm milestone gì, và ở
    label "Chương I" thay vì chỉ số La Mã trơ trọi. Tiếng Anh, lấy thẳng từ
    Bible/milestone (vốn đã sinh bằng tiếng Anh nội bộ, không lộ ra UI trước
    đây) — không cần dịch, mục đích chỉ để TRACKING, không phải lời thoại."""
    milestone = db.load_current_milestone(state) if state else None
    act_index = db.safe_int(state["campaign_act_index"], 0) if state else 0

    act_purpose = None
    bible = campaign.load_campaign_bible()
    if bible:
        for a in bible.get("acts") or []:
            if db.safe_int(a.get("act"), -1) == act_index + 1:
                act_purpose = a.get("purpose")
                break

    return {
        "milestone_title": milestone.get("title") if milestone else None,
        "milestone_objective": milestone.get("objective") if milestone else None,
        "act_index": act_index,
        "act_title": act_purpose,
    }


def _scene_context_dict(state) -> dict:
    """Context panel hiện tại (location/quái/NPC mới nhất) — dùng chung giữa
    /game_state (resume sau reload) và /scene_context (poll ảnh sau khi
    imagegen.ensure_context_image chạy xong bất đồng bộ)."""
    if not state:
        return {
            "kind": None, "name": None, "description": None, "image_path": None,
            **_milestone_and_act(None),
        }
    # name hiển thị: ưu tiên tên tiếng Việt (context_name_vi), fallback tên
    # English canonical (context_name) cho các row cũ chưa có bản dịch.
    state_keys = state.keys()
    name_vi = state["context_name_vi"] if "context_name_vi" in state_keys else None
    return {
        "kind": state["context_kind"],
        "name": name_vi or state["context_name"],
        "description": state["context_desc"],
        "image_path": state["context_image_path"],
        **_milestone_and_act(state),
    }


@app.get("/scene_context")
async def scene_context():
    char = db.get_latest_character()
    if not char:
        return {"kind": None, "name": None, "description": None, "image_path": None}
    return _scene_context_dict(db.get_campaign_state(char["id"]))


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
