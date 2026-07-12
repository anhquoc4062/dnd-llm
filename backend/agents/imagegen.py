"""
imagegen.py — Generate ảnh minh hoạ scene (location/quái/NPC mới xuất hiện) bằng
Stable Diffusion 1.5 (DreamShaper 8) + LCM-LoRA. Chạy hoàn toàn bất đồng bộ với
lượt kể chuyện của DM — dungeon_master.py chỉ fire-and-forget gọi
ensure_context_image(), KHÔNG await kết quả trước khi trả response /chat/narrate.

Model được load lười (lazy) ở lần gọi đầu tiên, không chặn lúc backend khởi
động. ThreadPoolExecutor(max_workers=1) vừa giữ event loop rảnh (generate là
tác vụ blocking CPU/GPU), vừa tự nhiên serialize truy cập GPU — tránh tranh
VRAM với Ollama đang chạy song song (đã benchmark: peak VRAM SD ~2.75-3GB,
generate ~0.6-0.7s/ảnh trên RTX 4070 Super).
"""

import asyncio
import os
import re
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor

import db

MODEL_ID = "Lykon/dreamshaper-8"
LCM_LORA_ID = "latent-consistency/lcm-lora-sdv1-5"

STYLE_SUFFIX = """
dark fantasy illustration,
digital painting,
fantasy concept art,
tabletop RPG artwork,
Dungeons & Dragons style,
painterly,
cinematic storytelling,
dynamic composition,
expressive characters,
realistic anatomy,
highly detailed,
dramatic cinematic lighting,
rich environmental storytelling,
medieval fantasy,
weathered,
grounded realism,
masterpiece
"""
NEGATIVE_PROMPT = """
anime,
manga,
cartoon,
chibi,
3d render,
cgi,
plastic skin,
photograph,
modern clothing,
sci-fi,
cyberpunk,
oversaturated,
low quality,
low detail,
blurry,
deformed,
bad anatomy,
bad hands,
extra fingers,
missing fingers,
duplicate,
cropped,
watermark,
text,
logo,
signature
"""

# Style RIÊNG theo từng loại ảnh — nối SAU STYLE_SUFFIX chung. Đây là chỗ phân
# biệt "cảnh gì" (trước đây mọi ảnh dùng chung 1 style nên quái hay ra giống
# người). Từ khoá lấy theo yêu cầu người dùng.
KIND_STYLE = {
    "npc": """
character portrait,
upper body framing,
detailed expressive face,
travel-worn attire,
subtle facial detail,
determined presence
""",
    "monster": """
grotesque creature,
inhuman monster,
twisted anatomy,
unnatural proportions,
non-human silhouette,
menacing presence,
ancient evil,
full-body creature shot
""",
    "location": """
vast environment,
epic scale,
atmospheric perspective,
immersive scenery,
environmental storytelling,
no characters
""",
    "object": """
detailed object study,
centered focal subject,
tangible materials,
intricate surface detail,
still object,
no people
""",
}

# Style phụ trợ cho cảnh chiến đấu — chỉ nối thêm khi entity là quái/NPC thù
# địch đang giao chiến (imagegen được gọi kèm cờ combat). KHÔNG phải 1 loại ảnh
# riêng, chỉ là lớp style động chồng lên.
COMBAT_STYLE = """
dynamic action pose,
sense of motion,
flying debris,
intense expression,
dramatic perspective
"""

# Negative RIÊNG theo loại — quan trọng nhất: ép quái KHÔNG ra hình người, và
# ép location/object KHÔNG lòi ra sinh vật (bug thật: ảnh tòa tháp ra 1 con thú
# khổng lồ vì visual_prompt bị lẫn mô tả quái).
KIND_NEGATIVE = {
    "monster": "human face, normal human, humanoid, attractive, symmetrical human anatomy, portrait of a person, cute",
    "location": "creature, monster, beast, animal, giant creature, humanoid figure, person, people, character, portrait, face",
    "object": "creature, monster, beast, animal, person, people, character, face, living being",
}

# Tương đối so với cwd của backend lúc chạy (uvicorn chạy từ trong backend/,
# xem main.py: StaticFiles(directory="../") -> "../data/..." map ra
# D:\LLM\UI\data\..., truy cập được qua /static/data/generated/...
OUTPUT_ROOT = "../data/generated"

_pipe = None
_pipe_lock = threading.Lock()
_executor = ThreadPoolExecutor(max_workers=1)


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", (name or "").strip().lower()).strip("_")
    return slug or "unknown"


def _image_path(kind: str, name: str) -> str:
    return os.path.join(OUTPUT_ROOT, kind, f"{_slugify(name)}.png")


def clear_generated_images():
    """Xoá toàn bộ ảnh đã generate (location/monster/npc) — gọi lúc tạo nhân
    vật mới (main.py: /create_character), cùng lúc với DELETE FROM character/
    history. Game chỉ có 1 save-slot nên ảnh của campaign cũ không còn lý do
    gì để giữ lại — nếu không xoá, ảnh cache theo tên (xem ensure_context_image)
    có thể bị TÁI DÙNG NHẦM cho 1 entity/location trùng tên nhưng thuộc
    campaign hoàn toàn khác (vd 2 campaign khác nhau đều có 1 quái tên
    "Shadow Wraith" nhưng appearance khác nhau)."""
    if os.path.isdir(OUTPUT_ROOT):
        shutil.rmtree(OUTPUT_ROOT)
    os.makedirs(OUTPUT_ROOT, exist_ok=True)


def _get_pipe():
    """Load pipeline lười, chỉ 1 lần, thread-safe (2 request đầu tiên gọi gần
    nhau không load 2 lần)."""
    global _pipe
    if _pipe is not None:
        return _pipe
    with _pipe_lock:
        if _pipe is None:
            import torch
            from diffusers import StableDiffusionPipeline, LCMScheduler

            pipe = StableDiffusionPipeline.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.float16,
                safety_checker=None,
            )
            pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
            pipe.load_lora_weights(LCM_LORA_ID)
            pipe.fuse_lora()
            pipe.to("cuda")
            _pipe = pipe
    return _pipe


def _compose_prompt(kind: str, visual_prompt: str, combat: bool) -> tuple[str, str]:
    """Ghép prompt cuối theo loại: STYLE_SUFFIX chung + KIND_STYLE riêng
    (+ COMBAT_STYLE nếu đang chiến đấu) + visual_prompt của chủ thể. Trả
    (prompt, negative_prompt) — negative cũng cộng thêm phần riêng theo loại
    (vd chặn quái ra hình người)."""
    parts = [STYLE_SUFFIX.strip(), KIND_STYLE.get(kind, "").strip()]
    if combat and kind in ("monster", "npc"):
        parts.append(COMBAT_STYLE.strip())
    if visual_prompt:
        parts.append(visual_prompt.strip())
    prompt = ", ".join(p for p in parts if p)

    neg = NEGATIVE_PROMPT.strip()
    if kind in KIND_NEGATIVE:
        neg = f"{neg}, {KIND_NEGATIVE[kind]}"
    return prompt, neg


def _generate_and_save(kind: str, name: str, visual_prompt: str, combat: bool = False) -> str:
    """Chạy TRONG thread pool — blocking, gọi GPU thật. Trả về path tương đối
    (dùng để lưu DB / build URL cho frontend)."""
    pipe = _get_pipe()
    prompt, negative = _compose_prompt(kind, visual_prompt, combat)

    image = pipe(
        prompt=prompt,
        negative_prompt=negative,
        num_inference_steps=4,
        guidance_scale=1.0,
        width=768,
        height=512,
    ).images[0]

    path = _image_path(kind, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    image.save(path)
    return path


def _to_static_url(path: str) -> str:
    """Đổi path tương đối (vd '../data/generated/monster/erased_one.png') ->
    URL frontend dùng được qua /static (StaticFiles mount tại D:\\LLM\\UI)."""
    normalized = path.replace("\\", "/").lstrip("./")
    if normalized.startswith("../"):
        normalized = normalized[3:]
    return f"/static/{normalized}"


async def ensure_context_image(char_id: int, kind: str, name: str, visual_prompt: str, combat: bool = False):
    """Fire-and-forget: gọi qua asyncio.create_task(...), KHÔNG await ở nơi gọi.
    Cache theo (kind, name) — nếu file đã tồn tại thì dùng lại luôn, không
    generate lại. Chỉ ghi context_image_path vào DB nếu context_name hiện tại
    VẪN còn khớp `name` (tránh race: player đã chuyển sang context khác trong
    lúc ảnh này còn đang generate).

    LƯU Ý: context_* nằm ở bảng campaign_state (không phải character — cột cùng
    tên bên character chỉ là di tích migration cũ, luôn NULL). Trước đây hàm này
    đọc/ghi nhầm bảng character -> campaign_state.context_image_path không bao
    giờ được set -> frontend (đọc campaign_state qua /scene_context) poll ảnh
    mãi không ra. Đã sửa về đúng campaign_state."""
    path = _image_path(kind, name)

    if not os.path.exists(path):
        loop = asyncio.get_running_loop()
        try:
            path = await loop.run_in_executor(_executor, _generate_and_save, kind, name, visual_prompt, combat)
        except Exception as e:
            print(f"[DEBUG] imagegen: lỗi generate ảnh cho {kind}='{name}': {e}")
            return

    conn = db.get_conn()
    row = conn.execute("SELECT context_name FROM campaign_state WHERE character_id = ?", (char_id,)).fetchone()
    if row and (row["context_name"] or "") == name:
        conn.execute(
            "UPDATE campaign_state SET context_image_path = ? WHERE character_id = ?",
            (_to_static_url(path), char_id),
        )
        conn.commit()
    conn.close()

def entity_image_url(kind: str, name: str) -> str | None:
    """URL ảnh đã generate cho 1 entity (monster/npc) trong panel "Trong cảnh",
    dùng làm avatar thay emoji khi đã có sẵn — KHÔNG tự generate mới ở đây
    (ensure_context_image mới là nơi generate, chạy khi entity đó từng là
    context chính). Trả None nếu chưa có ảnh -> frontend fallback về emoji."""
    if not kind or not name:
        return None
    path = _image_path(kind, name)
    return _to_static_url(path) if os.path.isfile(path) else None
