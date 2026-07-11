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
import threading
from concurrent.futures import ThreadPoolExecutor

import db

MODEL_ID = "Lykon/dreamshaper-8"
LCM_LORA_ID = "latent-consistency/lcm-lora-sdv1-5"

STYLE_SUFFIX = "dark fantasy illustration, painterly, dramatic lighting, cinematic composition"
NEGATIVE_PROMPT = "low quality, blurry, deformed, text, watermark, letters, writing, signature"

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


def _generate_and_save(kind: str, name: str, visual_prompt: str) -> str:
    """Chạy TRONG thread pool — blocking, gọi GPU thật. Trả về path tương đối
    (dùng để lưu DB / build URL cho frontend)."""
    pipe = _get_pipe()
    prompt = f"{STYLE_SUFFIX}, {visual_prompt}" if visual_prompt else STYLE_SUFFIX

    image = pipe(
        prompt=prompt,
        negative_prompt=NEGATIVE_PROMPT,
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


async def ensure_context_image(char_id: int, kind: str, name: str, visual_prompt: str):
    """Fire-and-forget: gọi qua asyncio.create_task(...), KHÔNG await ở nơi gọi.
    Cache theo (kind, name) — nếu file đã tồn tại thì dùng lại luôn, không
    generate lại. Chỉ ghi context_image_path vào DB nếu context_name hiện tại
    của nhân vật VẪN còn khớp `name` (tránh race: player đã chuyển sang context
    khác trong lúc ảnh này còn đang generate)."""
    path = _image_path(kind, name)

    if not os.path.exists(path):
        loop = asyncio.get_running_loop()
        try:
            path = await loop.run_in_executor(_executor, _generate_and_save, kind, name, visual_prompt)
        except Exception as e:
            print(f"[DEBUG] imagegen: lỗi generate ảnh cho {kind}='{name}': {e}")
            return

    conn = db.get_conn()
    row = conn.execute("SELECT context_name FROM character WHERE id = ?", (char_id,)).fetchone()
    if row and (row["context_name"] or "") == name:
        conn.execute(
            "UPDATE character SET context_image_path = ? WHERE id = ?",
            (_to_static_url(path), char_id),
        )
        conn.commit()
    conn.close()
