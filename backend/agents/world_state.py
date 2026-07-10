"""
world_state.py — Hệ thống ngày/đêm + thời tiết cho campaign.

Thiết kế nhẹ, không cần lịch game phức tạp:
- Thời gian trong ngày (time-of-day) được TÍNH THUẦN từ số turn đã chơi, không
  cần lưu DB riêng — mỗi period kéo dài TURNS_PER_PERIOD lượt, lặp vòng qua
  8 giai đoạn trong ngày (khoảng nửa ngày tới 1 ngày rưỡi cho 1 session
  45p-1h chơi ~15-25 turn, đủ để cảm nhận được sự trôi qua của thời gian).
- Thời tiết CẦN lưu DB (character.weather, character.weather_since_turn) vì
  nó phải "dính" qua vài turn rồi mới đổi (đổi mỗi turn sẽ vô lý), roll ngẫu
  nhiên có trọng số theo khí hậu từng region (lấy từ lore.get_weather_pool()).
- Cả 2 chỉ dùng để NHẮC bối cảnh cho model narrate cho khớp (ánh sáng, tầm
  nhìn, không khí...) — không phải cơ chế mà model cần báo cáo lại, nên
  không cần mở rộng OUTPUT FORMAT.
"""

import json
import os
import random

_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEBUG_STATE_PATH = os.path.join(_BACKEND_DIR, "game-data", "world_state.json")

TIME_PERIODS = [
    ("Dawn", "Bình minh vừa ló dạng, ánh sáng còn yếu và lành lạnh."),
    ("Morning", "Buổi sáng, ánh nắng rõ ràng, mọi hoạt động bắt đầu nhộn nhịp."),
    ("Midday", "Giữa trưa, nắng gắt nhất trong ngày, tầm nhìn tốt nhất."),
    ("Afternoon", "Buổi chiều, ánh nắng bắt đầu nghiêng, không khí dịu dần."),
    ("Dusk", "Hoàng hôn buông xuống, ánh sáng chuyển màu cam đỏ, bóng tối bắt đầu kéo dài."),
    ("Evening", "Buổi tối, trời đã tối hẳn, đèn đuốc/lửa trại là nguồn sáng chính."),
    ("Night", "Đêm khuya, bóng tối bao trùm, tầm nhìn hạn chế, nguy hiểm rình rập dễ ẩn mình hơn."),
    ("Deep Night", "Đêm muộn nhất, tĩnh lặng đến rợn người, hầu hết mọi sinh vật bình thường đã ngủ say."),
]

TURNS_PER_PERIOD = 2  # mỗi giai đoạn kéo dài 2 turn -> 1 vòng ngày/đêm = 16 turn

# Sau bao nhiêu turn thì roll lại thời tiết 1 lần (ngẫu nhiên trong khoảng này
# để không quá đều đặn/máy móc)
WEATHER_CHANGE_MIN_TURNS = 5
WEATHER_CHANGE_MAX_TURNS = 9


def get_time_period(turn_number: int):
    """Trả về (period_name, period_description, is_daytime)."""
    index = (max(turn_number, 0) // TURNS_PER_PERIOD) % len(TIME_PERIODS)
    name, desc = TIME_PERIODS[index]
    is_daytime = name in ("Dawn", "Morning", "Midday", "Afternoon")
    return name, desc, is_daytime


def roll_weather(region_name, current_weather=None, weather_since_turn=0, turn_number=0):
    """Roll thời tiết mới NẾU đã đến lúc đổi (dựa trên khoảng cách turn kể từ
    lần roll trước). Nếu chưa đến lúc, giữ nguyên thời tiết hiện tại.

    Trả về (weather, weather_since_turn) — weather_since_turn cập nhật thành
    turn_number hiện tại NẾU vừa roll thời tiết mới, ngược lại giữ nguyên.
    """
    import lore  # import trễ để tránh vòng lặp import nếu lore.py sau này cần world_state

    turns_elapsed = turn_number - (weather_since_turn or 0)
    needs_reroll = (
        current_weather is None
        or turns_elapsed >= random.randint(WEATHER_CHANGE_MIN_TURNS, WEATHER_CHANGE_MAX_TURNS)
    )

    if not needs_reroll:
        return current_weather, weather_since_turn

    pool = lore.get_weather_pool(region_name)
    if not pool:
        return current_weather or "Clear skies", turn_number

    weights = [entry.get("weight", 1) for entry in pool]
    choice = random.choices(pool, weights=weights, k=1)[0]
    return choice["weather"], turn_number


def format_world_state_context(turn_number: int, weather: str) -> str:
    """Block text nhét vào messages mỗi turn để model narrate khớp bối cảnh
    thời gian/thời tiết hiện tại."""
    period_name, period_desc, is_daytime = get_time_period(turn_number)

    lighting_note = (
        "Tầm nhìn tốt, dễ phát hiện kẻ địch/vật thể từ xa; hành động lén lút khó khăn hơn."
        if is_daytime else
        "Tầm nhìn hạn chế, dễ bị phục kích hoặc bất ngờ; hành động lén lút/ẩn nấp dễ dàng hơn."
    )

    return f"""## WORLD STATE (thời gian & thời tiết hiện tại — PHẢI phản ánh trong story, không cần nói rõ giờ/số liệu)
Thời điểm: {period_name} — {period_desc}
Thời tiết: {weather}
Ảnh hưởng: {lighting_note}

Hãy để bối cảnh này thấm vào cách miêu tả (ánh sáng, nhiệt độ, âm thanh, tâm trạng nhân
vật/NPC) một cách TỰ NHIÊN — không liệt kê "hiện tại là buổi X, thời tiết Y" như báo cáo,
mà lồng vào câu chuyện. Nếu thời tiết/thời điểm hợp lý ảnh hưởng đến độ khó của hành động
(vd trời tối dễ lẻn hơn, bão dễ trượt chân, sương mù khó ngắm bắn), hãy để điều đó ảnh
hưởng đến cách bạn miêu tả hệ quả — nhưng đừng bịa thêm cơ chế roll mới, chỉ là màu sắc
tường thuật."""


def write_debug_snapshot(turn_number: int, region: str, weather: str, weather_since_turn: int):
    """Ghi trạng thái world hiện tại ra world_state.json — CHỈ để debug/check
    trực tiếp bằng mắt (vd 'sao vẫn thấy Dawn' -> mở file này xem turn_number
    thực tế là bao nhiêu, có tăng không). KHÔNG phải nguồn sự thật cho game
    logic — DB (character.weather/weather_since_turn) vẫn là nơi duy nhất
    được đọc lại để tính toán, tránh rủi ro 2 nguồn dữ liệu lệch nhau nếu ghi
    file thất bại giữa chừng (vd hết dung lượng đĩa, quyền ghi...)."""
    period_name, period_desc, is_daytime = get_time_period(turn_number)
    snapshot = {
        "turn_number": turn_number,
        "region": region,
        "time_period": period_name,
        "is_daytime": is_daytime,
        "weather": weather,
        "weather_since_turn": weather_since_turn,
        "turns_per_period": TURNS_PER_PERIOD,
        "period_index": (max(turn_number, 0) // TURNS_PER_PERIOD) % len(TIME_PERIODS),
    }
    try:
        with open(DEBUG_STATE_PATH, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, ensure_ascii=False, indent=2)
    except OSError as e:
        print(f"[DEBUG] Không ghi được world_state.json: {e}")