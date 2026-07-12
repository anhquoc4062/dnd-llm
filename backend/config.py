MODEL = "qwen3:8b"
# MODEL = "qwen3:14b"
# MODEL = "mistral-small:22b"
# MODEL = "phi4:14b"
# MODEL = "mistral-nemo:12b"

OPTIONS = {
    "num_ctx": 8196,
    # 800 từng gây cắt cụt JSON ở những lượt vừa có entity mới + location mới +
    # 4 choices dài cùng lúc (fallback parse hỏng -> lộ JSON thô ra story cho
    # người chơi). num_predict chỉ là TRẦN, model tự dừng sớm khi đóng JSON
    # xong -> nâng lên không làm chậm các lượt bình thường (vốn đã dừng dưới
    # 800 từ trước), chỉ cho các lượt hiếm bị tràn thêm chỗ để nói xong (~40
    # token/s trên RTX 4070 Super -> tối đa +7-8s cho các lượt đó).
    "num_predict": 1100,
    "temperature": 0.7,
}

ATTR_KEYS = ["str", "dex", "con", "int", "wis", "cha"]
