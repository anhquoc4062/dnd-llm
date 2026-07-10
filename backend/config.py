MODEL = "qwen3:14b"
# MODEL = "mistral-small:22b"
# MODEL = "phi4:14b"
# MODEL = "mistral-nemo:12b"

OPTIONS = {
    "num_ctx": 8196,
    "num_predict": 800,
    "temperature": 0.7,
}

ATTR_KEYS = ["str", "dex", "con", "int", "wis", "cha"]
