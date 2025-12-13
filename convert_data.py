import json
from pathlib import Path

# ====================================
# 写死路径（按你的真实路径修改）
# ====================================

INPUT_JSON = Path("/mnt/sda/shaoyang/datasets/cxr/cxr_splits.json")
OUTPUT_DIR = Path("/mnt/sda/shaoyang/datasets/cxr/llava_llada/")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_OUT = OUTPUT_DIR / "train_llava_llada.json"
TEST_OUT  = OUTPUT_DIR / "test_llava_llada.json"


# ====================================
# Prompt 构造（无 system，LLaDA 自动加）
# ====================================

ASSISSTANT_PROMPT = "<image>\nProvide a detailed description of the findings in the radiology image."
CLINICAL_PROMPT   = "Following clinical context:"


def _clean_text(t):
    if t is None:
        return ""
    t = str(t).replace("\n", " ").replace("\r", " ")
    return " ".join(t.split()).strip()


def _first_non_empty(features, *keys):
    for k in keys:
        v = _clean_text(features.get(k, ""))
        if v:
            return v
    return ""


def _build_clinical_prompt(features):
    parts = []
    ind = _first_non_empty(features, "indication", "Indication", "INDICATION")
    comp = _first_non_empty(features, "comparison", "Comparison", "COMPARISON")
    hist = _first_non_empty(features, "history", "History", "HISTORY")
    tech = _first_non_empty(features, "technique", "Technique", "TECHNIQUE")

    if ind:  parts.append(f"Indication: {ind}")
    if comp: parts.append(f"Comparison: {comp}")
    if hist: parts.append(f"History: {hist}")
    if tech: parts.append(f"Technique: {tech}")

    return " ".join(parts)


def _build_total_prompt(features):
    clinical = _build_clinical_prompt(features)
    if clinical:
        return f"{ASSISSTANT_PROMPT} {CLINICAL_PROMPT} {clinical}"
    else:
        return ASSISSTANT_PROMPT


# ====================================
# split 转换逻辑（⚠ test 保留 label）
# ====================================

def convert_split(split_items):
    converted = []

    for ex in split_items:
        prompt = _build_total_prompt(ex)
        report = _clean_text(ex.get("report", ""))

        # Image path
        paths = ex.get("image_path", [])
        img = paths[0] if isinstance(paths, list) and len(paths) > 0 else ""

        conversations = [
            {"from": "human", "value": prompt},
            {"from": "gpt",   "value": report}  # ⚠ test 也保留 label
        ]

        converted.append({
            "id": ex.get("id", ""),
            "image": img,
            "conversations": conversations
        })

    return converted


# ====================================
# 主流程
# ====================================

print(f"读取源文件: {INPUT_JSON}")

data = json.loads(INPUT_JSON.read_text())

train_raw = data["train"]
test_raw  = data["test"]

print(f"[原始] train: {len(train_raw)}")
print(f"[原始] test : {len(test_raw)}")

train_out = convert_split(train_raw)
test_out  = convert_split(test_raw)

TRAIN_OUT.write_text(json.dumps(train_out, indent=2, ensure_ascii=False))
TEST_OUT.write_text(json.dumps(test_out, indent=2, ensure_ascii=False))

print(f"[输出] train_llava_llada.json: {len(train_out)}")
print(f"[输出] test_llava_llada.json : {len(test_out)}")

assert len(train_raw) == len(train_out)
assert len(test_raw) == len(test_out)

print("✅ 完成：train/test 数量一致，test 已包含 gpt label，可用于评估指标。")
