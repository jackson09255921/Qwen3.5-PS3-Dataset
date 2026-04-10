import json
import random
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple

# ========== 共用工具 ==========

def normalize_role(role: str) -> str:
    role = role.lower()
    if role in ["human", "user"]:
        return "human"
    if role in ["gpt", "assistant", "bot"]:
        return "gpt"
    if role == "system":
        return "system"
    return role

def randomize_image_position_in_value(text: str) -> str:
    """
    保證同一個 value 內，<image> 只在最前面或最後面。
    不改變 <image> 數量，只做頭/尾兩種 pattern 的隨機分配。
    """
    if "<image>" not in text:
        return text
    
    parts = text.split("<image>")
    base = "".join(parts).strip("\n")

    k = len(parts) - 1
    if k <= 0:
        return text

    image_blocks = "\n".join(["<image>"] * k)
    
    if base == "":
        return image_blocks
    
    put_at_head = random.random() < 0.5
    
    if put_at_head:
        return f"{image_blocks}\n{base}"
    else:
        return f"{base}\n{image_blocks}"

# 抓 [0.196, 0.003, 0.776, 0.968] 這種 bbox
BBOX_PATTERN = re.compile(
    r"\[\s*([0-9.]+)\s*,\s*([0-9.]+)\s*,\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\]"
)

def extract_svit_bboxes_and_phrases(
    text: str,
    image_index: int = 0,
) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    專門給 SVIT referring_qa 用：
    原始是 <st>phrase<ed> [x1,y1,x2,y2]，從中抽 phrase + bbox。
    轉 float 出錯時：
      1) print 出壞掉的 groups 和部分 context
      2) 當成普通文字略過，不中斷整個轉檔
    """
    bboxes: List[Dict[str, Any]] = []
    targets: List[Dict[str, Any]] = []

    new_text_parts: List[str] = []
    last_end = 0

    # 找所有 bbox
    for m in BBOX_PATTERN.finditer(text):
        start, end = m.span()

        # 往前找最近一個 <st> ... <ed>
        st_pos = text.rfind("<st>", 0, start)
        ed_pos = text.rfind("<ed>", 0, start)
        phrase = "the region"

        if st_pos != -1 and ed_pos != -1 and st_pos < ed_pos:
            phrase = text[st_pos + len("<st>"):ed_pos].strip()
            head = text[last_end:st_pos]
        else:
            head = text[last_end:start]

        # anchor phrase: head + <st>phrase<ed>
        anchor_phrase = f"{head}<st>{phrase}<ed>"

        # 用 try/except 做 bbox 轉 float & debug
        coords: List[float] = []
        ok = True
        bad_groups = []
        for i in range(1, 5):
            g = m.group(i)
            bad_groups.append(g)
            try:
                coords.append(float(g))
            except (TypeError, ValueError):
                ok = False
                break

        if not ok:
            # debug：印出壞掉的 group 跟附近 context
            print("==== BAD SVIT BBOX MATCH ====")
            print("groups:", bad_groups)
            ctx_start = max(0, start - 120)
            ctx_end = min(len(text), end + 120)
            print("context:", text[ctx_start:ctx_end].replace("\n", "\\n"))
            print("================================")
            # 這一個 bracket 不產生 bbox，原文保留
            new_text_parts.append(text[last_end:end])
            last_end = end
            continue

        bbox_index = len(bboxes)
        bboxes.append({
            "image_index": image_index,
            "bbox": coords,
        })
        targets.append({
            "type": "refer",
            "phrase": phrase,
            "bbox_index": bbox_index,
        })

        new_text_parts.append(anchor_phrase)
        new_text_parts.append(" <bbox> ")
        last_end = end

    new_text_parts.append(text[last_end:])
    new_text = "".join(new_text_parts)
    return new_text, bboxes, targets


def get_image_path_from_item(item: Dict[str, Any], img_path_tmpl: str) -> str:
    """
    優先用原 json 裡的 image / image_path 欄位；
    如果沒有，再 fallback 用 image_id + img_path_tmpl。
    """
    image_path = item.get("image") or item.get("image_path")
    if image_path is not None:
        return image_path
    image_id = item["image_id"]
    return img_path_tmpl.format(image_id)


# ========== SVIT 4 檔轉換 ==========

def build_svit_referring_sample(
    image_id: int,
    img_path_tmpl: str,
    qa_idx: int,
    qa_item: Dict[str, Any],
    full_item: Dict[str, Any],
    random_image_pos: bool = True,
) -> Dict[str, Any]:
    """
    referring_qa.json:
      { "image_id": int, "image": "...", "conversations": [...] }
    這裡把每個 content 視為單條樣本。
    """
    image_path = get_image_path_from_item(full_item, img_path_tmpl)
    conv = qa_item["content"]
    assert len(conv) == 2, "referring_qa 裡預期是一問一答"

    user_raw = conv[0]
    asst_raw = conv[1]

    user = normalize_role(user_raw["from"])
    asst = normalize_role(asst_raw["from"])

    user_val = user_raw["value"]
    asst_val = asst_raw["value"]

    if random_image_pos and "<image>" in user_val:
        user_val = randomize_image_position_in_value(user_val)

    new_asst, bboxes, targets = extract_svit_bboxes_and_phrases(
        asst_val,
        image_index=0,
    )

    convs = [
        {"from": user, "value": user_val},
        {"from": asst, "value": new_asst},
    ]

    sid = f"svit_ref_{image_id}_{qa_idx}"
    return {
        "id": sid,
        "task_type": "ground_rec",
        "images": [image_path],
        "bboxes": bboxes,
        "grounding_targets": targets,
        "conversations": convs,
        "meta": {
            "src_dataset": "svit_referring_qa",
            "image_id": image_id,
            "qa_idx": qa_idx,
        },
    }

def build_svit_detail_sample(
    image_id: int,
    img_path_tmpl: str,
    item: Dict[str, Any],
    random_image_pos: bool = True,
) -> Dict[str, Any]:
    """
    detail_description.json:
      { "image_id": int, "image": "...", "conversations": [...] }
    通常只有一組 user 要求 + 長 caption。
    """
    image_path = get_image_path_from_item(item, img_path_tmpl)
    content = item["conversations"][0]["content"]
    convs = []
    for turn in content:
        role = normalize_role(turn["from"])
        val = turn["value"]
        if random_image_pos and role == "user" and "<image>" in val:
            val = randomize_image_position_in_value(val)
        convs.append({"from": role, "value": val})

    sid = f"svit_detail_{image_id}"
    return {
        "id": sid,
        "task_type": "vqa",
        "images": [image_path],
        "bboxes": [],
        "grounding_targets": [],
        "conversations": convs,
        "meta": {
            "src_dataset": "svit_detail_description",
            "image_id": image_id,
        },
    }

def build_svit_complex_sample(
    image_id: int,
    img_path_tmpl: str,
    qa_idx: int,
    qa_item: Dict[str, Any],
    full_item: Dict[str, Any],
    random_image_pos: bool = True,
) -> Dict[str, Any]:
    """
    complex_reasoning.json:
      { "image_id": int, "image": "...", "conversations": [...] }
    每個 content 就是一問一答。
    """
    image_path = get_image_path_from_item(full_item, img_path_tmpl)
    conv = qa_item["content"]
    assert len(conv) == 2
    user_raw = conv[0]
    asst_raw = conv[1]

    user = normalize_role(user_raw["from"])
    asst = normalize_role(asst_raw["from"])
    user_val = user_raw["value"]
    asst_val = asst_raw["value"]

    if random_image_pos and "<image>" in user_val:
        user_val = randomize_image_position_in_value(user_val)

    convs = [
        {"from": user, "value": user_val},
        {"from": asst, "value": asst_val},
    ]

    sid = f"svit_complex_{image_id}_{qa_idx}"
    return {
        "id": sid,
        "task_type": "vqa",
        "images": [image_path],
        "bboxes": [],
        "grounding_targets": [],
        "conversations": convs,
        "meta": {
            "src_dataset": "svit_complex_reasoning",
            "image_id": image_id,
            "qa_idx": qa_idx,
        },
    }

def build_svit_conversation_samples(
    image_id: int,
    img_path_tmpl: str,
    item: Dict[str, Any],
    random_image_pos: bool = True,
) -> List[Dict[str, Any]]:
    """
    conversation.json:
      { "image_id": int, "image": "...", "conversations": [...] }
    這裡選擇「每個 topic → 一條 multi-turn sample」。
    """
    image_path = get_image_path_from_item(item, img_path_tmpl)
    out: List[Dict[str, Any]] = []

    for topic_idx, topic_block in enumerate(item["conversations"]):
        content = topic_block["content"]
        convs = []
        for turn in content:
            role = normalize_role(turn["from"])
            val = turn["value"]
            if random_image_pos and role == "user" and "<image>" in val:
                val = randomize_image_position_in_value(val)
            convs.append({"from": role, "value": val})

        sid = f"svit_conv_{image_id}_{topic_idx}"
        sample = {
            "id": sid,
            "task_type": "vqa",
            "images": [image_path],
            "bboxes": [],
            "grounding_targets": [],
            "conversations": convs,
            "meta": {
                "src_dataset": "svit_conversation",
                "image_id": image_id,
                "topic": topic_block.get("topic", ""),
                "topic_idx": topic_idx,
            },
        }
        out.append(sample)
    return out


# ========== I/O 主程式 ==========

def load_json(path: str | Path) -> Any:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def svit_convert(
    in_path: str,
    out_path: str,
    svit_type: str,
    img_path_tmpl: str = "svit_images/{:06d}.jpg",
    random_image_pos: bool = True,
    seed: int = 42,
):
    """
    svit_type: one of ["referring_qa", "detail_description", "conversation", "complex_reasoning"]
    img_path_tmpl: 把 image_id 映成實際路徑的模板（只有在 json 沒有 image 欄位時才會用到）
    """
    random.seed(seed)
    in_path = Path(in_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data = load_json(in_path)

    with out_path.open("w", encoding="utf-8") as f_out:
        for item in data:
            image_id = item["image_id"]

            if svit_type == "referring_qa":
                for qa_idx, qa in enumerate(item["conversations"]):
                    sample = build_svit_referring_sample(
                        image_id,
                        img_path_tmpl,
                        qa_idx,
                        qa,
                        full_item=item,
                        random_image_pos=random_image_pos,
                    )
                    f_out.write(json.dumps(sample, ensure_ascii=False) + "\n")

            elif svit_type == "detail_description":
                sample = build_svit_detail_sample(
                    image_id,
                    img_path_tmpl,
                    item,
                    random_image_pos=random_image_pos,
                )
                f_out.write(json.dumps(sample, ensure_ascii=False) + "\n")

            elif svit_type == "conversation":
                samples = build_svit_conversation_samples(
                    image_id,
                    img_path_tmpl,
                    item,
                    random_image_pos=random_image_pos,
                )
                for s in samples:
                    f_out.write(json.dumps(s, ensure_ascii=False) + "\n")

            elif svit_type == "complex_reasoning":
                for qa_idx, qa in enumerate(item["conversations"]):
                    sample = build_svit_complex_sample(
                        image_id,
                        img_path_tmpl,
                        qa_idx,
                        qa,
                        full_item=item,
                        random_image_pos=random_image_pos,
                    )
                    f_out.write(json.dumps(sample, ensure_ascii=False) + "\n")

            else:
                raise ValueError(f"Unknown svit_type={svit_type}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", required=True,
                        help="Path to SVIT json")
    parser.add_argument("--out", dest="out_path", required=True,
                        help="Path to output unified jsonl")
    parser.add_argument("--type", dest="svit_type", required=True,
                        choices=["referring_qa", "detail_description", "conversation", "complex_reasoning"])
    parser.add_argument("--img_tmpl", type=str, default="svit_images/{:06d}.jpg",
                        help="Template to map image_id to image path, e.g. 'svit/{:06d}.jpg'")
    parser.add_argument("--no_random_image_pos", action="store_true",
                        help="Disable randomizing image position in value")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    svit_convert(
        args.in_path,
        args.out_path,
        svit_type=args.svit_type,
        img_path_tmpl=args.img_tmpl,
        random_image_pos=not args.no_random_image_pos,
        seed=args.seed,
    )
