import json
import random
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple
import argparse

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
    """保證 <image> 只在頭或尾，隨機分配"""
    if "<image>" not in text:
        return text
    
    parts = text.split("<image>")
    k = len(parts) - 1
    if k <= 0:
        return text
    
    base = "".join(parts).strip("\n")
    image_blocks = "\n".join(["<image>"] * k)
    
    if random.random() < 0.5:
        return f"{image_blocks}\n{base}"
    else:
        return f"{base}\n{image_blocks}"

# Bbox 解析
BBOX_PATTERN = re.compile(r"\[\s*([0-9.]+)\s*,\s*([0-9.]+)\s*,\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\]")

def extract_ref3rec_phrase_from_user(user_text: str) -> str:
    """Ref3Rec 專用：從 human prompt 抽精確 phrase"""
    patterns = [
        r"locate\s+(.*?)(?:in\s+and|\s+and|\s+in\s+|\?|$)",  
        r"coordinates of\s+(.*?)(?:\?|$)",  
        r"find and tell me the coordinates of\s+(.*?)(?:\?|$)",
        r"the coordinates of\s+(.*?)(?:\?|$)"

    ]
    for pat in patterns:
        m = re.search(pat, user_text, re.IGNORECASE)
        if m:
            phrase = m.group(1).strip(",.?")
            phrase = phrase.replace(" in  and", "").replace(" in  ", "").strip()
            return phrase if phrase else "the region"
    return "the region"

def extract_bbox_and_phrase_from_answer(text: str, image_index: int = 0) -> Tuple[str, List[Dict], List[Dict]]:
    """從回答抽 bbox + phrase，改寫成 <st>phrase<ed> <bbox>"""
    bboxes = []
    targets = []
    matches = list(BBOX_PATTERN.finditer(text))
    
    if not matches:
        return text, bboxes, targets
    
    new_text_parts = []
    last_end = 0
    
    for m in matches:
        start, end = m.span()
        prefix = text[last_end:start].rstrip()
        
        sep_pos = max([prefix.rfind(","), prefix.rfind(".")], default=-1)
        phrase = prefix[sep_pos+1:].strip() if sep_pos != -1 else prefix.strip()
        phrase = phrase or "the region"
        
        head = prefix[:len(prefix) - len(phrase)] if phrase else prefix
        
        coords = [float(m.group(i)) for i in range(1, 5)]
        bbox_index = len(bboxes)
        
        bboxes.append({"image_index": image_index, "bbox": coords})
        targets.append({"type": "refer", "phrase": phrase, "bbox_index": bbox_index})
        
        new_text_parts.extend([head, f"<st>{phrase}<ed>", " <bbox>"])
        last_end = end
    
    new_text_parts.append(text[last_end:])
    return "".join(new_text_parts), bboxes, targets

def classify_task_type(bboxes: List[Dict], grounding_targets: List[Dict], convs: List[Dict]) -> str:
    """完美分類邏輯"""
    if not bboxes:
        user_text = convs[0]["value"].lower()
        if "<image>" not in user_text:
            return "text_instruct"
        return "vqa"
    
    gt_types = [t.get("type", "refer") for t in grounding_targets]
    if "refer" in gt_types:
        return "ground_rec"
    elif "region_caption" in gt_types:
        return "ground_region_caption"
    elif "region_qa" in gt_types:
        return "ground_region_qa"
    return "vqa"

def is_vqa_only_dataset(src_name: str) -> bool:
    """純 VQA 資料集，不解析 bbox"""
    vqa_datasets = {
        "gqa", "llava-simple", "svit-simple", "llava-150k-simple",
        "textvqa", "docvqa", "scienceqa", "llava_single", "llava_multi"
    }
    return src_name.lower() in vqa_datasets

def build_unified_sample(
    raw_item: Dict[str, Any],
    src_name: str,
    random_image_pos: bool = True,
    enable_bbox: bool = True,
) -> Dict[str, Any]:
    """🔥 完整支援 Ref3Rec + Ref3Reg！"""
    sid = str(raw_item.get("id", raw_item.get("sample_id", "")))
    image_path = raw_item.get("image") or raw_item.get("image_path")
    images = [image_path] if image_path else []
    
    convs = raw_item.get("conversations", [])
    norm_convs = []
    unified_bboxes = []
    grounding_targets = []
    
    # 🔥 Ref3Rec：存 user phrase
    ref3rec_user_phrase = None
    if src_name.startswith("ref3rec") and convs:
        ref3rec_user_phrase = extract_ref3rec_phrase_from_user(convs[0].get("value", ""))
    
    # 🔥 Ref3Reg：先解析 user bbox
    ref3reg_user_bboxes = []
    if src_name.startswith("ref3reg") and convs:
        user_value = convs[0].get("value", "")
        user_matches = list(BBOX_PATTERN.finditer(user_value))
        for m in user_matches:
            coords = [float(m.group(i)) for i in range(1, 5)]
            ref3reg_user_bboxes.append({"image_index": 0, "bbox": coords})
    
    for turn_idx, c in enumerate(convs):
        role = normalize_role(c.get("from", ""))
        value = c.get("value", "")
        
        # 🔥 Ref3Reg：human bbox → region_caption
        if src_name.startswith("ref3reg") and role == "human" and ref3reg_user_bboxes:
            unified_bboxes.extend(ref3reg_user_bboxes)
            # gpt 回答 → caption
            caption = convs[turn_idx+1].get("value", "") if turn_idx+1 < len(convs) else ""
            grounding_targets.append({
                "type": "region_caption",
                "bbox_index": len(unified_bboxes) - len(ref3reg_user_bboxes),
                "caption": caption
            })
            # human prompt：數字 → <bbox>
            value = BBOX_PATTERN.sub("<bbox>", value)
        
        # 隨機化 image 位置
        if random_image_pos and role == "human" and "<image>" in value:
            value = randomize_image_position_in_value(value)
        
        # gpt bbox 解析 (Shikra/Ref3Rec)
        if enable_bbox and role == "gpt" and not is_vqa_only_dataset(src_name):
            new_value, bboxes, targets = extract_bbox_and_phrase_from_answer(value)
            
            if bboxes:
                base_idx = len(unified_bboxes)
                unified_bboxes.extend(bboxes)
                
                # Ref3Rec：用 human phrase
                if src_name.startswith("ref3rec") and ref3rec_user_phrase:
                    for t in targets:
                        t["phrase"] = ref3rec_user_phrase
                
                for t in targets:
                    t_copy = t.copy()
                    t_copy["bbox_index"] = base_idx + t["bbox_index"]
                    grounding_targets.append(t_copy)
                
                value = new_value
        
        norm_convs.append({"from": role, "value": value})
    
    task_type = classify_task_type(unified_bboxes, grounding_targets, norm_convs)
    
    return {
        "id": f"{src_name}_{sid}",
        "task_type": task_type,
        "images": images,
        "bboxes": unified_bboxes,
        "grounding_targets": grounding_targets,
        "conversations": norm_convs,
        "meta": {"src_dataset": src_name, "src_id": sid}
    }

def load_data(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        if path.suffix == ".jsonl":
            return [json.loads(line) for line in f if line.strip()]
        return json.load(f)

def infer_src_name(path: Path) -> str:
    return path.stem.split('.')[0]

def custom_convert(in_path: str, out_path: str, random_image_pos: bool = True, seed: int = 42, src_name: str | None = None, enable_bbox: bool = True):
    random.seed(seed)
    in_path, out_path = Path(in_path), Path(out_path)
    src_name = src_name or infer_src_name(in_path)
    
    print(f"🔄 {in_path} -> {out_path} | {src_name}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = load_data(in_path)
    
    vqa_only = is_vqa_only_dataset(src_name)
    print(f"🎯 VQA-only: {vqa_only}")
    
    stats = {"vqa": 0, "ground_rec": 0, "ground_region_caption": 0, "text_instruct": 0, "errors": 0}
    
    with out_path.open("w", encoding="utf-8") as f_out:
        for i, raw_item in enumerate(data):
            try:
                sample = build_unified_sample(raw_item, src_name, random_image_pos, enable_bbox and not vqa_only)
                f_out.write(json.dumps(sample, ensure_ascii=False) + "\n")
                stats[sample["task_type"]] += 1
                
                if i < 3:
                    phrase = sample['grounding_targets'][0].get('phrase', sample['grounding_targets'][0].get('caption', 'none'))[:30] + "..." if sample['grounding_targets'] else "none"
                    print(f"  {i}: {sample['task_type']} | bboxes={len(sample['bboxes'])} | phrase='{phrase}'")
            except Exception as e:
                stats["errors"] += 1
                print(f"  ❌ {raw_item.get('id', i)}: {e}")
    
    print(f"✅ Done! Stats: {stats}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified VLM dataset converter")
    parser.add_argument("--in", dest="in_path", required=True)
    parser.add_argument("--out", dest="out_path", required=True)
    parser.add_argument("--no-random-image-pos", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--src-name", type=str, default=None)
    parser.add_argument("--no-bbox", action="store_true")
    
    args = parser.parse_args()
    custom_convert(args.in_path, args.out_path, not args.no_random_image_pos, args.seed, args.src_name, not args.no_bbox)
