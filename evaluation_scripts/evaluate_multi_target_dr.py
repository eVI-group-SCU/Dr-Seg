import argparse
import ast
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from datasets import load_dataset, load_from_disk
from PIL import Image as PILImage
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from qwen_vl_utils import process_vision_info
from sam2.sam2_image_predictor import SAM2ImagePredictor

# 默认输入会被缩放到该尺寸后再喂给模型
DEFAULT_RESIZE = 840

# QUESTION_TEMPLATE = (
#     "Please find \"{Question}\" with bboxs and points."
#     "Compare the difference between object(s) and find the most closely matched object(s)."
#     "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags."
#     "Output the bbox(es) and point(s) inside the interested object(s) in JSON format."
#     "i.e., <think> thinking process here </think>"
#     "<answer>{Answer}</answer>"
# )
QUESTION_TEMPLATE = (
    "Please find \"{Question}\" with bboxs and points."
    "Compare the difference between object(s) and find the most closely matched object(s)."
    "Output the thinking process inside <think>...</think>."
    "Inside this reasoning, you must include one or more <look>...</look> blocks, enclosing the parts of the reasoning where you pay special attention to certain visual information."
    " Then, output the final answer inside <answer>...</answer>."
    " Output the bbox(es) and point(s) inside the interested object(s) in JSON format."
    " i.e., <think> [your reasoning text] <look> [your visual focus] </look> [more reasoning text] </think>"
    "<answer>{Answer}</answer>" 
)

EXAMPLE_ANSWER = (
    "[{\"bbox_2d\": [10,100,200,210], \"point_2d\": [30,110]}, {\"bbox_2d\": [225,296,706,786], \"point_2d\": [302,410]}]"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reasoning-model", dest="reasoning_model", type=str, default="qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--segmentation-model", dest="segmentation_model", type=str, default="facebook/sam2-hiera-large")
    parser.add_argument("--dataset", type=str, default="hao05/cocopan", help="HF 数据集名称或 load_from_disk 路径")
    parser.add_argument("--dataset-split", dest="dataset_split", type=str, default="test")
    parser.add_argument("--output-dir", dest="output_dir", type=str, default="./new_reasonseg_eval_results")
    parser.add_argument("--idx", type=int, default=0)
    parser.add_argument("--num-parts", dest="num_parts", type=int, default=1)
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=16)
    parser.add_argument("--resize", type=int, default=DEFAULT_RESIZE)
    parser.add_argument("--max-new-tokens", dest="max_new_tokens", type=int, default=3000)
    parser.add_argument("--sam-workers", dest="sam_workers", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", dest="top_p", type=float, default=1.0)
    parser.add_argument("--top-k", dest="top_k", type=int, default=0)
    parser.add_argument(
        "--processor-model",
        dest="processor_model",
        type=str,
        default=None,
        help="若推理 checkpoint 缺少 tokenizer 配置，可单独指定 Processor 模型路径",
    )
    return parser.parse_args()


def load_eval_dataset(dataset_path: str, split: str):
    if os.path.isdir(dataset_path):
        return load_from_disk(dataset_path)[split]
    return load_dataset(dataset_path, split=split)


def extract_answer_payload(text: str) -> Optional[str]:
    answer_match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL)
    if answer_match:
        return answer_match.group(1).strip()
    return text.strip() if text.strip() else None


def extract_first_json_block(payload: str) -> Optional[str]:
    start = payload.find("[")
    if start == -1:
        return None
    depth = 0
    for idx in range(start, len(payload)):
        char = payload[idx]
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                return payload[start:idx + 1]
    return None


def parse_structured_answer(raw_text: str) -> List[Dict[str, Any]]:
    payload = extract_answer_payload(raw_text)
    if not payload:
        return []
    block = extract_first_json_block(payload)
    if not block:
        return []
    try:
        return json.loads(block)
    except Exception:
        try:
            parsed = ast.literal_eval(block)
        except Exception:
            return []
        if isinstance(parsed, list):
            return parsed
    return []


def batch_to_records(batch: Any) -> List[Dict[str, Any]]:
    """将 HuggingFace Dataset 的批量切片转换为逐样本字典列表。"""
    if isinstance(batch, dict):
        if not batch:
            return []
        length = len(next(iter(batch.values()), []))
        return [{key: value[idx] for key, value in batch.items()} for idx in range(length)]
    if isinstance(batch, list):
        return batch
    # Dataset 切片可能返回 Dataset 对象，可直接迭代
    try:
        return list(batch)
    except TypeError:
        return []


def load_processor(model_path: str, padding_side: str, fallback_model: Optional[str] = None) -> AutoProcessor:
    candidates: List[str] = []
    if model_path:
        candidates.append(model_path)
    if fallback_model:
        for cand in fallback_model.split("::"):
            cand = cand.strip()
            if cand and cand not in candidates:
                candidates.append(cand)

    errors: List[str] = []
    for candidate in candidates:
        for force_download in (False, True):
            try:
                return AutoProcessor.from_pretrained(
                    candidate,
                    padding_side=padding_side,
                    trust_remote_code=True,
                    force_download=force_download,
                )
            except json.JSONDecodeError as exc:
                errors.append(f"{candidate} (force_download={force_download}): {exc}")
                continue
            except ValueError as exc:
                if "Expecting value" in str(exc):
                    errors.append(f"{candidate} (force_download={force_download}): {exc}")
                    continue
                raise
    raise RuntimeError(
        "AutoProcessor 加载失败，请检查模型权重或通过 --processor-model 指定可用路径。尝试记录: "
        + "; ".join(errors)
    )


def to_int(value: Any) -> Optional[int]:
    try:
        return int(round(float(value)))
    except Exception:
        return None


def scale_bbox(
    bbox: Sequence[Any],
    width: int,
    height: int,
    x_factor: float,
    y_factor: float,
) -> Optional[List[int]]:
    if len(bbox) != 4:
        return None
    if any(to_int(v) is None for v in bbox):
        return None
    floats = np.array([float(v) for v in bbox], dtype=np.float64)
    normalized = np.max(np.abs(floats)) <= 1.5
    if normalized:
        x_scale = width
        y_scale = height
    else:
        x_scale = x_factor
        y_scale = y_factor
    x1 = int(np.clip(round(float(bbox[0]) * x_scale), 0, width - 1))
    y1 = int(np.clip(round(float(bbox[1]) * y_scale), 0, height - 1))
    x2 = int(np.clip(round(float(bbox[2]) * x_scale), 0, width - 1))
    y2 = int(np.clip(round(float(bbox[3]) * y_scale), 0, height - 1))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    if x1 == x2 or y1 == y2:
        return None
    return [x1, y1, x2, y2]


def scale_point(
    point: Sequence[Any],
    width: int,
    height: int,
    x_factor: float,
    y_factor: float,
) -> Optional[List[int]]:
    if len(point) != 2:
        return None
    if any(to_int(v) is None for v in point):
        return None
    floats = np.array([float(v) for v in point], dtype=np.float64)
    normalized = np.max(np.abs(floats)) <= 1.5
    if normalized:
        px = int(np.clip(round(float(point[0]) * width), 0, width - 1))
        py = int(np.clip(round(float(point[1]) * height), 0, height - 1))
    else:
        px = int(np.clip(round(float(point[0]) * x_factor), 0, width - 1))
        py = int(np.clip(round(float(point[1]) * y_factor), 0, height - 1))
    return [px, py]



def encode_mask_to_rle(mask: np.ndarray) -> Dict[str, Any]:
    h, w = mask.shape
    flat = mask.flatten(order="F")
    counts: List[int] = []
    run = 0
    last_val = 0
    for val in flat:
        if val == last_val:
            run += 1
        else:
            counts.append(run)
            run = 1
            last_val = int(val)
    counts.append(run)
    if last_val == 1:
        counts.append(0)
    return {"size": [h, w], "counts": counts}


def compute_iou(pred: np.ndarray, gt: np.ndarray) -> Tuple[int, int]:
    pred_bool = pred.astype(bool)
    gt_bool = gt.astype(bool)
    intersection = int(np.logical_and(pred_bool, gt_bool).sum())
    union = int(np.logical_or(pred_bool, gt_bool).sum())
    return intersection, union


def run_sam_parallel(
    predictor: SAM2ImagePredictor,
    prompts: List[Tuple[List[int], List[int]]],
    workers: int,
) -> List[np.ndarray]:
    def _run_single(bbox_point: Tuple[List[int], List[int]]) -> np.ndarray:
        bbox, point = bbox_point
        masks, scores, _ = predictor.predict(
            point_coords=[point],
            point_labels=[1],
            box=bbox,
        )
        best = int(np.argmax(scores))
        return masks[best].astype(bool)

    if not prompts:
        return []
    workers = max(1, min(workers, len(prompts)))
    if workers == 1:
        return [_run_single(bp) for bp in prompts]
    futures = {}
    results: List[Optional[np.ndarray]] = [None] * len(prompts)
    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for idx, prompt in enumerate(prompts):
                futures[executor.submit(_run_single, prompt)] = idx
            for future in as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()
    except Exception:
        return [_run_single(bp) for bp in prompts]
    return [mask for mask in results if mask is not None]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)
    jsonl_path = os.path.join(args.output_dir, f"predictions_{args.idx}.jsonl")
    summary_path = os.path.join(args.output_dir, f"output_{args.idx}.json")

    reasoning_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.reasoning_model,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    segmentation_model = SAM2ImagePredictor.from_pretrained(args.segmentation_model)
    processor_fallback = args.processor_model or "Ricky06662/Seg-Zero-7B"
    processor = load_processor(
        args.reasoning_model,
        padding_side="left",
        fallback_model=processor_fallback,
    )

    dataset = load_eval_dataset(args.dataset, args.dataset_split)
    total_len = len(dataset)
    part_size = total_len // args.num_parts
    start_idx = args.idx * part_size
    end_idx = start_idx + part_size if args.idx < args.num_parts - 1 else total_len
    dataset = dataset.select(range(start_idx, end_idx))

    resize_size = args.resize
    reasoning_model.eval()

    records: List[Dict[str, Any]] = []
    giou_sum = 0.0
    giou_count = 0

    with open(jsonl_path, "w", encoding="utf-8") as jsonl_file:
        for batch_start in tqdm(range(0, len(dataset), args.batch_size), desc="batches"):
            raw_batch = dataset[batch_start: batch_start + args.batch_size]
            batch = batch_to_records(raw_batch)
            if not batch:
                continue
            messages = []
            metas = []
            for item in batch:
                if not isinstance(item, dict):
                    continue
                image: PILImage.Image = item["image"].convert("RGB")
                resized_image = image.resize((resize_size, resize_size), PILImage.BILINEAR)
                text_prompt = QUESTION_TEMPLATE.format(
                    Question=(item.get("text") or "").lower().strip(".\"?!"),
                    Answer=EXAMPLE_ANSWER,
                )
                message = [{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": resized_image},
                        {"type": "text", "text": text_prompt},
                    ],
                }]
                messages.append(message)
                metas.append({
                    "image": image,
                    "image_id": item.get("image_id"),
                    "ann_id": item.get("ann_id"),
                    "mask": np.asarray(item["mask"], dtype=bool),
                    "height": image.height,
                    "width": image.width,
                })

            texts = [
                processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                for msg in messages
            ]
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to("cuda")

            gen_kwargs = {
                "use_cache": True,
                "max_new_tokens": args.max_new_tokens,
                "do_sample": args.temperature > 0,
            }
            if args.temperature > 0:
                gen_kwargs.update({"temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k})

            generated = reasoning_model.generate(**inputs, **gen_kwargs)
            trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated)]
            outputs = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

            del inputs, generated, trimmed
            torch.cuda.empty_cache()

            for idx, raw_answer in enumerate(outputs):
                meta = metas[idx]
                width, height = meta["width"], meta["height"]
                x_factor = width / resize_size
                y_factor = height / resize_size
                gt_mask = meta["mask"]
                errors: List[str] = []
                predictor_prompts: List[Tuple[List[int], List[int]]] = []

                parsed = parse_structured_answer(raw_answer)
                if not isinstance(parsed, list):
                    parsed = []
                for entry in parsed:
                    bbox = entry.get("bbox_2d") if isinstance(entry, dict) else None
                    point = entry.get("point_2d") if isinstance(entry, dict) else None
                    scaled_bbox = scale_bbox(bbox or [], width, height, x_factor, y_factor) if bbox else None
                    scaled_point = scale_point(point or [], width, height, x_factor, y_factor) if point else None
                    if scaled_bbox and scaled_point:
                        predictor_prompts.append((scaled_bbox, scaled_point))
                    else:
                        errors.append("invalid_prompt")

                pred_mask = np.zeros((height, width), dtype=bool)
                if predictor_prompts:
                    try:
                        segmentation_model.set_image(meta["image"])
                        masks = run_sam_parallel(segmentation_model, predictor_prompts, args.sam_workers)
                        for mask in masks:
                            if mask.shape != pred_mask.shape:
                                mask = np.array(mask, dtype=bool)
                                mask = mask[:height, :width]
                            pred_mask = np.logical_or(pred_mask, mask)
                    except Exception as exc:
                        errors.append(f"sam_error:{exc}")
                        pred_mask = np.zeros((height, width), dtype=bool)
                else:
                    errors.append("empty_prompt")

                intersection, union = compute_iou(pred_mask, gt_mask)
                iou = float(intersection / union) if union > 0 else 0.0
                if union > 0:
                    giou_sum += iou
                    giou_count += 1

                record = {
                    "image_id": meta["image_id"],
                    "ann_id": meta["ann_id"],
                    "raw_answer": raw_answer,
                    "predicted_mask_rle": encode_mask_to_rle(pred_mask),
                    "intersection": intersection,
                    "union": union,
                    "iou": iou,
                    "errors": errors,
                }
                jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                records.append({k: record[k] for k in ("image_id", "ann_id", "intersection", "union", "iou")})

    avg_giou = giou_sum / giou_count if giou_count else 0.0
    print(f"Average gIoU: {avg_giou:.4f}")

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
