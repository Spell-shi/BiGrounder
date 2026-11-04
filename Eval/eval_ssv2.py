import os, json, argparse
import math, torch, base64, numpy as np, re, hashlib
import torch.multiprocessing as mp
from pathlib import Path
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer
from tqdm import tqdm
from PIL import Image

system_prompt = "You are a helpful assistant."

# ----------------- Parsing helpers -----------------
def extract_user_instruction_from_user_prompt(user_content: str) -> str:
    if not isinstance(user_content, str): return ""
    s = user_content.replace("<image>", "").strip()
    m = re.search(r"##\s*User Instruction\s*(.+)", s, flags=re.I)
    return (m.group(1).strip() if m else s)

def stable_uid_from_simplified(item: dict) -> str:
    imgs = item.get("images") or []
    user_text = ""
    for m in item.get("messages", []):
        if m.get("role") == "user":
            user_text = m.get("content") or ""
            break
    instr = extract_user_instruction_from_user_prompt(user_text)
    img_basename = os.path.basename(imgs[0]) if imgs else ""
    unique_string = f"{img_basename}::{instr}"
    return hashlib.sha1(unique_string.encode('utf-8')).hexdigest()

# Pattern that matches (number, number)
RE_POINT = re.compile(r"\(\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\)")

def parse_xy_from_action(action_text: str):
    """
    Parse coordinates from the model output.
    Note: the parsed values are the raw numbers (i.e. normalized to the 0-1000 range).
    """
    if not isinstance(action_text, str): return None
    all_matches = RE_POINT.findall(action_text)
    if not all_matches: return None
    try:
        last_match = all_matches[-1]
        x = float(last_match[0])
        y = float(last_match[1])
        return x, y  # return floats for downstream computation
    except (ValueError, IndexError):
        return None

# ----------------- Core evaluation logic (single-GPU worker) -----------------
def evaluate_on_subset(
    dataset_or_records,
    output_path,
    device=0,
    model_path=None,
    task_history_path=None,  # accepted for external-call compatibility
    subtask_map=None,        # accepted for external-call compatibility
    **kwargs                 # accept any other redundant arguments
):
    assert model_path, "model_path must be provided"

    torch.cuda.set_device(device)

    # Load the BiGrounder model (built on Qwen3-VL)
    print(f"Loading model from {model_path}...")
    model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            trust_remote_code=True
        ).to(device).eval()

    # AutoProcessor handles min_pixels / max_pixels and other config automatically.
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    eval_json = []
    tot = hit = 0
    pbar = tqdm(dataset_or_records, total=len(dataset_or_records), desc=f"eval → {os.path.basename(output_path)}")

    for item in pbar:
        uid = item.get("action_uid") or stable_uid_from_simplified(item)
        img_path = (item.get("images") or [""])[0]

        if not img_path or not os.path.exists(img_path):
            raise FileNotFoundError(f"uid={uid}: image not found: {img_path}")

        # Prepare the image
        image = Image.open(img_path).convert("RGB")
        img_w, img_h = image.size  # real image width/height

        # Prepare the prompt
        user_text = ""
        for m in item.get("messages", []):
            if m.get("role") == "user":
                user_text = m.get("content") or ""
                break

        instr = extract_user_instruction_from_user_prompt(user_text)
        prompt = 'You are a helpful assistant.' + user_text
        print(" ".join(prompt.split()[:20]))

        # Build messages (standard Qwen-VL format)
        messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text",  "text": prompt},
            ]
        }]

        # Prepare inputs
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt").to(device)

        # Inference
        with torch.inference_mode():
            gen_ids = model.generate(**inputs, max_new_tokens=128)

        # Decode
        gen_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, gen_ids)
        ]
        output_text = processor.batch_decode(
            gen_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        action = output_text.split("Action:")[-1].strip()

        print(f"Output: {action}")

        # --- Handle normalized coordinates ---
        pt_norm = parse_xy_from_action(action)  # coordinates in the 0-1000 range
        pt_abs = None  # absolute coordinates

        if pt_norm is None:
            tqdm.write(f"[WARN] uid={uid} - Failed to parse: '{action}'")
        else:
            # Denormalize (0-1000 -> absolute pixels): abs = (norm / 1000) * size
            x_norm, y_norm = pt_norm
            x_abs = (x_norm / 1000.0) * img_w
            y_abs = (y_norm / 1000.0) * img_h
            pt_abs = (int(x_abs), int(y_abs))

        bbox = item.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError(f"uid={uid}: expected bbox with four values, got {bbox!r}")

        x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1

        # Every valid benchmark sample contributes to the denominator. A response
        # that cannot be parsed is an incorrect prediction, not a skipped sample.
        tot += 1
        if pt_abs is None:
            hit_flag = False
            tqdm.write(f"[cmp] uid={uid} parse_failed bbox=[{x1},{y1},{x2},{y2}] hit=False")
        else:
            x, y = pt_abs
            hit_flag = (x1 <= x <= x2) and (y1 <= y <= y2)
            tqdm.write(
                f"[cmp] uid={uid} Norm({pt_norm[0]},{pt_norm[1]}) "
                f"-> Abs({x},{y}) bbox=[{x1},{y1},{x2},{y2}] hit={hit_flag}"
            )

        hit += int(hit_flag)
        pbar.set_postfix(hits=f"{hit}/{tot}", acc=f"{hit / tot:.2%}")

        result_item = {
            "uid": uid,
            "action_raw": action,  # raw output
            "point_norm": pt_norm,  # normalized coordinates
            "point_abs": pt_abs,    # absolute coordinates
            "images": item.get("images"),
            "bbox": bbox,
            "hit": hit_flag
        }
        eval_json.append(result_item)

        # Free GPU memory
        del inputs, gen_ids, gen_ids_trimmed
        torch.cuda.empty_cache()

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(eval_json, f, indent=2, ensure_ascii=False)

    if tot > 0:
        print(f"\n==== Click-in-BBox Accuracy ====")
        print(f"evaluated: {tot} | hits: {hit} | acc: {hit/tot:.2%}")


def summarize_results(results):
    """Compute accuracy with parse failures included as incorrect predictions."""
    total = len(results)
    hits = sum(item.get("hit") is True for item in results)
    parse_failures = sum(item.get("point_abs") is None for item in results)
    return {
        "evaluated": total,
        "hits": hits,
        "parse_failures": parse_failures,
        "accuracy": hits / total if total else 0.0,
    }

# ----------------- Multi-GPU launcher -----------------
def run_worker(rank, gpu_id, world_size, args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    data = json.load(open(args.simplified_json, "r", encoding="utf-8"))

    if args.sample_every and args.sample_every > 1:
        data = data[::args.sample_every]
        print(f"[rank {rank}] subsample every {args.sample_every} → {len(data)} records")

    total = len(data)
    print(total)
    chunk = total // world_size
    start = rank * chunk
    end   = total if rank == world_size - 1 else (rank + 1) * chunk
    records = data[start:end]

    out_file = os.path.join(args.out_dir, f"eval_result_rank{rank}.json")
    evaluate_on_subset(
        records,
        out_file,
        device=0,                 # pinned to this process's GPU via CUDA_VISIBLE_DEVICES
        task_history_path=None,   # compatibility placeholder
        subtask_map=None,         # compatibility placeholder
        model_path=args.model_path
    )
    print(f"[rank {rank}] finished, wrote {out_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--simplified-json", required=True,
                        help="JSON containing messages/images/bbox")
    parser.add_argument("--model-path", required=True,
                        help="Model path or hub name")
    parser.add_argument("--out-dir", default="./eval_output_from_json")
    parser.add_argument("--min-free", type=int, default=40,
                        help="Select GPUs with free memory >= this many GB")
    parser.add_argument("--sample-every", type=int, default=1,
                    help="Uniform subsampling step; 10 means take 1/10 (data[::10])")

    args = parser.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # Select available GPUs
    import pynvml
    pynvml.nvmlInit()
    avail = []
    for i in range(pynvml.nvmlDeviceGetCount()):
        mem = pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(i))
        if mem.free / 1024**3 >= args.min_free:
            avail.append(i)
    pynvml.nvmlShutdown()

    if not avail:
        raise RuntimeError("No GPU satisfies free-mem requirement")
    world_size = len(avail)
    print("Launch", world_size, "workers on", avail)

    mp.set_start_method("spawn", force=True)
    procs = []
    for rank, gpu in enumerate(avail):
        p = mp.Process(target=run_worker, args=(rank, gpu, world_size, args))
        p.start(); procs.append(p)
    for p in procs:
        p.join()

    failed_workers = [rank for rank, p in enumerate(procs) if p.exitcode != 0]
    if failed_workers:
        raise RuntimeError(f"Evaluation workers failed: {failed_workers}")

    merged = []
    for rank in range(world_size):
        fp = Path(args.out_dir) / f"eval_result_rank{rank}.json"
        if fp.exists():
            data_slice = json.load(open(fp, "r", encoding="utf-8"))
            if isinstance(data_slice, list):
                merged.extend(data_slice)

    expected_data = json.load(open(args.simplified_json, "r", encoding="utf-8"))
    if args.sample_every and args.sample_every > 1:
        expected_data = expected_data[::args.sample_every]
    if len(merged) != len(expected_data):
        raise RuntimeError(
            f"Incomplete evaluation: expected {len(expected_data)} results, got {len(merged)}"
        )

    with open(Path(args.out_dir) / "merged_output.json", "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    metrics = summarize_results(merged)
    with open(Path(args.out_dir) / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"merged {len(merged)} records → {args.out_dir}")
    print(
        "overall accuracy: "
        f"{metrics['hits']}/{metrics['evaluated']} = {metrics['accuracy']:.2%}; "
        f"parse failures: {metrics['parse_failures']}"
    )

if __name__ == "__main__":
    main()

# python eval_ssv2.py \
#     --simplified-json /path/to/ssv2_all_test.json \
#     --model-path /path/to/checkpoint \
#     --out-dir /path/to/output/eval_ssv2 \
#     --min-free 30 \
#     --sample-every 1 2>&1 | tee ssv2.txt
