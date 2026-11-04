"""
Evaluate BiGrounder on OSWorld-G with Pass@1 and Pass@4.

Uses the GUI agent prompt style with click(start_box='(x,y)') format.

Usage:
    python eval_osworld_g.py \
        --original /path/to/osworld-g/OSWorld-G_refined.json \
        --variants /path/to/osworld-g/OSWorld-G-R-Variants.json \
        --image_dir /path/to/osworld-g/images \
        --classification_path /path/to/osworld-g/classification_result.json \
        --model_path /path/to/checkpoint \
        --model_name bigrounder \
        --port 8908
"""

import json
import argparse
import time
import re
import math
import base64
import socket
import subprocess
import numpy as np
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

from PIL import Image
from openai import OpenAI
from tqdm import tqdm


# ================= Prompt =================
SYSTEM_PROMPT = "You are a helpful assistant."

USER_PROMPT_TEMP = """You are a helpful assistant,
    'You are a GUI agent. You are given a task, with screenshots. You need to perform the next action to complete the task. '
    '## Output Format '
    'Action: ... '
    '## Action Space '
    "click(start_box='(x1,y1)') "
    "rightClick(start_box='(x1,y1)')"
    "doubleClick(start_box='(x1,y1)') "
    "select(start_box='(x1,y1)') "
    "type(content='', start_box='(x1,y1)') "
    "hotkey(key='') "
    "stop(reason='') "
    "scrolldown() "
    "scrollup() "
    "scrollmenu(start_box='(x1,y1),(x2,y2)') "
    "finish() "
"""


def smart_resize(height, width, factor=28, min_pixels=56*56, max_pixels=1_000_000_000):
    if height < factor or width < factor:
        raise ValueError("Image too small.")
    if max(height, width) / min(height, width) > 200:
        raise ValueError("Aspect ratio too extreme.")
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def encode_image(image):
    target_h, target_w = smart_resize(image.height, image.width)
    if (image.width, image.height) != (target_w, target_h):
        image = image.resize((target_w, target_h), Image.Resampling.LANCZOS)
    buf = BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def parse_coordinates(response):
    """Parse (x, y) integer coordinates from the model output."""
    try:
        if "stop" in response.lower() or "finish" in response.lower():
            return [-1, -1, -1, -1]
        
        # Match (123, 456) pattern
        pattern = r"\((\d+)\s*,\s*(\d+)\)"
        matches = re.findall(pattern, response)
        
        if matches:
            last_match = matches[-1]
            x_int = int(last_match[0])
            y_int = int(last_match[1])
            return [x_int, y_int, x_int, y_int]
        
        return [0, 0, 0, 0]
    except Exception:
        return [0, 0, 0, 0]


def is_point_in_rectangle(point, rect):
    return rect[0] <= point[0] <= rect[2] and rect[1] <= point[1] <= rect[3]


def is_point_in_polygon(point, polygon):
    x, y = point
    n = len(polygon) // 2
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i*2], polygon[i*2+1]
        xj, yj = polygon[j*2], polygon[j*2+1]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def evaluate_prediction(raw_coords, item):
    """Check if prediction is correct for a given item."""
    img_w, img_h = item["image_size"]
    
    if raw_coords[0] == -1:
        predicted_coords = [-1, -1, -1, -1]
    else:
        # Convert 0-1000 coordinates to absolute pixel coordinates
        predicted_coords = [
            (raw_coords[0] / 1000.0) * img_w,
            (raw_coords[1] / 1000.0) * img_h,
            (raw_coords[2] / 1000.0) * img_w,
            (raw_coords[3] / 1000.0) * img_h
        ]
    
    center_x = (predicted_coords[0] + predicted_coords[2]) / 2
    center_y = (predicted_coords[1] + predicted_coords[3]) / 2
    center_point = [center_x, center_y]
    
    box_type = item["box_type"]
    box_coords = item["box_coordinates"]
    
    if box_type == "bbox":
        rect = [
            box_coords[0], box_coords[1],
            box_coords[0] + box_coords[2],
            box_coords[1] + box_coords[3]
        ]
        return is_point_in_rectangle(center_point, rect)
    elif box_type == "polygon":
        return is_point_in_polygon(center_point, box_coords)
    elif box_type == "refusal":
        return all(c < 0 for c in center_point)
    return False


def call_model(client, model_name, instruction, image_b64, max_retries=5):
    """Call the model and return the response text."""
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                        {"type": "text", "text": USER_PROMPT_TEMP + f"\n## User Instruction : {instruction}"}
                    ]}
                ],
                temperature=0,
                max_tokens=1024
            )
            if isinstance(completion, str):
                return completion
            return completion.choices[0].message.content
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)
            else:
                print(f"  Failed after {max_retries} retries: {e}")
                return ""
    return ""


def start_vllm_service(model_path, port, model_name):
    cmd = [
        "vllm", "serve", model_path,
        "--served-model-name", model_name,
        "--host", "0.0.0.0",
        "--port", str(port),
        "--tensor-parallel-size", "4",
        "--max-model-len", "32768",
        "--enforce-eager",
        "--limit-mm-per-prompt", '{"image": 1}',
        "--trust-remote-code"
    ]
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def wait_for_service(port, timeout=600):
    print(f"Waiting for vLLM service on port {port}...")
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if s.connect_ex(("localhost", port)) == 0:
                    print("Service is ready!")
                    return True
        except:
            pass
        time.sleep(2)
    return False


def terminate_service(proc):
    if proc:
        print("Terminating vLLM service...")
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", type=str, required=True, help="Path to OSWorld-G_refined.json")
    parser.add_argument("--variants", type=str, required=True, help="Path to OSWorld-G-R-Variants.json")
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--classification_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="bigrounder")
    parser.add_argument("--port", type=int, default=8908)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--output", type=str, default=None, help="Output path for eval results JSON")
    args = parser.parse_args()

    # Load data
    original_data = json.load(open(args.original))
    variants_data = json.load(open(args.variants))
    classification = json.load(open(args.classification_path))
    
    # Build combined test set
    all_instances = []
    
    # Add original items
    for item in original_data:
        inst = item.copy()
        inst["group_id"] = item["id"]
        inst["source"] = "original"
        all_instances.append(inst)
    
    # Add variant items
    for item in variants_data:
        inst = item.copy()
        inst["group_id"] = item["id"].rsplit("-v", 1)[0]
        inst["source"] = "variant"
        all_instances.append(inst)
    
    print(f"Total instances: {len(all_instances)} (original: {len(original_data)}, variants: {len(variants_data)})")
    print(f"Groups (unique samples): {len(original_data)}")
    print(f"Each group has 4 instructions (1 original + 3 variants)")
    
    # Start vLLM
    proc = start_vllm_service(args.model_path, args.port, args.model_name)
    
    try:
        if not wait_for_service(args.port):
            print("ERROR: vLLM service failed to start!")
            return
        
        client = OpenAI(base_url=f"http://localhost:{args.port}/v1", api_key="token-abc123")
        
        # Pre-encode images
        print("Pre-encoding images...")
        image_cache = {}
        import os
        unique_images = set(inst["image_path"] for inst in all_instances)
        for img_path in tqdm(unique_images, desc="Encoding images"):
            full_path = os.path.join(args.image_dir, img_path)
            image = Image.open(full_path)
            image_cache[img_path] = (encode_image(image), [image.width, image.height])
        
        # Run inference on all instances
        print(f"\nRunning inference on {len(all_instances)} instances...")
        results = [None] * len(all_instances)
        
        def process_instance(idx_inst):
            idx, inst = idx_inst
            img_b64, img_size = image_cache[inst["image_path"]]
            inst["image_size"] = img_size
            response = call_model(client, args.model_name, inst["instruction"], img_b64)
            raw_coords = parse_coordinates(response)
            is_correct = evaluate_prediction(raw_coords, inst)
            return idx, is_correct
        
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(process_instance, (i, inst)): i
                for i, inst in enumerate(all_instances)
            }
            with tqdm(total=len(all_instances), desc="Evaluating") as pbar:
                for future in as_completed(futures):
                    idx, is_correct = future.result()
                    results[idx] = is_correct
                    pbar.update(1)
        
        # Compute Pass@1 (overall accuracy on all instances)
        pass1_total = len(all_instances)
        pass1_correct = sum(1 for r in results if r)
        pass1_accuracy = pass1_correct / pass1_total
        
        # Compute Pass@4 (per-group: correct if ANY of 4 is correct)
        group_results = defaultdict(list)
        for i, inst in enumerate(all_instances):
            group_results[inst["group_id"]].append(results[i])
        
        pass4_total = len(group_results)
        pass4_correct = sum(1 for group_hits in group_results.values() if any(group_hits))
        pass4_accuracy = pass4_correct / pass4_total
        
        # Pass@1 on originals only
        orig_correct = sum(1 for i, inst in enumerate(all_instances) if inst["source"] == "original" and results[i])
        orig_accuracy = orig_correct / len(original_data)
        
        # Category breakdown
        cat_pass4 = defaultdict(lambda: {"total": 0, "correct": 0})
        
        for group_id, hits in group_results.items():
            cats = []
            for key, members in classification.items():
                if group_id in members:
                    cats.append(key)
            if not cats:
                cats = ["unknown"]
            
            any_correct = any(hits)
            for cat in cats:
                cat_pass4[cat]["total"] += 1
                if any_correct:
                    cat_pass4[cat]["correct"] += 1
        
        # === Variance Computation ===
        # 1. Per-group consistency score
        group_scores = []
        for group_id, hits in group_results.items():
            group_scores.append(sum(hits) / len(hits))
        group_scores = np.array(group_scores)
        
        # Variance of per-group scores
        variance_group = np.var(group_scores, ddof=0)
        std_group = np.std(group_scores, ddof=0)
        
        # 2. Bernoulli variance for Pass@1 (original only)
        var_pass1_orig = orig_accuracy * (1 - orig_accuracy)
        se_pass1_orig = math.sqrt(var_pass1_orig / len(original_data))
        
        # 3. Bernoulli variance for Pass@1 (all instances)
        var_pass1_all = pass1_accuracy * (1 - pass1_accuracy)
        se_pass1_all = math.sqrt(var_pass1_all / pass1_total)
        
        # 4. Bernoulli variance for Pass@4
        var_pass4 = pass4_accuracy * (1 - pass4_accuracy)
        se_pass4 = math.sqrt(var_pass4 / pass4_total)
        
        # Print results
        print("\n" + "=" * 60)
        print("           EVALUATION RESULTS")
        print("=" * 60)
        print(f"\n--- Pass@1 (Original only, {len(original_data)} samples) ---")
        print(f"Accuracy: {orig_accuracy*100:.2f}% ({orig_correct}/{len(original_data)})")
        print(f"Bernoulli Variance: {var_pass1_orig:.4f}")
        print(f"Standard Error: {se_pass1_orig:.4f} (95% CI: {(orig_accuracy-1.96*se_pass1_orig)*100:.2f}% ~ {(orig_accuracy+1.96*se_pass1_orig)*100:.2f}%)")
        print(f"\n--- Pass@1 (All {pass1_total} instances: original + variants) ---")
        print(f"Accuracy: {pass1_accuracy*100:.2f}% ({pass1_correct}/{pass1_total})")
        print(f"Bernoulli Variance: {var_pass1_all:.4f}")
        print(f"Standard Error: {se_pass1_all:.4f} (95% CI: {(pass1_accuracy-1.96*se_pass1_all)*100:.2f}% ~ {(pass1_accuracy+1.96*se_pass1_all)*100:.2f}%)")
        print(f"\n--- Pass@4 ({pass4_total} groups, any-of-4 correct) ---")
        print(f"Accuracy: {pass4_accuracy*100:.2f}% ({pass4_correct}/{pass4_total})")
        print(f"Bernoulli Variance: {var_pass4:.4f}")
        print(f"Standard Error: {se_pass4:.4f} (95% CI: {(pass4_accuracy-1.96*se_pass4)*100:.2f}% ~ {(pass4_accuracy+1.96*se_pass4)*100:.2f}%)")
        print(f"\n--- Robustness Variance (per-group consistency) ---")
        print(f"Mean group score: {np.mean(group_scores)*100:.2f}%")
        print(f"Variance (sigma^2): {variance_group:.4f}")
        print(f"Std Dev (sigma): {std_group:.4f}")
        print(f"  -> A group score of 1.0 means all 4 paraphrases correct; 0.0 means all wrong.")
        print(f"  -> Low variance = model is robust to instruction rephrasing.")
        print(f"  -> High variance = model is sensitive to how the instruction is worded.")
        
        print(f"\n--- Pass@4 Category Breakdown ---")
        for cat in sorted(cat_pass4.keys()):
            stats = cat_pass4[cat]
            acc = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
            print(f"  {cat:<25}: {acc:.2f}% ({stats['correct']}/{stats['total']})")
        
        print("=" * 60)
        
        # Save detailed results
        output_path = args.output if args.output else args.variants.replace(".json", "_eval_results.json")
        eval_output = {
            "prompt_style": "gui agent",
            "pass1_original": {"accuracy": orig_accuracy, "correct": orig_correct, "total": len(original_data),
                               "variance": var_pass1_orig, "standard_error": se_pass1_orig},
            "pass1_all": {"accuracy": pass1_accuracy, "correct": pass1_correct, "total": pass1_total,
                          "variance": var_pass1_all, "standard_error": se_pass1_all},
            "pass4": {"accuracy": pass4_accuracy, "correct": pass4_correct, "total": pass4_total,
                      "variance": var_pass4, "standard_error": se_pass4},
            "robustness": {"mean_group_score": float(np.mean(group_scores)),
                           "variance": float(variance_group), "std": float(std_group)},
            "pass4_categories": {k: v for k, v in cat_pass4.items()},
            "per_group": {gid: {"results": hits, "score": sum(hits)/len(hits), "pass4": any(hits)} for gid, hits in group_results.items()}
        }
        with open(output_path, "w") as f:
            json.dump(eval_output, f, indent=2)
        print(f"\nDetailed results saved to: {output_path}")
        
    finally:
        terminate_service(proc)


if __name__ == "__main__":
    main()
