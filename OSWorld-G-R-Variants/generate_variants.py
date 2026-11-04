"""
Generate OSWorld-G-R-Variants: For each instruction in OSWorld-G_refined.json,
use an LLM to produce 3 rephrased variants while preserving the same semantic meaning
and target element.

Usage:
    python generate_variants.py \
        --input /path/to/osworld-g/OSWorld-G_refined.json \
        --output /path/to/osworld-g/OSWorld-G-R-Variants.json \
        --api_base <YOUR_API_BASE_URL> \
        --api_key <YOUR_API_KEY> \
        --model <MODEL_NAME> \
        --workers 8
"""

import json
import argparse
import time
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from tqdm import tqdm


REWRITE_PROMPT = """You are a professional instruction rewriter for GUI grounding benchmarks.

Given an original instruction that tells a user to click/interact with a specific UI element on a screenshot, generate exactly 3 different rephrased versions of this instruction.

Requirements:
1. Each variant MUST refer to the EXACT SAME UI element/location as the original.
2. Each variant should use DIFFERENT wording, sentence structure, or perspective.
3. Maintain the same level of specificity — don't add or remove critical spatial/visual details.
4. Keep instructions natural and fluent in English.
5. Variants should be diverse from each other (not just synonym substitution).

Strategies you can use:
- Use different spatial references ("top-left" → "upper-left corner")
- Emphasize different visual attributes (color, shape, icon, text, position)
- Change verb choice ("click" → "select" → "tap on" → "press")
- Reorder the descriptive clauses

Return your answer as a JSON array of exactly 3 strings, nothing else:
["variant1", "variant2", "variant3"]
"""

MAX_RETRIES = 5

# Thread-safe list for collecting failures
failed_items_lock = threading.Lock()
failed_items = []


def rewrite_instruction(client, model, instruction, item_id):
    """Call LLM to generate 3 variants of a single instruction. Retries up to 5 times."""
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": REWRITE_PROMPT},
                    {"role": "user", "content": f"Original instruction:\n{instruction}"}
                ],
                temperature=0.7,
                max_tokens=1024,
            )
            # Handle case where response might be a raw string
            if isinstance(response, str):
                content = response.strip()
            else:
                content = response.choices[0].message.content.strip()
            
            # Extract JSON array from response
            match = re.search(r'\[.*\]', content, re.DOTALL)
            if match:
                variants = json.loads(match.group())
                if isinstance(variants, list) and len(variants) == 3:
                    return variants, True
            
            last_error = f"Parse failed: response not valid JSON array of 3. Content: {content[:200]}"
            print(f"  [{item_id}] Parse failed (attempt {attempt+1}/{MAX_RETRIES}), retrying...")
            
        except Exception as e:
            last_error = str(e)
            print(f"  [{item_id}] API error (attempt {attempt+1}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
    
    # All 5 retries failed — record failure
    failure_record = {
        "id": item_id,
        "instruction": instruction,
        "error": last_error,
        "retries": MAX_RETRIES
    }
    with failed_items_lock:
        failed_items.append(failure_record)
    
    print(f"  WARNING: All {MAX_RETRIES} retries failed for [{item_id}]: {instruction[:60]}...")
    return [instruction, instruction, instruction], False


def process_item(args):
    """Process a single item: rewrite its instruction into 3 variants."""
    idx, item, client, model = args
    variants, success = rewrite_instruction(client, model, item["instruction"], item["id"])
    return idx, variants, success


def main():
    parser = argparse.ArgumentParser(description="Generate instruction variants for OSWorld-G-R")
    parser.add_argument("--input", type=str, required=True, help="Path to OSWorld-G_refined.json")
    parser.add_argument("--output", type=str, required=True, help="Output path for variants JSON")
    parser.add_argument("--api_base", type=str, required=True, help="LLM API base URL")
    parser.add_argument("--api_key", type=str, required=True, help="LLM API key")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers")
    parser.add_argument("--batch_size", type=int, default=0, 
                        help="If >0, only process first N items (for testing)")
    args = parser.parse_args()

    # Load original annotations
    with open(args.input, "r") as f:
        data = json.load(f)
    
    if args.batch_size > 0:
        data = data[:args.batch_size]
    
    print(f"Loaded {len(data)} items. Generating 3 variants each...")
    print(f"Model: {args.model}")
    print(f"Workers: {args.workers}")
    print(f"Max retries per item: {MAX_RETRIES}")
    
    client = OpenAI(base_url=args.api_base, api_key=args.api_key)
    
    # Generate variants in parallel
    variants_map = {}  # idx -> [3 variants]
    success_count = 0
    fail_count = 0
    
    tasks = [(i, item, client, args.model) for i, item in enumerate(data)]
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_item, task): task[0] for task in tasks}
        
        with tqdm(total=len(data), desc="Generating variants") as pbar:
            for future in as_completed(futures):
                idx, variants, success = future.result()
                variants_map[idx] = variants
                if success:
                    success_count += 1
                else:
                    fail_count += 1
                pbar.update(1)
    
    # Build output: for each original item, create 3 new items with variant instructions
    output_data = []
    for i, item in enumerate(data):
        variants = variants_map[i]
        for v_idx, variant_instruction in enumerate(variants):
            new_item = item.copy()
            new_item["id"] = f"{item['id']}-v{v_idx}"
            new_item["instruction"] = variant_instruction
            new_item["original_instruction"] = item["instruction"]
            new_item["variant_index"] = v_idx
            output_data.append(new_item)
    
    # Save output
    with open(args.output, "w") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    # Save failure log
    failure_log_path = args.output.replace(".json", "_failures.json")
    with open(failure_log_path, "w") as f:
        json.dump(failed_items, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*40}")
    print(f"  GENERATION COMPLETE")
    print(f"{'='*40}")
    print(f"Total items: {len(data)}")
    print(f"Success: {success_count}")
    print(f"Failed (all {MAX_RETRIES} retries exhausted): {fail_count}")
    print(f"Output variants: {len(output_data)} ({len(data)} × 3)")
    print(f"Saved to: {args.output}")
    if fail_count > 0:
        print(f"Failure log: {failure_log_path}")
        print(f"  ⚠️  {fail_count} items used original instruction as fallback!")
    else:
        print(f"No failures! 🎉")
    print(f"{'='*40}")


if __name__ == "__main__":
    main()
