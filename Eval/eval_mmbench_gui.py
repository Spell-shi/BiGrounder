import copy
import itertools
import torch
import json
import re
import argparse
import os
from PIL import Image
import logging
from tqdm import tqdm
import multiprocessing as mp
mp.set_start_method('spawn', force=True)

from model_factory import build_model

logging.basicConfig(level=logging.INFO)
torch.manual_seed(114514)

GT_TYPES = ['positive', 'negative']
INSTRUCTION_STYLES = ['instruction', 'action', 'description']
LANGUAGES = ['en', 'cn']

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, required=True)
    parser.add_argument('--model_name_or_path', type=str, required=False)
    parser.add_argument('--images_dir', type=str, required=True,
                       help="Path to the images directory")
    parser.add_argument('--annotations_file', type=str, required=True,
                       help="Path to the annotations JSON file")
    parser.add_argument('--gt_type', type=str, default='positive', choices=GT_TYPES)
    parser.add_argument('--log_path', type=str, required=True)
    parser.add_argument('--num_gpu', type=int, default=8)
    args = parser.parse_args()
    return args

def initializer(gpu_queue, args):
    """Initialize each worker process with a unique GPU and load the model."""
    gpu_index = gpu_queue.get()
    device = f"cuda:{gpu_index}"
    torch.cuda.set_device(gpu_index)
    model = build_model(args, device=device)

    global worker_model
    worker_model = model

def worker_function(sample):
    """Process a single task on the assigned GPU."""
    global worker_model

    # Build complete image path
    img_path = os.path.join(sample["images_dir"], sample["platform"], sample["image_path"])

    # Execute grounding task
    if sample["gt_type"] == "positive":
        response = worker_model.ground_only_positive(instruction=sample["instruction"], image=img_path)
    elif sample["gt_type"] == "negative":
        response = worker_model.ground_allow_negative(instruction=sample["instruction"], image=img_path)

    point = response["point"]
    img_size = sample["image_size"]
    point_in_pixel = [point[0] * img_size[0], point[1] * img_size[1]] if point else None

    sample_result = {
        "img_path": img_path,
        "index": sample["index"],
        "platform": sample["platform"],
        "application": sample["app_name"],
        "instruction": sample["instruction"],
        "gt_type": sample["gt_type"],
        "ui_type": sample["data_type"],
        "grounding_type": sample.get("grounding_type", "basic"),
        "pred": point_in_pixel,
        "raw_response": response["raw_response"]
    }

    # Print results
    print("\n" + "="*80)
    print(f"Instruction: {sample['instruction']}")

    if sample["gt_type"] == "positive":
        correctness = eval_sample_positive_gt(sample, response)
        sample_result.update({"bbox": sample["bbox"]})

        print(f"Prediction (Pixel): {point_in_pixel}")
        print(f"Correct Answer (BBox): {sample['bbox']}")
        print(f"Result: {correctness.upper()}")

    elif sample["gt_type"] == "negative":
        correctness = eval_sample_negative_gt(sample, response)
        print(f"Result: {correctness.upper()}")

    sample_result.update({"correctness": correctness})
    return sample_result

def eval_sample_positive_gt(sample, response):
    bbox = sample["bbox"]
    # bbox is already normalized [x_min, y_min, x_max, y_max]
    click_point = response["point"]

    if click_point is None:
        return "wrong_format"

    # Check if point is within bbox
    if (bbox[0] <= click_point[0] <= bbox[2]) and (bbox[1] <= click_point[1] <= bbox[3]):
        return "correct"
    else:
        return "wrong"

def eval_sample_negative_gt(sample, response):
    if response["result"] == "negative":
        return "correct"
    elif response["result"] == "positive":
        return "wrong"
    else:
        return "wrong_format"

def collect_results_to_eval(results, platform=None, application=None, ui_type=None, grounding_type=None):
    filtered_results = []
    for sample in results:
        if (platform is None or sample.get("platform") == platform) and \
           (application is None or sample.get("application") == application) and \
           (ui_type is None or sample.get("ui_type") == ui_type) and \
           (grounding_type is None or sample.get("grounding_type") == grounding_type):
            filtered_results.append(sample)
    return filtered_results

def calc_metric_for_result_list(results):
    num_total = len(results)
    correct_num = sum(1 for res in results if res["correctness"] == "correct")
    wrong_format_num = sum(1 for res in results if res["correctness"] == "wrong_format")

    text_results = collect_results_to_eval(results, ui_type="text")
    icon_results = collect_results_to_eval(results, ui_type="icon")
    text_correct = sum(1 for res in text_results if res["correctness"] == "correct")
    text_total = len(text_results)
    icon_correct = sum(1 for res in icon_results if res["correctness"] == "correct")
    icon_total = len(icon_results)

    metrics = {
        "num_correct": correct_num,
        "num_total": num_total,
        "wrong_format_num": wrong_format_num,
        "accuracy": correct_num / num_total if num_total > 0 else 0,
        "text_acc": text_correct / text_total if text_total > 0 else 0,
        "icon_acc": icon_correct / icon_total if icon_total > 0 else 0
    }
    return metrics

def evaluate_by_platform(results):
    evaluation_result = {}
    platforms = set(sample.get("platform") for sample in results)

    for platform in platforms:
        filtered_results = collect_results_to_eval(results, platform=platform)
        metrics = calc_metric_for_result_list(filtered_results)
        if metrics['num_total'] > 0:
            evaluation_result[f"platform:{platform}"] = metrics
    return evaluation_result

def evaluate_by_application(results):
    evaluation_result = {}
    applications = set(sample.get("application") for sample in results)

    for application in applications:
        filtered_results = collect_results_to_eval(results, application=application)
        metrics = calc_metric_for_result_list(filtered_results)
        if metrics['num_total'] > 0:
            evaluation_result[f"app:{application}"] = metrics
    return evaluation_result

def evaluate_by_grounding_type(results):
    evaluation_result = {}
    grounding_types = set(sample.get("grounding_type") for sample in results)

    for grounding_type in grounding_types:
        filtered_results = collect_results_to_eval(results, grounding_type=grounding_type)
        metrics = calc_metric_for_result_list(filtered_results)
        if metrics['num_total'] > 0:
            evaluation_result[f"grounding:{grounding_type}"] = metrics
    return evaluation_result

def evaluate_overall(results):
    return calc_metric_for_result_list(results)

def evaluate(results):
    result_report = {"details": [], "metrics": {}}
    result_report["metrics"]["overall"] = evaluate_overall(results)
    result_report["metrics"]["by_platform"] = evaluate_by_platform(results)
    result_report["metrics"]["by_application"] = evaluate_by_application(results)
    result_report["metrics"]["by_grounding_type"] = evaluate_by_grounding_type(results)
    result_report["details"] = results
    return result_report

def main(args):
    # Load annotations
    with open(args.annotations_file, 'r') as f:
        all_annotations = json.load(f)

    print(f"Loaded {len(all_annotations)} annotations from {args.annotations_file}")

    tasks_to_run = []
    for annotation in all_annotations:
        task_instance = copy.deepcopy(annotation)
        task_instance["gt_type"] = args.gt_type
        task_instance["images_dir"] = args.images_dir
        tasks_to_run.append(task_instance)

    print(f"Total tasks to run: {len(tasks_to_run)}")

    # Create GPU queue
    gpu_queue = mp.Queue()
    for i in range(args.num_gpu):
        gpu_queue.put(i)

    # Process tasks in parallel
    with mp.Pool(processes=args.num_gpu, initializer=initializer, initargs=(gpu_queue, args)) as pool:
        results = []
        for result in tqdm(pool.imap_unordered(worker_function, tasks_to_run), total=len(tasks_to_run)):
            results.append(result)

    # Evaluate and save results
    result_report = evaluate(results)
    os.makedirs(os.path.dirname(args.log_path), exist_ok=True)
    with open(args.log_path, 'w') as f:
        json.dump(result_report, f, indent=4)
    logging.info("Evaluation of MMBench-GUI finished.")

if __name__ == "__main__":
    main(parse_args())


# python eval_mmbench_gui.py \
#     --model_type BiGrounder \
#     --model_name_or_path /path/to/checkpoint \
#     --images_dir /path/to/mmbench_gui/images \
#     --annotations_file /path/to/mmbench_gui/annotations.json \
#     --log_path /path/to/output/mmbench_gui_eval.json \
#     --gt_type positive \
#     --num_gpu 8
