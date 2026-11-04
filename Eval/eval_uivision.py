import copy
import numpy as np
import torch
import torch.multiprocessing as mp
import json
import argparse
import os
from tqdm import tqdm
import logging
import math

# Set the HF cache path (edit to your local cache directory)
os.environ['HF_HOME'] = '/path/to/hf_cache'

ANNOTATOR_IMAGE_SIZE = (800, 700)

logging.basicConfig(level=logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, required=True)
    parser.add_argument('--model_name_or_path', type=str, required=False)
    parser.add_argument('--uivision_imgs', type=str, required=True)
    parser.add_argument('--uivision_test_file', type=str, required=True,
                        help="Path to the test file containing the evaluation data")
    parser.add_argument('--task', type=str, required=True,
                        choices=['element', 'layout'],
                        help="Task type: 'element' or 'layout'")
    parser.add_argument('--log_path', type=str, required=True)
    parser.add_argument('--num_gpus', type=int, default=8, help="Number of GPUs to use")

    args = parser.parse_args()
    return args

def build_model(args):
    # This function is called inside each subprocess.
    model_type = args.model_type
    model_name_or_path = args.model_name_or_path

    # Model selection: load the corresponding model wrapper.
    if model_type == "BiGrounder":
        from models.bigrounder_uivision import BiGrounderModel

        model = BiGrounderModel()
        if args.model_name_or_path:
            model.load_model(model_name_or_path=model_name_or_path)
        else:
            model.load_model()
    else:
        raise ValueError(f"Unsupported model type {model_type} in worker.")

    model.set_generation_config(temperature=0, max_new_tokens=256)
    return model

# --- Evaluation Helper Functions ---

def calc_metric_for_result_list_simple(results):
    num_total = len(results)
    correct_num = sum(1 for res in results if res["correctness"] == "correct")
    wrong_format_num = sum(1 for res in results if res["correctness"] == "wrong_format")
    return {
        "num_correct_action": correct_num,
        "num_total": num_total,
        "wrong_format_num": wrong_format_num,
        "action_acc": correct_num / num_total if num_total > 0 else 0,
    }

def evaluate_overall_simple(results):
    metrics = calc_metric_for_result_list_simple(results)
    return metrics

def eval_sample_positive_gt(sample, response):
    bbox = sample["bbox"]
    bbox = [bbox[0], bbox[1], bbox[2], bbox[3]]
    if bbox[0] > bbox[2]: bbox[0], bbox[2] = bbox[2], bbox[0]
    if bbox[1] > bbox[3]: bbox[1], bbox[3] = bbox[3], bbox[1]
    img_size = sample["image_size"]
    bbox = [bbox[0] / img_size[0], bbox[1] / img_size[1], bbox[2] / img_size[0], bbox[3] / img_size[1]]

    try:
        click_point = response["point"]
    except:
        click_point = None
    if click_point is None:
        return "wrong_format"
    if (bbox[0] <= click_point[0] <= bbox[2]) and (bbox[1] <= click_point[1] <= bbox[3]):
        return "correct"
    else:
        return "wrong"

def evaluate_element(results):
    result_report = {"details": [], "metrics": {}}
    result_report["metrics"]["overall_simple"] = evaluate_overall_simple(results)
    result_report["details"] = results
    return result_report

def evaluate_single_bbox(pred_bbox, gt_bbox):
    def calculate_intersection_and_union(box1, box2):
        if not box1: return 0, 0, 0
        if box1[0] >= box1[2] or box1[1] >= box1[3]: return 0, 0, 0
        x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
        x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        return intersection, box1_area, box2_area

    intersection, pred_area, gt_area = calculate_intersection_and_union(pred_bbox, gt_bbox)
    union = pred_area + gt_area - intersection
    iou = intersection / union if union > 0 else 0
    precision = intersection / gt_area if gt_area > 0 else 0
    recall = intersection / pred_area if pred_area > 0 else 0
    return {"iou": iou, "precision": precision, "recall": recall, "intersection": intersection, "pred_area": pred_area, "gt_area": gt_area}

def eval_layout(gt_bbox, pred_bbox):
    return evaluate_single_bbox(pred_bbox, gt_bbox)

def evaluate_layout(results):
    result_report = {"details": [], "metrics": {}}
    iou_list = [sample["eval_results"]["iou"] for sample in results]
    precision_list = [sample["eval_results"]["precision"] for sample in results]
    recall_list = [sample["eval_results"]["recall"] for sample in results]

    result_report["metrics"]["mean_iou"] = np.mean(iou_list)
    result_report["metrics"]["mean_precision"] = np.mean(precision_list)
    result_report["metrics"]["mean_recall"] = np.mean(recall_list)
    result_report["metrics"]["std_iou"] = np.std(iou_list)
    result_report["metrics"]["std_precision"] = np.std(precision_list)
    result_report["metrics"]["std_recall"] = np.std(recall_list)
    result_report["details"] = results
    return result_report

# --- Core Inference Worker Function ---

def inference_worker(rank, args, data_subset, return_dict):
    """
    Worker function that runs inside each GPU process.
    """
    try:
        # 1. Key step: set the environment variable so this process only sees the
        # designated GPU. This ensures device_map="auto" in build_model uses only this card.
        os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)

        # Reset the seed
        torch.manual_seed(114514 + rank)

        print(f"[GPU {rank}] Loading model...")
        model = build_model(args)
        print(f"[GPU {rank}] Model loaded. Processing {len(data_subset)} samples.")

        local_results = []

        # Use tqdm; position avoids overlapping progress bars across processes.
        for sample in tqdm(data_subset, position=rank, desc=f"GPU {rank}"):
            img_path = os.path.join(args.uivision_imgs, sample["image_path"])

            # --- Element Task ---
            if args.task == "element":
                response = model.ground_only_positive(instruction=sample["prompt_to_evaluate"], image=img_path)

                try:
                    point = response["point"]
                except:
                    point = None

                img_size = sample["image_size"]
                point_in_pixel = [point[0] * img_size[0], point[1] * img_size[1]] if point else None

                try:
                    raw_response = response.get("raw_response", None)
                except:
                    raw_response = None

                sample_result = {
                    "image_path": img_path,
                    "platform": sample["platform"],
                    "prompt_to_evaluate": sample["prompt_to_evaluate"],
                    "ui_type": sample["element_type"],
                    "pred": point_in_pixel,
                    "raw_response": raw_response,
                    "ground_truth": sample.get("ground_truth", None),
                    "gt_bbox": sample.get("gt_bbox", None),
                    "bbox": sample["bbox"]
                }

                # Compute the correctness of this sample
                correctness = eval_sample_positive_gt(sample, response)
                sample_result["correctness"] = correctness
                local_results.append(sample_result)

            # --- Layout Task ---
            elif args.task == "layout":
                response = model.layout_gen(name=sample["name"], explanation=sample["explanation"], image=img_path)

                img_size = sample["image_size"]
                try:
                    box_in_pixel = [
                        response["bbox"][0] * img_size[0],
                        response["bbox"][1] * img_size[1],
                        response["bbox"][2] * img_size[0],
                        response["bbox"][3] * img_size[1]
                    ] if response["bbox"] else None
                except:
                    # print(f"[GPU {rank}] No box found")
                    box_in_pixel = None

                try:
                    raw_response = response.get("raw_response", None)
                except:
                    raw_response = None

                sample_result = {
                    "image_path": img_path,
                    "platform": sample["platform"],
                    "pred": box_in_pixel,
                    "raw_response": raw_response,
                    "bbox": sample["bbox"],
                    'image_size': sample['image_size'],
                    'name': sample['name'],
                    'explanation': sample['explanation'],
                }

                # Evaluate layout
                eval_results = eval_layout(copy.deepcopy(sample["bbox"]), copy.deepcopy(box_in_pixel))
                sample_result["eval_results"] = eval_results
                local_results.append(sample_result)

        # Store the results in the shared dict
        return_dict[rank] = local_results
        print(f"[GPU {rank}] Finished.")

    except Exception as e:
        print(f"[GPU {rank}] Error: {e}")
        import traceback
        traceback.print_exc()
        return_dict[rank] = {"error": repr(e)}
        raise

def main(args):
    # 1. Load data
    print(f"Loading data from {args.uivision_test_file}")
    with open(args.uivision_test_file, 'r') as f:
        tasks_data = json.load(f)

    # Full data list
    all_tasks = [copy.deepcopy(t) for t in tasks_data]
    total_samples = len(all_tasks)
    print(f"Total samples: {total_samples}")

    # 2. Data sharding
    num_gpus = args.num_gpus
    if num_gpus > total_samples:
        num_gpus = total_samples
        print(f"Warning: Num GPUs > Num Samples. Reduced to {num_gpus} GPUs.")

    chunk_size = math.ceil(total_samples / num_gpus)
    data_chunks = []
    for i in range(num_gpus):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, total_samples)
        if start_idx < end_idx:
            data_chunks.append(all_tasks[start_idx:end_idx])

    print(f"Splitting data into {len(data_chunks)} chunks for {num_gpus} GPUs.")

    # 3. Launch multiprocessing
    mp.set_start_method('spawn', force=True)  # spawn is required to start CUDA processes
    manager = mp.Manager()
    return_dict = manager.dict()
    processes = []

    for rank in range(len(data_chunks)):
        p = mp.Process(target=inference_worker, args=(rank, args, data_chunks[rank], return_dict))
        p.start()
        processes.append(p)

    # 4. Wait for all processes to finish
    for p in processes:
        p.join()

    failed_workers = [rank for rank, p in enumerate(processes) if p.exitcode != 0]
    if failed_workers:
        details = {
            rank: return_dict.get(rank, {"error": "worker exited without reporting an error"})
            for rank in failed_workers
        }
        raise RuntimeError(f"UI-Vision evaluation workers failed: {details}")

    # 5. Merge results
    print("Merging results...")
    all_results = []
    # Merge in rank order to keep the relative order (order usually does not matter for evaluation)
    for rank in range(len(data_chunks)):
        if rank not in return_dict:
            raise RuntimeError(f"No results returned from UI-Vision worker {rank}")
        worker_results = return_dict[rank]
        if not isinstance(worker_results, list):
            raise RuntimeError(f"Invalid result from UI-Vision worker {rank}: {worker_results!r}")
        expected_worker_results = len(data_chunks[rank])
        if len(worker_results) != expected_worker_results:
            raise RuntimeError(
                f"Incomplete results from UI-Vision worker {rank}: "
                f"expected {expected_worker_results}, got {len(worker_results)}"
            )
        all_results.extend(worker_results)

    if len(all_results) != total_samples:
        raise RuntimeError(
            f"Incomplete UI-Vision evaluation: expected {total_samples} results, "
            f"got {len(all_results)}"
        )

    print(f"Total results collected: {len(all_results)}")

    # 6. Metric Calculation
    print("Calculating metrics...")
    if args.task == "element":
        result_report = evaluate_element(all_results)
    elif args.task == "layout":
        result_report = evaluate_layout(all_results)
    else:
        raise ValueError("Unknown task")

    result_report["input_files"] = args.uivision_test_file
    result_report["task_type"] = args.task

    # 7. Save results
    output_report = {
        "model_type": args.model_type,
        "task_type": args.task,
        "results": result_report
    }

    save_dir = os.path.dirname(args.log_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(args.log_path, 'w') as f:
        json.dump(output_report, f, indent=4)

    logging.info(f"Evaluation of UIVision {args.task} completed using {num_gpus} GPUs.")

if __name__ == "__main__":
    main(parse_args())
