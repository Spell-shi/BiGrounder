#!/bin/bash
# Run from the package root. Edit paths and select a benchmark command.

# ScreenSpot-Pro
python3 Eval/eval_screenspot_pro.py \
    --model_type BiGrounder \
    --model_name_or_path /path/to/BiGrounder-checkpoint \
    --screenspot_imgs /path/to/screenspot-pro/images \
    --screenspot_test /path/to/screenspot-pro/annotations \
    --task all \
    --inst_style instruction \
    --language en \
    --gt_type positive \
    --num_gpu 8 \
    --log_path /path/to/results/screenspot_pro.json

# ScreenSpot-v2
python3 Eval/eval_ssv2.py \
    --model-path /path/to/BiGrounder-checkpoint \
    --simplified-json /path/to/screenspot-v2/simplified.json \
    --min-free 40 \
    --sample-every 1 \
    --out-dir /path/to/results/ssv2

# MMBench-GUI
python3 Eval/eval_mmbench_gui.py \
    --model_type BiGrounder \
    --model_name_or_path /path/to/BiGrounder-checkpoint \
    --images_dir /path/to/mmbench_gui/images \
    --annotations_file /path/to/mmbench_gui/annotations.json \
    --gt_type positive \
    --num_gpu 8 \
    --log_path /path/to/results/mmbench_gui.json

# UI-Vision
python3 Eval/eval_uivision.py \
    --model_type BiGrounder \
    --model_name_or_path /path/to/BiGrounder-checkpoint \
    --uivision_imgs /path/to/uivision/images \
    --uivision_test_file /path/to/uivision/test.json \
    --task element \
    --num_gpus 8 \
    --log_path /path/to/results/uivision.json

# OSWorld-G / OSWorld-G-R-Variants
python3 Eval/eval_osworld_g.py \
    --model_path /path/to/BiGrounder-checkpoint \
    --original /path/to/osworld-g/OSWorld-G_refined.json \
    --variants OSWorld-G-R-Variants/OSWorld-G-R-Variants.json \
    --image_dir /path/to/osworld-g/images \
    --classification_path /path/to/osworld-g/classification_result.json \
    --port 8908 \
    --output /path/to/results/osworld_g.json
