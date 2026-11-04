#!/bin/bash
# Run from the package root.
export PYTHONPATH="./Bi_SFT/ms-swift-4.0.0:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL_PATH="/path/to/Qwen3.5-35B-A3B"
DATASET_PATH="/path/to/data/bi_sft_100k.json"
OUTPUT_DIR="/path/to/output/Qwen3_5_35B_A3B_dual_SFT"

# Use the prepared 100k bidirectional SFT subset.
torchrun \
  --nproc_per_node=8 \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --master_port=29500 \
  -m swift.cli.sft \
  --model "$MODEL_PATH" \
  --model_type qwen3_5_moe \
  --tuner_type full \
  --dataset "$DATASET_PATH" \
  --torch_dtype bfloat16 \
  --experts_impl grouped_mm \
  --add_non_thinking_prefix true \
  --num_train_epochs 2 \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 1 \
  --learning_rate 1e-5 \
  --gradient_checkpointing true \
  --gradient_accumulation_steps 2 \
  --eval_strategy epoch \
  --save_strategy epoch \
  --logging_steps 8 \
  --max_length 20000 \
  --max_new_tokens 2048 \
  --warmup_ratio 0.1 \
  --dataloader_num_workers 16 \
  --freeze_vit true \
  --freeze_llm false \
  --freeze_aligner false \
  --dataset_num_proc 32 \
  --split_dataset_ratio 0.0001 \
  --save_total_limit 2 \
  --save_only_model false \
  --output_dir "$OUTPUT_DIR" \
  --deepspeed zero3 \
  --use_liger_kernel false \
  --attn_impl flash_attn \
  --ddp_find_unused_parameters false \
  --ddp_broadcast_buffers false
