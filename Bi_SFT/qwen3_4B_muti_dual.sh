export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Create cache directories
mkdir -p $MODELSCOPE_CACHE
mkdir -p $HF_HOME

NAME="Qwen3_vl_4B_dual_SFT"

# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

torchrun \
  --nproc_per_node=$GPU_NUM_PER_NODE \
  --nnodes=$WORLD_SIZE \
  --node_rank=$RANK \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  -m swift.cli.sft \
  --model "/path/to/Qwen3-VL-4B-Instruct" \
  --train_type full \
  --dataset "/path/to/data/task_type_1_action_grounding/merged_data_action_grounding_norm.json"\
          "/path/to/data/task_type_1_action_grounding_gemini/grounding_only_bbox_ref_0_100000_verified_norm.json"\
          "/path/to/data/task_type_1_action_grounding_gemini/grounding_only_bbox_ref_100000_200000_verified_norm.json"\
          "/path/to/data/task_type_1_action_grounding_gemini/grounding_only_bbox_ref_200000_675145_verified_norm.json"\
          "/path/to/data/task_type_2_bbox/merged_data_grounding_only_bbox_norm.json"\
          "/path/to/data/task_type_3_point/merged_data_grounding_only_point.json"\
          "/path/to/data/task_type_1_action_grounding_reverse/merged_data_action_grounding_reverse_norm.json"\
          "/path/to/data/task_type_1_action_grounding_gemini_reverse/grounding_only_bbox_0_100000_verified_reverse_norm.json"\
          "/path/to/data/task_type_1_action_grounding_gemini_reverse/grounding_only_bbox_100000_200000_verified_reverse_norm.json"\
          "/path/to/data/task_type_1_action_grounding_gemini_reverse/grounding_only_bbox_200000_675145_verified_reverse_norm.json"\
          "/path/to/data/task_type_2_bbox/merged_data_grounding_only_bbox_reverse_norm.json"\
          "/path/to/data/task_type_3_point_reverse/merged_data_grounding_only_point_reverse.json"\
  --torch_dtype bfloat16 \
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
  --output_dir "/path/to/output/${NAME}" \
  --deepspeed zero3 \
  --use_liger_kernel true \
  --attn_impl flash_attn \
  --ddp_find_unused_parameters false \
  --ddp_broadcast_buffers false