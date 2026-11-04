#!/bin/bash
# Qwen3-VL-4B bidirectional RL. Run from the package root.
export PYTHONPATH="./Bi_RL/verl-agent-master:$PYTHONPATH"
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export GLOO_SOCKET_IFNAME=eth0
export RAY_DISABLE_LOG_COMPRESSION=1
export PYTHONUNBUFFERED=1

MODEL_PATH="/path/to/Qwen3-VL-4B-Bi-SFT-checkpoint"
BIGROUNDER_TASKS_JSON="/path/to/rl_data.json"
BERT_MODEL_PATH="/path/to/roberta-large"
DATA_DIR="/path/to/data"
SAVE_DIR="/path/to/output/qwen3_4B_Bi_RL"
export TENSORBOARD_DIR="$SAVE_DIR/tensorboard_log"

# G: instructions per target; K: grounding rollouts per instruction.
GROUP_SIZE=4
FORWARD_ROLLOUTS=4
INSTRUCTION_SOURCE=online       # online / original / offline
UPDATE_REVERSE=true
CYCLE_VERIFICATION=true
REVERSE_REWARD_MODE=semantic   # semantic / binary
MAX_STEPS=5                   # K+1 for online; K for original or offline

# 1x16: G=1, K=16, INSTRUCTION_SOURCE=original, UPDATE_REVERSE=false, MAX_STEPS=16.
# 1x20: G=1, K=20, INSTRUCTION_SOURCE=original, UPDATE_REVERSE=false, MAX_STEPS=20.
# Online generation, forward updates only: UPDATE_REVERSE=false.
# Frozen generator: INSTRUCTION_SOURCE=offline, UPDATE_REVERSE=false, MAX_STEPS=4.
# Gate-only: REVERSE_REWARD_MODE=binary. No cycle gate: CYCLE_VERIFICATION=false.

# Prepare framework parquet files.
python3 -m agent_system.data_preprocess.prepare \
    --mode visual \
    --local_dir "$DATA_DIR" \
    --train_data_size 63000 \
    --val_data_size 512

# GRPO training
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$DATA_DIR/visual/train.parquet" \
    data.val_files="$DATA_DIR/visual/test.parquet" \
    data.train_batch_size=64 \
    data.val_batch_size=64 \
    data.max_prompt_length=20000 \
    data.max_response_length=256 \
    data.filter_overlong_prompts=False \
    data.truncation='error' \
    data.image_key=images \
    data.return_raw_chat=True \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.use_invalid_action_penalty=False \
    \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.max_num_batched_tokens=40960 \
    actor_rollout_ref.rollout.max_model_len=40960 \
    +actor_rollout_ref.rollout.block_size=16 \
    \
    actor_rollout_ref.rollout.temperature=0.9 \
    actor_rollout_ref.rollout.top_p=0.9 \
    actor_rollout_ref.rollout.top_k=40 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.9 \
    \
    env.env_name=bigrounder \
    env.seed=42 \
    env.max_steps=$MAX_STEPS \
    env.rollout.n=$GROUP_SIZE \
    env.resources_per_worker.num_cpus=0.9 \
    +env.bigrounder.tasks_json="$BIGROUNDER_TASKS_JSON" \
    +env.bigrounder.forward_rollouts=$FORWARD_ROLLOUTS \
    +env.bigrounder.instruction_source=$INSTRUCTION_SOURCE \
    +env.bigrounder.update_reverse=$UPDATE_REVERSE \
    +env.bigrounder.cycle_verification=$CYCLE_VERIFICATION \
    +env.bigrounder.reverse_reward_mode=$REVERSE_REWARD_MODE \
    +env.bigrounder.similarity_threshold=0.85 \
    +env.bigrounder.bert_model_path="$BERT_MODEL_PATH" \
    +env.bigrounder.coord_tolerance=6.0 \
    +env.bigrounder.format_reward_weight=0.10 \
    +env.bigrounder.action_match_reward_weight=0.90 \
    +env.bigrounder.format_penalty=-0.2 \
    +env.bigrounder.bbox_alpha=0.8 \
    +env.bigrounder.bbox_gamma=0.5 \
    +env.bigrounder.bert_num_layers=17 \
    +env.bigrounder.device=cpu \
    +env.bigrounder.verbose=True \
    trainer.critic_warmup=1 \
    trainer.logger="['console','tensorboard']" \
    trainer.project_name='bigrounder_rl' \
    trainer.experiment_name='qwen3_4B_bigrounder' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=4 \
    trainer.save_freq=250 \
    trainer.test_freq=250 \
    trainer.default_local_dir="$SAVE_DIR" \
    trainer.total_epochs=1 \
    trainer.val_before_train=True \
    +trainer.gradient_clip=1.0 \
    +trainer.log_freq=1 \
    +trainer.verbose=2
