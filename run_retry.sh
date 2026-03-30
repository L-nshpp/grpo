#!/bin/bash
set -x

export PROJ_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

export DATA_ROOT="/data/shipei/experiments_data_0117/sql_optimize"
mkdir -p "$DATA_ROOT"

# 1. W&B 日志路径 (大文件)
export WANDB_DIR="$DATA_ROOT/wandb_logs"
# 2. 模型 Checkpoint 输出路径 (巨大文件)
export OUTPUT_DIR="$DATA_ROOT/checkpoints"
# 3. 文本日志路径
export LOG_DIR="$DATA_ROOT/logs"

mkdir -p "$WANDB_DIR"
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export CUDA_VISIBLE_DEVICES=3,2,1,0
nproc_per_node=4
export RAY_TMPDIR="/data/shipei/ray_tmp" 
mkdir -p "$RAY_TMPDIR"

# 模型配置
MODEL_PATH="/home/shipei/eff_train/models/Qwen2.5-7B-Instruct-Merged-0112"
MODEL_NAME=$(basename "$MODEL_PATH")
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# W&B 离线配置
export WANDB_MODE=offline
export WANDB_PROJECT="sql_patch_optimize"
export WANDB_NAME="${MODEL_NAME}_${TIMESTAMP}"
export WANDB_SAVE_CODE=true
export WANDB_CACHE_DIR="$DATA_ROOT/wandb_cache"
export WANDB_CONFIG_DIR="$DATA_ROOT/wandb_config"


TRAIN_DATA="$PROJ_DIR/data/rl_train_filter_chat_format.parquet"
TEST_DATA="$PROJ_DIR/data/rl_dev_slowest_30.parquet"
REWARD_FILE="$PROJ_DIR/sql_reward_record.py" 
# 具体的输出子目录
RUN_OUTPUT_DIR="${OUTPUT_DIR}/${MODEL_NAME}_grpo_${TIMESTAMP}"
RUN_LOG_FILE="${LOG_DIR}/training_${TIMESTAMP}.log"

echo "=== 训练启动 ==="
echo "代码目录: $PROJ_DIR"
echo "数据存储: $DATA_ROOT"
echo "日志文件: $RUN_LOG_FILE"

{
  echo "W&B Run Name: $WANDB_NAME"
  
  pkill -f ray
  rm -rf "$RAY_TMPDIR/*" || true
  python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    +critic.enable=False \
    data.train_files="$TRAIN_DATA" \
    data.val_files="$TEST_DATA" \
    data.train_batch_size=32 \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    +model.lora_rank=64 \
    +model.lora_alpha=128 \
    +model.target_modules=all-linear \
    data.max_prompt_length=8192 \
    data.max_response_length=1024 \
    ++actor_rollout_ref.rollout.max_num_batched_tokens=10240 \
    ++actor_rollout_ref.model.model_kwargs.max_model_len=10240 \
    trainer.n_gpus_per_node=$nproc_per_node \
    ++model.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.actor.optim.lr=5e-6 \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    +reward_model.overlong_buffer.enable=True \
    +reward_model.overlong_buffer.penalty_factor=1.0 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    +model.enable_gradient_checkpointing=true \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    trainer.test_freq=25 \
    trainer.save_freq=20 \
    trainer.resume_mode=resume_from_path \
    trainer.resume_from_path=/data/shipei/experiments_data_0117/sql_optimize/checkpoints/global_step_150 \
    trainer.total_training_steps=300 \
    trainer.project_name='sql_patch_optimize' \
    trainer.experiment_name='v1_grpo_db_opt_300steps_part3' \
    trainer.default_local_dir="$OUTPUT_DIR" \
    custom_reward_function.path="$REWARD_FILE" \
    custom_reward_function.name='sql_optimize' \
    "$@"

} 2>&1 | tee -a "$RUN_LOG_FILE"


LATEST_RUN=$(find "$WANDB_DIR" -maxdepth 2 -type d -name "offline-run*${TIMESTAMP}*" | head -n 1)

if [ -z "$LATEST_RUN" ]; then
    LATEST_RUN=$(ls -td "$WANDB_DIR"/offline-run* | head -1)
fi

echo ""
echo "========================================================================"
echo "✅ 训练结束！数据已保存在 /data/shipei/..."
echo "📊 要查看图表，请复制下方命令在【本地笔记本】运行："
echo "------------------------------------------------------------------------"
echo "scp -r shipei@aiseon:$LATEST_RUN ./ && wandb sync $(basename "$LATEST_RUN")"
echo "------------------------------------------------------------------------"
echo ""
