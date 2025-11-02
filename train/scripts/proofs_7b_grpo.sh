export VLLM_ATTENTION_BACKEND=XFORMERS

DATA_DIR=/home/ubuntu/CustomTinyZero/data/proofs
BASE_MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
#BASE_MODEL=/home/ubuntu/o1-replication/CustomTinyZero/checkpoints/verl_intergration/llama3.2_3b_integration/actor/global_step_80
EXPERIMENT_NAME=Qwen7b-disc-rank
PROJECT_NAME=llmjudge_proofs
# Reward conversion options:
#   group_points (default) - average group point schedule
#   harmonic_rank          - apply 1, 1/2, 1/3, ... by global rank per problem
#   squared                - square the averaged group points
TRAIN_REWARD_CONVERSION_MODE=${TRAIN_REWARD_CONVERSION_MODE:-squared}

#####################################################

if [ -d "/home/ubuntu/test/CustomTinyZero/checkpoints/$PROJECT_NAME/$EXPERIMENT_NAME" ]; then
    echo "Directory already exists. You might overwrite existing saved models and logs!!!"
    echo "It is recommended to use a different experiment name, unless you are sure this experiment can be overwritten."
    echo "Are you sure you want to run with the current experiment name? (Y/n)"
    read answer
    # if [ "$answer" != "Y" ]; then
    #     echo "Exiting..."
    #     exit 1
    # fi
fi

mkdir -p /home/ubuntu/CustomTinyZero/checkpoints/$PROJECT_NAME/$EXPERIMENT_NAME
LOG_FILE=/home/ubuntu/test/CustomTinyZero/checkpoints/$PROJECT_NAME/$EXPERIMENT_NAME/logfile.txt

# Save a copy of this script to the experiment directory
cp "$0" "/home/ubuntu/CustomTinyZero/checkpoints/$PROJECT_NAME/$EXPERIMENT_NAME/$(basename $0)"

set -x

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    +judge.model=$BASE_MODEL \
    +judge.location=local \
    +judge.gpus=4 \
    data.train_batch_size=64 \
    data.val_batch_size=64 \
    data.max_prompt_length=2048 \
    data.max_response_length=10368 \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.actor.optim.lr=5e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24000 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.grad_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.default_hdfs_dir="/home/ubuntu/test/CustomTinyZero/checkpoints/$PROJECT_NAME/$EXPERIMENT_NAME" \
    trainer.default_local_dir="/home/ubuntu/test/CustomTinyZero/checkpoints/$PROJECT_NAME/$EXPERIMENT_NAME" \
    trainer.total_epochs=200 \
    train_reward_conversion_mode=$TRAIN_REWARD_CONVERSION_MODE \
    $@ 2>&1 | tee -a $LOG_FILE
