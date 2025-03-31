export VLLM_ATTENTION_BACKEND=XFORMERS

DATA_DIR=/home/ubuntu/o1-replication-central/CustomTinyZero/data/svg_variants
BASE_MODEL=Qwen/Qwen2.5-7B-Instruct
EXPERIMENT_NAME=qwen2.5_7b_svg_gpt4o_mini2
PROJECT_NAME=svg_judge_experiments

#####################################################

if [ -d "/home/ubuntu/o1-replication-central/CustomTinyZero/checkpoints/$PROJECT_NAME/$EXPERIMENT_NAME" ]; then
    echo "Directory already exists. You might overwrite existing saved models and logs!!!"
    echo "It is recommended to use a different experiment name, unless you are sure this experiment can be overwritten."
    echo "Are you sure you want to run with the current experiment name? (Y/n)"
    read answer
    # if [ "$answer" != "Y" ]; then
    #     echo "Exiting..."
    #     exit 1
    # fi
fi

mkdir -p /home/ubuntu/o1-replication-central/CustomTinyZero/checkpoints/$PROJECT_NAME/$EXPERIMENT_NAME
LOG_FILE=/home/ubuntu/o1-replication-central/CustomTinyZero/checkpoints/$PROJECT_NAME/$EXPERIMENT_NAME/logfile.txt

# Save a copy of this script to the experiment directory
cp "$0" "/home/ubuntu/o1-replication-central/CustomTinyZero/checkpoints/$PROJECT_NAME/$EXPERIMENT_NAME/$(basename $0)"

set -x

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    +judge.model=$BASE_MODEL \
    +judge.location=local \
    +judge.gpus=4 \
    data.train_batch_size=32 \
    data.val_batch_size=32 \
    data.max_prompt_length=512 \
    data.max_response_length=2048 \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24000 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.grad_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.default_hdfs_dir="/home/ubuntu/o1-replication-central/CustomTinyZero/checkpoints/$PROJECT_NAME/$EXPERIMENT_NAME" \
    trainer.default_local_dir="/home/ubuntu/o1-replication-central/CustomTinyZero/checkpoints/$PROJECT_NAME/$EXPERIMENT_NAME" \
    trainer.total_epochs=200 $@ 2>&1 | tee -a $LOG_FILE