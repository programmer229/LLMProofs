export VLLM_ATTENTION_BACKEND=XFORMERS

DATA_DIR=/home/ubuntu/o1-replication/CustomTinyZero/data/TextbookQuestions
#BASE_MODEL=/home/ubuntu/o1-replication/o_series/model_saves/qwen2.5_7B_1.0SFT # 7B model
#BASE_MODEL=/home/ubuntu/o1-replication/CustomTinyZero/checkpoints/verl_grpo_numina/qwen2.5_7b_numina_rl6/actor/global_step_1200 # 7B model from rl6 (which is from 1.0SFT model)
BASE_MODEL=Qwen/Qwen2.5-7B-Instruct # 7B model from combined_math, which is from rl6, which which is from 1.0SFT model)
PROJECT_NAME=verl_grpo_textbooks
EXPERIMENT_NAME=qwen2.5_7b_2_MATH_20k
OVERRIDE_CONFIG='{tokenizer_config:{eos_token:"<｜endoftext｜>",additional_special_tokens:["<｜end▁of▁sentence｜>"]},generation_config:{eos_token_id:151643}}'

#####################################################

if [ -d "/home/ubuntu/o1-replication/CustomTinyZero/checkpoints/$PROJECT_NAME/$EXPERIMENT_NAME" ]; then
    echo "Directory already exists. You might overwrite existing saved models and logs!!!"
    echo "It is recommended to use a different experiment name, unless you are sure this experiment can be overwritten."
    echo "Are you sure you want to run with the current experiment name? (Y/n)"
    read answer
    if [ "$answer" != "Y" ]; then
        echo "Exiting..."
        
    fi
fi

mkdir -p /home/ubuntu/o1-replication/CustomTinyZero/checkpoints/$PROJECT_NAME/$EXPERIMENT_NAME
LOG_FILE=/home/ubuntu/o1-replication/CustomTinyZero/checkpoints/$PROJECT_NAME/$EXPERIMENT_NAME/logfile.txt

# Save a copy of this script to the experiment directory
cp "$0" "/home/ubuntu/o1-replication/CustomTinyZero/checkpoints/$PROJECT_NAME/$EXPERIMENT_NAME/$(basename $0)"

set -x

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_batch_size=128 \
    data.val_batch_size=128 \
    data.max_prompt_length=1024 \
    data.max_response_length=2048 \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size=16 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24000 \
    critic.ppo_max_token_len_per_gpu=16192 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=2 \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=2 \
    critic.ulysses_sequence_parallel_size=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.002 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.grad_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.003 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.default_hdfs_dir="/home/ubuntu/o1-replication/CustomTinyZero/checkpoints/$PROJECT_NAME/$EXPERIMENT_NAME" \
    trainer.default_local_dir="/home/ubuntu/o1-replication/CustomTinyZero/checkpoints/$PROJECT_NAME/$EXPERIMENT_NAME" \
    trainer.total_epochs=15 $@ 2>&1 | tee -a $LOG_FILE
    