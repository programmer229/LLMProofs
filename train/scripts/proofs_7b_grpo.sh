export VLLM_ATTENTION_BACKEND=XFORMERS

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_DATA_ROOT="/home/ubuntu/CustomTinyZero/data"
DEFAULT_CHECKPOINT_ROOT="/home/ubuntu/test/CustomTinyZero/checkpoints"

BASE_MODEL=${BASE_MODEL:-deepseek-ai/DeepSeek-R1-Distill-Qwen-7B}
#BASE_MODEL=/home/ubuntu/o1-replication/CustomTinyZero/checkpoints/verl_intergration/llama3.2_3b_integration/actor/global_step_80
DATASET=${DATASET:-creative_writing}

# Reward conversion options:
#   group_points (default) - average group point schedule
#   harmonic_rank          - apply 1, 1/2, 1/3, ... by global rank per problem
#   squared                - square the averaged group points
#   judge_score            - per-sample judge scoring (normalized from <JUDGE_SCORE> outputs)
TRAIN_REWARD_CONVERSION_MODE=${TRAIN_REWARD_CONVERSION_MODE:-harmonic_rank}

case "$DATASET" in
    creative_writing)
        DATA_DIR=${DATA_DIR:-${DEFAULT_DATA_ROOT}/creative_writing}
        PROJECT_NAME=${PROJECT_NAME:-llmjudge_creative}
        EXPERIMENT_NAME=${EXPERIMENT_NAME:-CreativeWriting7b-${TRAIN_REWARD_CONVERSION_MODE}}
        TRAIN_SPLIT_FILE=${TRAIN_SPLIT_FILE:-train.parquet}
        VAL_SPLIT_FILE=${VAL_SPLIT_FILE:-val.parquet}
        ;;
    proofs)
        DATA_DIR=${DATA_DIR:-${DEFAULT_DATA_ROOT}/proofs}
        PROJECT_NAME=${PROJECT_NAME:-llmjudge_proofs}
        EXPERIMENT_NAME=${EXPERIMENT_NAME:-Proofs7b-${TRAIN_REWARD_CONVERSION_MODE}}
        TRAIN_SPLIT_FILE=${TRAIN_SPLIT_FILE:-train.parquet}
        VAL_SPLIT_FILE=${VAL_SPLIT_FILE:-test.parquet}
        ;;
    *)
        echo "Unsupported DATASET '${DATASET}'. Supported values: creative_writing, proofs."
        exit 1
        ;;
esac

# Always (re)generate the creative writing dataset so schema updates propagate
if [ "$DATASET" = "creative_writing" ]; then
    TRAIN_PATH="$DATA_DIR/$TRAIN_SPLIT_FILE"
    VAL_PATH="$DATA_DIR/$VAL_SPLIT_FILE"

    STORIES_CSV_CANDIDATE="${TRAIN_DIR}/../stories.csv"
    if [ ! -f "$STORIES_CSV_CANDIDATE" ]; then
        echo "stories.csv not found at ${STORIES_CSV_CANDIDATE}. Cannot generate creative writing data."
        exit 1
    fi

    echo "Regenerating creative-writing dataset at ${DATA_DIR} from ${STORIES_CSV_CANDIDATE}."
    mkdir -p "$DATA_DIR"
    python3 "${TRAIN_DIR}/data/create_creative_writing.py" \
        --input "$STORIES_CSV_CANDIDATE" \
        --output "$DATA_DIR"

    if [ ! -f "$TRAIN_PATH" ] || [ ! -f "$VAL_PATH" ]; then
        echo "Expected parquet files not found after generation attempt:"
        echo "  $TRAIN_PATH"
        echo "  $VAL_PATH"
        echo "Set DATA_DIR to an existing dataset or ensure stories.csv is available."
        exit 1
    fi
fi

# Announce configuration
echo "Running dataset: ${DATASET}"
echo "Using data directory: ${DATA_DIR}"

# Reward conversion options:

#####################################################

CHECKPOINT_DIR="${DEFAULT_CHECKPOINT_ROOT}/${PROJECT_NAME}/${EXPERIMENT_NAME}"

if [ -d "$CHECKPOINT_DIR" ]; then
    echo "Directory already exists. You might overwrite existing saved models and logs!!!"
    echo "It is recommended to use a different experiment name, unless you are sure this experiment can be overwritten."
    echo "Are you sure you want to run with the current experiment name? (Y/n)"
    read answer
    # if [ "$answer" != "Y" ]; then
    #     echo "Exiting..."
    #     exit 1
    # fi
fi

mkdir -p "$CHECKPOINT_DIR"
LOG_FILE="${CHECKPOINT_DIR}/logfile.txt"

# Save a copy of this script to the experiment directory
cp "$0" "${CHECKPOINT_DIR}/$(basename "$0")"

set -x

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATA_DIR/$TRAIN_SPLIT_FILE \
    data.val_files=$DATA_DIR/$VAL_SPLIT_FILE \
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
    trainer.default_hdfs_dir="$CHECKPOINT_DIR" \
    trainer.default_local_dir="$CHECKPOINT_DIR" \
    trainer.total_epochs=200 \
    train_reward_conversion_mode=$TRAIN_REWARD_CONVERSION_MODE \
    "$@" 2>&1 | tee -a $LOG_FILE
