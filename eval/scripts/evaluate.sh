# export HF_HOME="/mnt/sdb/pretrained_models/LLaDA"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export CUDA_VISIBLE_DEVICES=0,1,2,3
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-4}
EVAL_LIMIT=${EVAL_LIMIT:-}  # empty = full dataset; e.g. EVAL_LIMIT=20 for smoke tests
GEN_LENGTH=${GEN_LENGTH:-160}  # max generation tokens; 120 truncates many reports
TEMPERATURE=${TEMPERATURE:-0}
CFG_SCALE=${CFG_SCALE:-0}
REMASKING=${REMASKING:-low_confidence}
BLOCK_LENGTH=${BLOCK_LENGTH:-$GEN_LENGTH}
STEPS=${STEPS:-60}
## Ensure local lmms-eval package is on PYTHONPATH so `-m lmms_eval` works
export PYTHONPATH=$(pwd)/lmms-eval:$(pwd):$PYTHONPATH
mkdir -p "${HF_HOME}"
# Define multiple model paths
MODEL_PATHS=(
    "/mnt/bn/tns-algo-search-my/yingshu.li/code/DRRG-V/train/exp/llada_v_core/checkpoint-12693"
)

# Set output path — include key hyperparams for sweep disambiguation
OUTPUT_PATH=exp/llava_v_core_eval_bs${EVAL_BATCH_SIZE}_gl${GEN_LENGTH}_bl${BLOCK_LENGTH}_s${STEPS}_t${TEMPERATURE}_cfg${CFG_SCALE}
# Set task names
TASK_NAMES="mimic_cxr"

TOTAL_GPUS=4

# Split task names into array
IFS=',' read -ra TASKS <<< "$TASK_NAMES"

# Create task queue
declare -a TASK_QUEUE
for model_path in "${MODEL_PATHS[@]}"; do
    MODEL=llava_oneversion_qwen
    MODEL_NAME=llava_qwen
    CONV_TEMPLATE=qwen

    for task in "${TASKS[@]}"; do
        case $task in
            mimic_cxr)
                # Medical report generation: longer output, use semi-autoregressive
                GEN_KWARGS="{\"temperature\":${TEMPERATURE},\"cfg_scale\":${CFG_SCALE},\"remasking\":\"${REMASKING}\",\"gen_length\":${GEN_LENGTH},\"block_length\":${BLOCK_LENGTH},\"steps\":${STEPS}}"
                ;;
            *)
                GEN_KWARGS='{"temperature":0,"cfg_scale":0,"remasking":"low_confidence","gen_length":2,"block_length":1,"steps":2}'
                ;;
        esac

        TASK_QUEUE+=("$model_path $task $MODEL $MODEL_NAME $CONV_TEMPLATE $GEN_KWARGS")
    done
done

TOTAL_TASKS=${#TASK_QUEUE[@]}
echo "Total $TOTAL_TASKS evaluation tasks to execute"

COUNT=0
for CURRENT_TASK_STRING in "${TASK_QUEUE[@]}"; do
    COUNT=$((COUNT + 1))

    read -r MODEL_PATH TASK_NAME CURRENT_MODEL CURRENT_MODEL_NAME CURRENT_CONV_TEMPLATE CURRENT_GEN_KWARGS <<< "$CURRENT_TASK_STRING"

    MODEL_PATH_LAST=$(basename "$MODEL_PATH")
    CURRENT_OUTPUT_PATH="$OUTPUT_PATH/$MODEL_PATH_LAST"
    LOG_FILE_NAME="${TASK_NAME}_${CURRENT_GEN_KWARGS//,/_}.log"
    mkdir -p "$CURRENT_OUTPUT_PATH"
    export LMMS_EVAL_OUTPUT_PATH="$CURRENT_OUTPUT_PATH"

    echo "Task: $TASK_NAME" > "$CURRENT_OUTPUT_PATH/$LOG_FILE_NAME"
    echo "Starting task ($COUNT / $TOTAL_TASKS): Evaluating $MODEL_PATH_LAST on $TASK_NAME using ALL $TOTAL_GPUS GPUs"

    CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONUNBUFFERED=1 accelerate launch --num_processes=$TOTAL_GPUS --main_process_port 2277 -m lmms_eval \
        --model "$CURRENT_MODEL" \
        ${CURRENT_GEN_KWARGS:+--gen_kwargs="$CURRENT_GEN_KWARGS"} \
        --model_args "pretrained=$MODEL_PATH,conv_template=$CURRENT_CONV_TEMPLATE,model_name=$CURRENT_MODEL_NAME" \
        --tasks "$TASK_NAME" \
        --batch_size $EVAL_BATCH_SIZE \
        ${EVAL_LIMIT:+--limit $EVAL_LIMIT} \
        --log_samples \
        --log_samples_suffix "$TASK_NAME" \
        --output_path "$CURRENT_OUTPUT_PATH" 2>&1 | tee -a "$CURRENT_OUTPUT_PATH/$LOG_FILE_NAME"

    echo "Task $TASK_NAME completed."
    echo "----------------------------------------"
done

echo "All evaluation tasks completed!"
