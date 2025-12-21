export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=enp1s0f1np1
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=ALL
export HF_HOME="/mnt/sdb/pretrained_models/Qwen3-0.6B-diffusion"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export PYTHONPATH=$(pwd)/$PYTHONPATH
mkdir -p "${HF_HOME}"
num_node=$1
gpu_num=$2

if [[ -z "$num_node" || -z "$gpu_num" ]]; then
  echo "Usage: bash $0 <num_node> <gpu_num>"
  exit 1
fi
# need to change num_node and gpu_num! 
# Configuration note: This script is typically run with 4 nodes and 8 GPUs per node.
# The gradient_accumulation_steps should be adjusted based on your GPU count to maintain effective batch size.
# For example, with 8 GPUs, set gradient_accumulation_steps=8.

MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-"29199"}
RANK=${RANK:-"0"}

echo "master_addr ${MASTER_ADDR}"
echo "master_port ${MASTER_PORT}"
echo "node_rank ${RANK}"
echo "gpu_num ${gpu_num}"
echo "num_node ${num_node}"

LLM_VERSION="dllm-collection/Qwen3-0.6B-diffusion-mdlm-v0.1"
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION=/mnt/sdb/pretrained_models/siglip2-so400m-patch14-384
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"

############### Finetune ################

PROMPT_VERSION="qwen"

BASE_RUN_NAME="llada_v_finetune5"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"

ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node=${gpu_num} --nnodes=${num_node} --master_addr=${MASTER_ADDR} --master_port ${MASTER_PORT} --node_rank=${RANK} \
    llava/train/train_mem.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path "/mnt/sdb/pretrained_models/Qwen3-0.6B-diffusion" \
    --version ${PROMPT_VERSION} \
    --data_path "/mnt/sda/shaoyang/model/LLaDA/LLaDA-V/data/train_llava_llada.json" \
    --image_folder "/mnt/sdb/datasets/mimic_original/2.0.0/files" \
    --video_folder "" \
    --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_language_model" \
    --mm_vision_tower_lr=2e-6 \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --pretrain_mm_mlp_adapter "/mnt/sda/shaoyang/model/LLaDA/LLaDA-V/train/llada_v_prepare/mm_projector.bin" \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio pad \
    --image_grid_pinpoints "(1x1),...,(6x6)" \
    --mm_patch_merge_type flat \
    --bf16 True \
    --run_name $BASE_RUN_NAME \
    --output_dir "exp/$BASE_RUN_NAME" \
    --num_train_epochs 2 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 5000 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 1024 \
    --gradient_checkpointing True \
    --dataloader_num_workers 0 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --dataloader_drop_last True \
    --attn_implementation sdpa \
    --use_conversation_mask False \
    # --lora_enable True \
    # --double_quant True \
    # --quant_type nf4 \
    # --lora_r 4 \
    # --lora_alpha 8 \
    # --lora_dropout 0.1 \
    # --lora_bias "none"
    # --deepspeed scripts/zero3.json \
    # --torch_compile False \
    # --torch_compile_backend "inductor" \
    # --mm_resampler_type perceiver \