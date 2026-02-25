# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LLaDA-V-Qwen: A diffusion-based language model for radiology report generation. Combines a SigLIP vision encoder with a Qwen3 0.6B diffusion language model, using complementary masking for semi-autoregressive text generation. Built on LLaVA-NeXT architecture, adapted from autoregressive to diffusion-based generation.

## Common Commands

### Environment Setup
```bash
# Install dLLM framework
git clone https://github.com/ZHZisZZ/dllm.git && cd dllm && pip install -e . && cd ..

# Install training environment
cd train && bash init_env.sh

# Install evaluation framework
cd eval/lmms-eval && pip install -e . && pip install -e ".[metrics]"
```

### Training
```bash
# Stage 1: Mapper pretraining (vision encoder + LM frozen, only mapper trains)
cd train && bash scripts/llada_v_pretrain.sh <num_nodes> <gpu_num>

# Stage 2: End-to-end finetuning (all components trainable)
cd train && bash scripts/llada_v_finetune.sh <num_nodes> <gpu_num>
# Note: Edit script to set data_path, image_folder, video_folder paths
```

Training uses DeepSpeed ZeRO-3 (`scripts/zero3.json`) with `torchrun` for distributed training. Entry point is `llava/train/train_mem.py`.

### Evaluation
```bash
cd eval && bash scripts/evaluate.sh
# Edit MODEL_PATHS array and CUDA_VISIBLE_DEVICES in evaluate.sh first
# Uses accelerate launch with lmms_eval framework
```

### Inference Demo
```bash
cd train && python generate_demo.py
```

### Data Preparation
```bash
python convert_data.py  # Converts MIMIC-CXR annotations to LLaVA format
# Outputs: train_llava_llada.json, test_llava_llada.json
```

## Architecture

### Pipeline
```
Input Image → SigLIP Vision Encoder → Vision Resampler (QFormer) → MLP Projector → Qwen3 Diffusion LM → Report
```

### Key Source Locations

- **Model architecture entry**: `train/llava/model/llava_arch.py` — `LlavaMetaModel` and `LlavaMetaForCausalLM` base classes
- **Qwen3 diffusion LM**: `train/llava/model/language_model/modeling_Qwem.py` — core diffusion logic, complementary masking, generation
- **Qwen3 LLaVA wrapper**: `train/llava/model/language_model/llava_qwen.py` — combines vision + language into `LlavaQwen3ModelLM`
- **Vision encoder**: `train/llava/model/multimodal_encoder/siglip_encoder.py` — SigLIP (`google/siglip2-so400m-patch14-384`)
- **Vision resampler**: `train/llava/model/multimodal_resampler/` — QFormer, Perceiver, spatial pooling options
- **MM projector**: `train/llava/model/multimodal_projector/builder.py` — MLP2x_GELU projection
- **Training loop**: `train/llava/train/train.py` — dataset loading, preprocessing, data collation
- **Custom trainer**: `train/llava/train/llava_trainer.py`
- **Conversation templates**: `train/llava/conversation.py` — uses `qwen` template with radiology system prompt
- **Constants**: `train/llava/constants.py` — `IGNORE_INDEX=-100`, `IMAGE_TOKEN_INDEX=-200`, special tokens

### Complementary Masking (Core Innovation)

In `modeling_Qwem.py`, the forward pass creates two views of input embeddings:
1. **Noisy view**: t% of tokens masked (standard diffusion)
2. **Complementary view**: (1-t)% of tokens masked (the complement)

Both views are concatenated, run through the model, and loss is computed across both, normalized by their respective mask ratios. This provides bidirectional context during training. The prompt tokens (where `labels == -100`) are excluded from masking via `target_mask`.

### Generation

- `generate()` — single modality (text-only) inference
- `generate_with_embeds()` — multimodal inference with image embeddings
- Semi-autoregressive: generates in blocks with `gen_length`, `block_length`, `gen_steps` params
- `remasking="low_confidence"` — re-masks least confident predictions each step

### Two-Stage Training

- **Stage 1** (pretrain): Freezes vision encoder + LM. Trains only the MM adapter (resampler + projector). Uses generic image-caption data. LR=1e-3.
- **Stage 2** (finetune): All components trainable. Vision encoder at very low LR (2e-6), model at 1e-5. Uses MIMIC-CXR radiology data. Max length 1024 tokens.

## Data Format

Training data is JSON with LLaVA conversation format:
```json
{
  "id": "dicom-uid",
  "image": "p10/p10000898/s50771383/image.jpg",
  "conversations": [
    {"from": "human", "value": "<image>\nProvide a detailed description... Following clinical context: Indication: ..."},
    {"from": "gpt", "value": "PA and lateral views of the chest..."}
  ]
}
```

Human prompts concatenate: `<image>` token + assistant prompt + clinical context (indication, comparison, history, technique).

## Evaluation

MIMIC-CXR evaluation uses ROUGE-L, METEOR, BLEU-1, BLEU-4 metrics. Task config at `eval/lmms-eval/lmms_eval/tasks/mimic_cxr/`. The eval model class is `llava_oneversion_qwen` with `qwen` conversation template.

## Key Dependencies

- PyTorch 2.x, Transformers, DeepSpeed, PEFT, Accelerate
- `dllm` package (external, installed separately) — diffusion LLM framework
- Training scripts use BF16, gradient checkpointing, SDPA attention
- DeepSpeed configs in `train/scripts/zero2.json`, `zero3.json`, `zero3pp.json`
