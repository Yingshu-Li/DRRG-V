# DRRG-V: Diffusion Radiology Report Generation with Vision

A diffusion-based multimodal model for radiology report generation. Combines a SigLIP vision encoder with a Qwen3 0.6B diffusion language model, using complementary masking for semi-autoregressive text generation.

**Pretrained Weights:**
- Language model: [Qwen3-0.6B-diffusion](https://huggingface.co/dllm-collection/Qwen3-0.6B-diffusion-mdlm-v0.1)
- Vision encoder: [SigLIP2-so400m-patch14-384](https://huggingface.co/google/siglip2-so400m-patch14-384)

## Environment Setup

```bash
# Install dLLM framework
git clone https://github.com/ZHZisZZ/dllm.git && cd dllm && pip install -e . && cd ..

# Install training dependencies
cd train && bash init_env.sh

# Install evaluation framework
cd eval/lmms-eval && pip install -e . && pip install -e ".[metrics]"
```

## Data Preparation

Training data uses LLaVA conversation format on MIMIC-CXR:

```json
{
  "id": "0c4eb1e1-b801903c-bcebe8a4-3da9cd3c-3b94a27c",
  "image": "p10/p10000898/s50771383/0c4eb1e1-b801903c-bcebe8a4-3da9cd3c-3b94a27c.jpg",
  "conversations": [
    {
      "from": "human",
      "value": "<image>\nProvide a detailed description of the findings in the radiology image. Following clinical context: Indication: F with chest pain pna"
    },
    {
      "from": "gpt",
      "value": "PA and lateral views of the chest provided. ..."
    }
  ]
}
```

Use `convert_data.py` to convert MIMIC-CXR annotations to this format.

## Training

### Stage 1: Mapper Pretraining
Freezes vision encoder and language model, trains only the multimodal adapter (resampler + projector).

```bash
cd train && bash scripts/llada_v_core.sh <num_nodes> <gpu_num>
```

### Stage 3: Length Prediction
Trains the length prediction head.

```bash
cd train && bash scripts/llada_v_length.sh <num_nodes> <gpu_num>
```

> **Note:** Edit the scripts to set `data_path`, `image_folder`, `model_name_or_path` before running.

## Evaluation

```bash
cd eval && bash scripts/evaluate.sh
```

Edit `MODEL_PATHS` and `CUDA_VISIBLE_DEVICES` in `evaluate.sh` first. Evaluates on MIMIC-CXR using ROUGE-L, METEOR, BLEU-1/4 metrics.

## Acknowledgments

Built upon [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT), [LLaDA-V](https://github.com/ML-GSAI/LLaDA-V), [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval), and [dLLM](https://github.com/ZHZisZZ/dllm).
