"""
MIMIC-CXR Report Generation Evaluation Utils

Metrics: ROUGE-L, METEOR, BLEU
"""

import json
import os
import re
from typing import Dict, List, Any, Optional
from collections import defaultdict
import evaluate

from loguru import logger as eval_logger

# Image folder path for MIMIC-CXR images
MIMIC_CXR_IMAGE_FOLDER = "/mnt/sdb/datasets/mimic_original/2.0.0/files"


def mimic_cxr_doc_to_visual(doc):
    """Extract image from document."""
    from PIL import Image
    
    image_path = doc.get("image", "")
    full_path = os.path.join(MIMIC_CXR_IMAGE_FOLDER, image_path)
    
    if os.path.exists(full_path):
        try:
            img = Image.open(full_path).convert("RGB")
            return [img]
        except Exception as e:
            eval_logger.warning(f"Failed to load image {full_path}: {e}")
            return []
    else:
        eval_logger.warning(f"Image not found: {full_path}")
        return []


def mimic_cxr_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    """Extract question/prompt from document."""
    conversations = doc.get("conversations", [])
    
    # Find the human message (prompt)
    for conv in conversations:
        if conv.get("from") == "human":
            prompt = conv.get("value", "")
            # Remove <image> token as it will be added by the framework
            prompt = prompt.replace("<image>\n", "").replace("<image>", "")
            return prompt
    
    return ""


def mimic_cxr_doc_to_target(doc):
    """Extract ground truth report from document."""
    conversations = doc.get("conversations", [])
    
    # Find the gpt message (ground truth)
    for conv in conversations:
        if conv.get("from") == "gpt":
            return conv.get("value", "")
    
    return ""


def compute_bleu(
    predictions: List[str],
    references: List[str],
    bleu_metric: Optional[object] = None,
) -> Dict[str, float]:
    # 1. 基本校验
    if len(predictions) != len(references):
        raise ValueError(f"样本数不一致: preds={len(predictions)} != refs={len(references)}")
    if not predictions:
        return {"bleu_1": 0.0, "bleu_2": 0.0, "bleu_3": 0.0, "bleu_4": 0.0}

    # 4. 准备 metric
    if bleu_metric is None:
        bleu_metric = evaluate.load("bleu")

    # 5. 格式封装
    refs_wrapped = [[r] for r in references]

    scores: Dict[str, float] = {}
    # 计算 BLEU-1 到 BLEU-4
    for k in (1, 2, 3, 4):
        # 即使 preds 中含有空字符串，evaluate.bleu 也能正常处理（不会报错，会给低分）
        res = bleu_metric.compute(predictions=predictions, references=refs_wrapped, max_order=k)
        scores[f"bleu_{k}"] = float(res["bleu"])  # 注意：evaluate 输出通常是 0.0-1.0

    return scores


def compute_rouge(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Compute ROUGE scores."""
    from rouge_score import rouge_scorer
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    
    for pred, ref in zip(predictions, references):
        if not pred or not ref:
            continue
        
        scores = scorer.score(ref, pred)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)
    
    return {
        'rouge_1': sum(rouge1_scores) / len(rouge1_scores) if rouge1_scores else 0,
        'rouge_2': sum(rouge2_scores) / len(rouge2_scores) if rouge2_scores else 0,
        'rouge_l': sum(rougeL_scores) / len(rougeL_scores) if rougeL_scores else 0,
    }


def compute_meteor(predictions: List[str], references: List[str]) -> float:
    """Compute METEOR score."""
    from nltk.translate.meteor_score import meteor_score
    import nltk
    
    # Ensure nltk data is downloaded
    try:
        nltk.data.find('wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)
    try:
        nltk.data.find('omw-1.4')
    except LookupError:
        nltk.download('omw-1.4', quiet=True)
    
    meteor_scores = []
    
    for pred, ref in zip(predictions, references):
        pred_tokens = pred.lower().split()
        ref_tokens = ref.lower().split()
        
        if len(pred_tokens) == 0 or len(ref_tokens) == 0:
            continue
        
        try:
            score = meteor_score([ref_tokens], pred_tokens)
            meteor_scores.append(score)
        except Exception as e:
            eval_logger.warning(f"METEOR computation failed: {e}")
    
    return sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0


# Global counter for debug printing
_DEBUG_PRINT_COUNT = 0
_DEBUG_PRINT_LIMIT = 0


def mimic_cxr_process_results(doc, results):
    """
    Process results for a single document.
    
    Args:
        doc: Document containing ground truth
        results: List containing model prediction
    
    Returns:
        Dictionary with metrics data for aggregation
    """
    global _DEBUG_PRINT_COUNT
    
    pred = results[0].strip() if results else ""
    gt = mimic_cxr_doc_to_target(doc)
    doc_id = doc.get("id", "unknown")
    # prompt = mimic_cxr_doc_to_text(doc)
    # image_path = doc.get("image", "")
    
    # # Debug print first N samples
    # if _DEBUG_PRINT_COUNT < _DEBUG_PRINT_LIMIT:
    #     _DEBUG_PRINT_COUNT += 1
    #     eval_logger.info("=" * 80)
    #     eval_logger.info(f"[DEBUG SAMPLE {_DEBUG_PRINT_COUNT}/{_DEBUG_PRINT_LIMIT}]")
    #     eval_logger.info(f"Doc ID: {doc_id}")
    #     eval_logger.info(f"Image: {image_path}")
    #     eval_logger.info(f"Prompt: {prompt[:200]}..." if len(prompt) > 200 else f"Prompt: {prompt}")
    #     eval_logger.info(f"Ground Truth: {gt[:300]}..." if len(gt) > 300 else f"Ground Truth: {gt}")
    #     eval_logger.info(f"Prediction: {pred[:300]}..." if len(pred) > 300 else f"Prediction: {pred}")
    #     eval_logger.info("=" * 80)
    
    # Store prediction and ground truth for aggregation
    data_dict = {
        "doc_id": doc_id,
        "prediction": pred,
        "ground_truth": gt,
    }
    
    return {
        "rouge_l": data_dict,
        "meteor": data_dict,
        "bleu_1": data_dict,
        "bleu_4": data_dict,
    }


def mimic_cxr_aggregate_results(results: List[Dict]) -> float:
    """
    Aggregate results and compute final metrics.
    
    Args:
        results: List of result dictionaries from process_results
    
    Returns:
        Aggregated metric value
    """
    if not results:
        return 0.0
    
    predictions = [r["prediction"] for r in results]
    references = [r["ground_truth"] for r in results]
    
    # Compute all metrics
    bleu_scores = compute_bleu(predictions, references)
    rouge_scores = compute_rouge(predictions, references)
    meteor = compute_meteor(predictions, references)
    
    # Save detailed results to JSON file
    output_dir = os.environ.get("LMMS_EVAL_OUTPUT_PATH", ".")
    output_file = os.path.join(output_dir, "mimic_cxr_results.json")
    
    detailed_results = {
        "metrics": {
            "bleu_1": bleu_scores['bleu_1'],
            "bleu_2": bleu_scores['bleu_2'],
            "bleu_3": bleu_scores['bleu_3'],
            "bleu_4": bleu_scores['bleu_4'],
            "rouge_1": rouge_scores['rouge_1'],
            "rouge_2": rouge_scores['rouge_2'],
            "rouge_l": rouge_scores['rouge_l'],
            "meteor": meteor,
            "num_samples": len(predictions),
        },
        "samples": [
            {
                "doc_id": r["doc_id"],
                "prediction": r["prediction"],
                "ground_truth": r["ground_truth"],
            }
            for r in results
        ]
    }
    
    try:
        os.makedirs(output_dir, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, indent=2, ensure_ascii=False)
        eval_logger.info(f"Detailed results saved to: {output_file}")
    except Exception as e:
        eval_logger.warning(f"Failed to save detailed results: {e}")
    
    # Print all metrics
    eval_logger.info("=" * 60)
    eval_logger.info("MIMIC-CXR Report Generation Results:")
    eval_logger.info(f"  BLEU-1: {bleu_scores['bleu_1']:.4f}")
    eval_logger.info(f"  BLEU-2: {bleu_scores['bleu_2']:.4f}")
    eval_logger.info(f"  BLEU-3: {bleu_scores['bleu_3']:.4f}")
    eval_logger.info(f"  BLEU-4: {bleu_scores['bleu_4']:.4f}")
    eval_logger.info(f"  ROUGE-1: {rouge_scores['rouge_1']:.4f}")
    eval_logger.info(f"  ROUGE-2: {rouge_scores['rouge_2']:.4f}")
    eval_logger.info(f"  ROUGE-L: {rouge_scores['rouge_l']:.4f}")
    eval_logger.info(f"  METEOR: {meteor:.4f}")
    eval_logger.info(f"  Samples: {len(predictions)}")
    eval_logger.info("=" * 60)
    
    # Return ROUGE-L as the primary metric (this will be called for each metric type)
    # The framework will use the metric name to determine which value to return
    return rouge_scores['rouge_l']
