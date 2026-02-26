"""
Evaluation Metrics
==================
Tính toán Perplexity, ROUGE, BLEU cho mô hình ngôn ngữ.
"""

import math
import torch
import numpy as np
from typing import Dict, List, Any


def compute_perplexity(
    model,
    tokenizer,
    texts: List[str],
    max_length: int = 2048,
    batch_size: int = 4,
) -> float:
    """
    Tính Perplexity trên tập dữ liệu.
    
    PPL = exp(average negative log-likelihood)
    PPL càng thấp => mô hình càng tốt.
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        texts: Danh sách văn bản cần đánh giá
        max_length: Độ dài tối đa
        batch_size: Batch size cho inference
    
    Returns:
        Perplexity score (float)
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            encodings = tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=True,
            ).to(model.device)

            outputs = model(**encodings, labels=encodings["input_ids"])
            
            # Tính loss trên các token không phải padding
            mask = encodings["attention_mask"]
            num_tokens = mask.sum().item()
            total_loss += outputs.loss.item() * num_tokens
            total_tokens += num_tokens

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)

    return perplexity


def compute_rouge(
    predictions: List[str],
    references: List[str],
) -> Dict[str, float]:
    """
    Tính ROUGE-1, ROUGE-2, ROUGE-L scores.
    
    Args:
        predictions: Danh sách câu trả lời do mô hình sinh ra
        references: Danh sách câu trả lời đúng (ground truth)
    
    Returns:
        Dict với keys: rouge1, rouge2, rougeL (F1 scores)
    """
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=False,  # Không dùng stemmer cho tiếng Việt
    )

    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []

    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        rouge1_scores.append(scores["rouge1"].fmeasure)
        rouge2_scores.append(scores["rouge2"].fmeasure)
        rougeL_scores.append(scores["rougeL"].fmeasure)

    return {
        "rouge1": float(np.mean(rouge1_scores)),
        "rouge2": float(np.mean(rouge2_scores)),
        "rougeL": float(np.mean(rougeL_scores)),
    }


def compute_bleu(
    predictions: List[str],
    references: List[str],
) -> float:
    """
    Tính BLEU score.
    
    Args:
        predictions: Danh sách câu trả lời do mô hình sinh ra
        references: Danh sách câu trả lời đúng (ground truth)
    
    Returns:
        BLEU score (float, 0-100)
    """
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

    # Tokenize bằng cách split theo khoảng trắng (đơn giản cho tiếng Việt)
    pred_tokens = [pred.split() for pred in predictions]
    ref_tokens = [[ref.split()] for ref in references]  # Wrap trong list cho corpus_bleu

    smoothing = SmoothingFunction().method1

    bleu = corpus_bleu(ref_tokens, pred_tokens, smoothing_function=smoothing)

    return float(bleu * 100)


def compute_all_metrics(
    model,
    tokenizer,
    predictions: List[str],
    references: List[str],
    eval_texts: List[str] = None,
    compute_ppl: bool = True,
) -> Dict[str, Any]:
    """
    Tính tất cả metrics: Perplexity, ROUGE, BLEU.
    
    Args:
        model: Language model (cần cho Perplexity)
        tokenizer: Tokenizer (cần cho Perplexity)
        predictions: Câu trả lời sinh ra
        references: Ground truth
        eval_texts: Văn bản cho PPL (nếu None, dùng references)
        compute_ppl: Có tính Perplexity không
    
    Returns:
        Dict chứa tất cả metrics
    """
    results = {}

    # ROUGE
    print("[EVAL] Computing ROUGE scores...")
    rouge_scores = compute_rouge(predictions, references)
    results.update(rouge_scores)
    print(f"  ROUGE-1: {rouge_scores['rouge1']:.4f}")
    print(f"  ROUGE-2: {rouge_scores['rouge2']:.4f}")
    print(f"  ROUGE-L: {rouge_scores['rougeL']:.4f}")

    # BLEU
    print("[EVAL] Computing BLEU score...")
    bleu = compute_bleu(predictions, references)
    results["bleu"] = bleu
    print(f"  BLEU: {bleu:.2f}")

    # Perplexity
    if compute_ppl and model is not None:
        print("[EVAL] Computing Perplexity...")
        texts_for_ppl = eval_texts if eval_texts else references
        ppl = compute_perplexity(model, tokenizer, texts_for_ppl)
        results["perplexity"] = ppl
        print(f"  Perplexity: {ppl:.2f}")

    return results
