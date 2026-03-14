"""
Evaluation Metrics (Multi-task)
================================
Tính toán metrics cho 3 task pháp luật:
  - Task 1 (Citation / NLI):  Accuracy  (Eq. 7)
  - Task 2 (MCQ):             Accuracy  (Eq. 7)
  - Task 3 (Open-ended QA):   Exact Match with normalization (Eq. 8)
  - Chung:                    Perplexity
"""

import re
import math
import string
import torch
import numpy as np
from typing import Dict, List, Any, Optional


# ============================================================================
# Text normalization ν(·) for Exact Match  (Eq. 8)
# ============================================================================
def _normalize_text(text: str) -> str:
    """
    Light normalization ν(·): lowercasing, punctuation removal,
    and whitespace collapsing / stripping.
    """
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = " ".join(text.split())
    return text


# ============================================================================
# Perplexity (chung cho tất cả task)
# ============================================================================
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


# ============================================================================
# Accuracy  (Eq. 7)  –  dùng cho classification tasks (Task 1 & Task 2)
# ============================================================================
def compute_accuracy(
    predictions: List[Any],
    references: List[Any],
) -> float:
    """
    Acc = (1/N) * Σ 1[ŷ_i = y_i]

    So sánh trực tiếp predicted label với gold label.
    """
    n = len(predictions)
    if n == 0:
        return 0.0
    correct = sum(1 for y_hat, y in zip(predictions, references) if y_hat == y)
    return correct / n


# ============================================================================
# Exact Match  (Eq. 8)  –  dùng cho free-form QA (Task 3)
# ============================================================================
def compute_exact_match(
    predictions: List[str],
    references: List[str],
) -> float:
    """
    EM = (1/N) * Σ 1[ν(â_i) = ν(a_i)]

    So sánh câu trả lời sau normalization ν(·)
    (lowercasing, punctuation & whitespace removal).
    """
    n = len(predictions)
    if n == 0:
        return 0.0
    correct = sum(
        1 for pred, ref in zip(predictions, references)
        if _normalize_text(pred) == _normalize_text(ref)
    )
    return correct / n


# ============================================================================
# Helpers: trích xuất label từ raw model output
# ============================================================================
def _extract_answer_after_think(text: str) -> str:
    """
    Trích xuất câu trả lời sau thẻ </think>.
    Nếu không có thẻ </think>, lấy toàn bộ text.
    """
    match = re.search(r"</think>\s*(.*)", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def _normalize_yes_no(answer: str) -> Optional[str]:
    """Chuẩn hóa câu trả lời Có/Không (Task 1 – NLI / Citation)."""
    answer_lower = answer.lower().strip()
    if re.search(r"\bcó\b", answer_lower):
        return "Có"
    if re.search(r"\bkhông\b", answer_lower):
        return "Không"
    return None


def _extract_mcq_answer(text: str) -> Optional[int]:
    """
    Trích xuất index đáp án từ text (Task 2 – MCQ).
    Tìm pattern: "Đáp án đúng là: X" hoặc "Đáp án: X" hoặc chỉ số X.
    """
    answer_text = _extract_answer_after_think(text)

    # Pattern 1: "Đáp án đúng là: 2" hoặc "Đáp án đúng là: 2."
    match = re.search(r"[Đđ]áp\s*án\s*(?:đúng\s*là)?\s*:?\s*(\d)", answer_text)
    if match:
        return int(match.group(1))

    # Pattern 2: tìm số đầu tiên (0-3)
    match = re.search(r"\b([0-3])\b", answer_text)
    if match:
        return int(match.group(1))

    return None


# ============================================================================
# Task-level wrappers
# ============================================================================
def compute_citation_accuracy(
    predictions: List[str],
    references: List[str],
) -> Dict[str, float]:
    """
    Accuracy cho Task 1 (Citation / NLI).
    Trích xuất "Có"/"Không" rồi tính Acc (Eq. 7).
    """
    y_pred = []
    y_true = []
    parse_failures = 0

    for pred, ref in zip(predictions, references):
        pred_label = _normalize_yes_no(_extract_answer_after_think(pred))
        ref_label = _normalize_yes_no(_extract_answer_after_think(ref))

        if pred_label is None or ref_label is None:
            parse_failures += 1
            pred_label = pred_label or "Không"
            ref_label = ref_label or "Không"

        y_pred.append(pred_label)
        y_true.append(ref_label)

    total = len(predictions)
    accuracy = compute_accuracy(y_pred, y_true)
    parse_rate = (total - parse_failures) / total if total > 0 else 0.0

    return {
        "citation_accuracy": float(accuracy),
        "citation_parse_rate": float(parse_rate),
    }


def compute_mcq_accuracy(
    predictions: List[str],
    references: List[str],
) -> Dict[str, float]:
    """
    Accuracy cho Task 2 (MCQ).
    Trích xuất index đáp án (0-3) rồi tính Acc (Eq. 7).
    """
    y_pred = []
    y_true = []
    parse_failures = 0

    for pred, ref in zip(predictions, references):
        pred_idx = _extract_mcq_answer(pred)
        ref_idx = _extract_mcq_answer(ref)

        if pred_idx is None or ref_idx is None:
            parse_failures += 1
            continue

        y_pred.append(pred_idx)
        y_true.append(ref_idx)

    total = len(predictions)
    accuracy = compute_accuracy(y_pred, y_true)
    parse_rate = (total - parse_failures) / total if total > 0 else 0.0

    return {
        "mcq_accuracy": float(accuracy),
        "mcq_parse_rate": float(parse_rate),
    }


def compute_qa_exact_match(
    predictions: List[str],
    references: List[str],
) -> Dict[str, float]:
    """
    Exact Match cho Task 3 (Open-ended QA).
    Trích xuất câu trả lời sau </think> rồi tính EM (Eq. 8).
    """
    preds_clean = [_extract_answer_after_think(p) for p in predictions]
    refs_clean = [_extract_answer_after_think(r) for r in references]

    em = compute_exact_match(preds_clean, refs_clean)

    return {
        "qa_exact_match": float(em),
    }


# ============================================================================
# Master metric dispatcher
# ============================================================================
def compute_all_metrics(
    model=None,
    tokenizer=None,
    predictions: List[str] = None,
    references: List[str] = None,
    eval_texts: List[str] = None,
    compute_ppl: bool = True,
    task_type: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Tính metrics phù hợp với task_type.

    Args:
        model:       Language model (cần cho Perplexity)
        tokenizer:   Tokenizer (cần cho Perplexity)
        predictions: Câu trả lời sinh ra
        references:  Ground truth
        eval_texts:  Văn bản cho PPL (nếu None, dùng references)
        compute_ppl: Có tính Perplexity không
        task_type:   "task1" | "task2" | "task3" | None (tính tất cả)

    Returns:
        Dict chứa metrics phù hợp
    """
    results = {}

    # ----- Task 1: Citation Classification (Accuracy – Eq. 7) -----
    if task_type in (None, "task1"):
        print("[EVAL] Computing Citation Accuracy (Task 1)...")
        citation_metrics = compute_citation_accuracy(predictions, references)
        results.update(citation_metrics)
        print(f"  Accuracy:   {citation_metrics['citation_accuracy']:.4f}")
        print(f"  Parse rate: {citation_metrics['citation_parse_rate']:.4f}")

    # ----- Task 2: Multiple Choice (Accuracy – Eq. 7) -----
    if task_type in (None, "task2"):
        print("[EVAL] Computing MCQ Accuracy (Task 2)...")
        mcq_metrics = compute_mcq_accuracy(predictions, references)
        results.update(mcq_metrics)
        print(f"  MCQ Accuracy:   {mcq_metrics['mcq_accuracy']:.4f}")
        print(f"  MCQ Parse rate: {mcq_metrics['mcq_parse_rate']:.4f}")

    # ----- Task 3: Open-ended QA (Exact Match – Eq. 8) -----
    if task_type in (None, "task3"):
        print("[EVAL] Computing Exact Match (Task 3)...")
        qa_metrics = compute_qa_exact_match(predictions, references)
        results.update(qa_metrics)
        print(f"  Exact Match: {qa_metrics['qa_exact_match']:.4f}")

    # ----- Perplexity (chung) -----
    if compute_ppl and model is not None:
        print("[EVAL] Computing Perplexity...")
        texts_for_ppl = eval_texts if eval_texts else references
        ppl = compute_perplexity(model, tokenizer, texts_for_ppl)
        results["perplexity"] = ppl
        print(f"  Perplexity: {ppl:.2f}")

    return results
