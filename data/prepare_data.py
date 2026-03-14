"""
Data Preparation Script (Multi-task)
=====================================
Tiền xử lý dữ liệu Pháp luật Tiếng Việt cho 3 task:
  - Task 1: Legal Citation / NLI (Có/Không)
  - Task 2: Multiple Choice Questions (Trắc nghiệm)
  - Task 3: Open-ended / Syllogism Questions (Tự luận)

Hỗ trợ nguồn dữ liệu:
  - HuggingFace train:  QuangTran276/new_reasoning  (conversation format)
  - HuggingFace test:   VLSP2025-LegalSML/Public-Test (3 configs)
  - Local files:        data/raw/ (JSON/JSONL)

Usage:
    # Từ HuggingFace + Public Test (khuyến nghị)
    python data/prepare_data.py \\
        --hf_dataset QuangTran276/new_reasoning \\
        --public_test VLSP2025-LegalSML/Public-Test

    # Chỉ từ HuggingFace (tự split test)
    python data/prepare_data.py --hf_dataset QuangTran276/new_reasoning

    # Từ local files
    python data/prepare_data.py --input_dir data/raw
"""

import os
import json
import re
import argparse
import random
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from datasets import load_dataset


# ---------------------------------------------------------------------------
# System Prompts cho từng task (dùng để auto-classify)
# ---------------------------------------------------------------------------
SYSTEM_PROMPTS = {
    "task1": (
        "Bạn là một chuyên gia pháp luật Việt Nam. "
        "Nhiệm vụ của bạn là xác định xem một điều luật "
        "có thể được sử dụng để trả lời câu hỏi pháp lý cụ thể hay không."
    ),
    "task2": (
        "Bạn là một chuyên gia pháp luật Việt Nam. "
        "Hãy trả lời câu hỏi trắc nghiệm sau dựa trên "
        "văn bản pháp luật được cung cấp."
    ),
    "task3": (
        "Bạn là một chuyên gia pháp luật Việt Nam. "
        "Hãy trả lời câu hỏi mở sau theo cấu trúc "
        "lập luận pháp lý chuyên sâu."
    ),
}

# Mapping keyword → task_type (dùng khi system prompt không khớp chính xác)
TASK_KEYWORDS = {
    "task1": ["xác định xem một điều luật", "trả lời câu hỏi pháp lý cụ thể"],
    "task2": ["trắc nghiệm", "multiple choice", "câu hỏi trắc nghiệm"],
    "task3": ["câu hỏi mở", "tự luận", "lập luận pháp lý"],
}


# ---------------------------------------------------------------------------
# Auto-classify task type from system prompt
# ---------------------------------------------------------------------------
def classify_task_from_system(system_prompt: str) -> str:
    """
    Phân loại task dựa trên nội dung system prompt.

    So sánh exact match trước, rồi fallback sang keyword matching.
    """
    # Exact match
    for task_type, prompt in SYSTEM_PROMPTS.items():
        if system_prompt.strip() == prompt.strip():
            return task_type

    # Keyword match
    system_lower = system_prompt.lower()
    for task_type, keywords in TASK_KEYWORDS.items():
        if any(kw.lower() in system_lower for kw in keywords):
            return task_type

    # Fallback
    return "task3"


# ---------------------------------------------------------------------------
# Convert HuggingFace conversation format → ChatML
# ---------------------------------------------------------------------------
def convert_messages_to_chatml(messages: List[Dict[str, str]]) -> Dict[str, str]:
    """
    Chuyển đổi conversation messages [{role, content}, ...] sang ChatML format.

    Input (từ HuggingFace dataset):
        [
            {"role": "system",    "content": "..."},
            {"role": "user",      "content": "..."},
            {"role": "assistant", "content": "<think>...</think> Có"},
        ]

    Output:
        {
            "system": "...",
            "user": "...",
            "assistant": "<think>...</think> Có",
            "task_type": "task1",
            "text": "<|im_start|>system\n...<|im_end|>\n...",
        }
    """
    # Trích xuất các role
    system_content = ""
    user_content = ""
    assistant_content = ""

    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            system_content = content
        elif role == "user":
            user_content = content
        elif role == "assistant":
            assistant_content = content

    # Auto-classify task type
    task_type = classify_task_from_system(system_content)

    # Build Qwen3 ChatML formatted text
    chatml_text = (
        f"<|im_start|>system\n{system_content}<|im_end|>\n"
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        f"<|im_start|>assistant\n{assistant_content}<|im_end|>"
    )

    return {
        "system": system_content,
        "user": user_content,
        "assistant": assistant_content,
        "task_type": task_type,
        "text": chatml_text,
        # Backward compatibility
        "instruction": user_content,
        "input": "",
        "output": assistant_content,
    }


# ---------------------------------------------------------------------------
# Load from HuggingFace (training data)
# ---------------------------------------------------------------------------
def load_hf_dataset(dataset_name: str) -> List[Dict]:
    """
    Load dataset từ HuggingFace và convert sang ChatML format.

    Args:
        dataset_name: Tên dataset trên HuggingFace (vd: "QuangTran276/new_reasoning")

    Returns:
        List[Dict] đã convert sang ChatML format
    """
    print(f"[DATA] Loading HuggingFace dataset: {dataset_name}")
    ds = load_dataset(dataset_name, split="train")
    print(f"[DATA] Loaded {len(ds)} samples from HuggingFace")

    processed = []
    for sample in ds:
        messages = sample["messages"]
        converted = convert_messages_to_chatml(messages)
        processed.append(converted)

    return processed


# ---------------------------------------------------------------------------
# Load Public Test from HuggingFace (VLSP2025-LegalSML/Public-Test)
# ---------------------------------------------------------------------------
def load_public_test(dataset_name: str) -> List[Dict]:
    """
    Load VLSP2025 Public Test dataset và convert sang ChatML format.

    Dataset có 3 configs:
      - multichoice_questions (task2): question, choices[], answer (int)
      - nli_questions         (task1): legal_document, specific_question,
                                       question, choices[], answer (int)
      - syllogism_questions   (task3): question, answer (string)

    Args:
        dataset_name: "VLSP2025-LegalSML/Public-Test"

    Returns:
        List[Dict] đã convert sang ChatML format
    """
    print(f"[DATA] Loading Public Test: {dataset_name}")
    all_test = []

    # --- Config 1: multichoice_questions → task2 ---
    ds_mcq = load_dataset(dataset_name, "multichoice_questions", split="train")
    print(f"[DATA]   multichoice_questions: {len(ds_mcq)} samples")
    for sample in ds_mcq:
        choices = sample["choices"]
        choices_str = "\n".join(f"{i}. {c}" for i, c in enumerate(choices))
        user_input = (
            f"Câu hỏi: {sample['question']}\n"
            f"Lựa chọn:\n{choices_str}"
        )
        answer = str(sample["answer"])  # int → str
        system_prompt = SYSTEM_PROMPTS["task2"]

        chatml_text = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{user_input}<|im_end|>\n"
            f"<|im_start|>assistant\n{answer}<|im_end|>"
        )
        all_test.append({
            "system": system_prompt,
            "user": user_input,
            "assistant": answer,
            "task_type": "task2",
            "text": chatml_text,
            "instruction": user_input,
            "input": "",
            "output": answer,
        })

    # --- Config 2: nli_questions → task1 ---
    ds_nli = load_dataset(dataset_name, "nli_questions", split="train")
    print(f"[DATA]   nli_questions: {len(ds_nli)} samples")
    for sample in ds_nli:
        user_input = (
            f"Điều luật: {sample['legal_document']}\n"
            f"Câu hỏi: {sample['specific_question']}\n"
            f"{sample['question']}"
        )
        # answer là int index vào choices ("Có"=0, "Không"=1)
        choices = sample["choices"]
        answer = choices[sample["answer"]] if sample["answer"] < len(choices) else str(sample["answer"])
        system_prompt = SYSTEM_PROMPTS["task1"]

        chatml_text = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{user_input}<|im_end|>\n"
            f"<|im_start|>assistant\n{answer}<|im_end|>"
        )
        all_test.append({
            "system": system_prompt,
            "user": user_input,
            "assistant": answer,
            "task_type": "task1",
            "text": chatml_text,
            "instruction": user_input,
            "input": "",
            "output": answer,
        })

    # --- Config 3: syllogism_questions → task3 ---
    ds_syl = load_dataset(dataset_name, "syllogism_questions", split="train")
    print(f"[DATA]   syllogism_questions: {len(ds_syl)} samples")
    for sample in ds_syl:
        user_input = sample["question"]
        answer = sample["answer"]
        system_prompt = SYSTEM_PROMPTS["task3"]

        chatml_text = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{user_input}<|im_end|>\n"
            f"<|im_start|>assistant\n{answer}<|im_end|>"
        )
        all_test.append({
            "system": system_prompt,
            "user": user_input,
            "assistant": answer,
            "task_type": "task3",
            "text": chatml_text,
            "instruction": user_input,
            "input": "",
            "output": answer,
        })

    print(f"[DATA] Total Public Test samples: {len(all_test)}")
    return all_test


# ---------------------------------------------------------------------------
# Load from local files (fallback)
# ---------------------------------------------------------------------------
def load_raw_data(input_dir: str) -> List[Dict]:
    """
    Load dữ liệu từ thư mục raw.
    Hỗ trợ: .json, .jsonl
    """
    data = []
    input_path = Path(input_dir)

    for file_path in sorted(input_path.glob("**/*")):
        if file_path.suffix == ".json":
            with open(file_path, "r", encoding="utf-8") as f:
                content = json.load(f)
                if isinstance(content, list):
                    data.extend(content)
                else:
                    data.append(content)
        elif file_path.suffix == ".jsonl":
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data.append(json.loads(line))

    print(f"[DATA] Loaded {len(data)} samples from {input_dir}")
    return data


def convert_local_sample(sample: Dict[str, Any]) -> Dict[str, str]:
    """
    Convert local sample format sang ChatML.

    Hỗ trợ 2 format:
      - Conversation format: sample có key "messages" (list of {role, content})
      - Legacy format: sample có keys riêng biệt
    """
    # Nếu sample đã có messages format (conversation)
    if "messages" in sample:
        return convert_messages_to_chatml(sample["messages"])

    # Legacy format: phân loại task + build user input
    task_type = _classify_task_legacy(sample)
    system_prompt = SYSTEM_PROMPTS[task_type]
    user_input = _build_user_input(sample, task_type)
    raw_output = sample.get("output", sample.get("answer", sample.get("response", "")))

    chatml_text = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_input}<|im_end|>\n"
        f"<|im_start|>assistant\n{raw_output}<|im_end|>"
    )

    return {
        "system": system_prompt,
        "user": user_input,
        "assistant": raw_output,
        "task_type": task_type,
        "text": chatml_text,
        "instruction": user_input,
        "input": "",
        "output": raw_output,
    }


# ---------------------------------------------------------------------------
# Legacy helpers (cho local files không có conversation format)
# ---------------------------------------------------------------------------
def _classify_task_legacy(sample: Dict[str, Any]) -> str:
    """Phân loại task dựa trên cấu trúc sample (legacy format)."""
    if "task_type" in sample:
        return sample["task_type"]
    for key in ("task1", "task2", "task3"):
        if key in sample:
            return key
    if "legal_document" in sample or "specific_question" in sample:
        return "task1"
    if "choices" in sample or "choice_0" in sample:
        return "task2"
    return "task3"


def _build_user_input(sample: Dict[str, Any], task_type: str) -> str:
    """Build user input cho legacy format."""
    if task_type == "task1":
        legal_doc = sample.get("legal_document", sample.get("context", ""))
        question = sample.get("specific_question", sample.get("question", ""))
        return (
            f"Điều luật: {legal_doc}\n"
            f"Câu hỏi: {question}\n"
            f"Điều luật được cung cấp có thể dùng để trả lời câu hỏi trên hay không?"
        )
    elif task_type == "task2":
        context = sample.get("document_context", sample.get("context", ""))
        question = sample.get("question", "")
        if "choices" in sample:
            choices = sample["choices"]
        else:
            choices = [sample.get(f"choice_{i}", "") for i in range(4)]
        choices_str = "\n".join(f"{i}. {c}" for i, c in enumerate(choices))
        return (
            f"Văn bản: {context}\n"
            f"Câu hỏi: {question}\n"
            f"Lựa chọn:\n{choices_str}"
        )
    else:  # task3
        return sample.get("question", sample.get("instruction", ""))


# ---------------------------------------------------------------------------
# Data splitting
# ---------------------------------------------------------------------------
def split_data(
    data: List[Dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 3407,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Chia dữ liệu thành train/val/test."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Tỷ lệ train + val + test phải bằng 1.0"

    random.seed(seed)
    shuffled = list(data)
    random.shuffle(shuffled)

    n = len(shuffled)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    return shuffled[:train_end], shuffled[train_end:val_end], shuffled[val_end:]


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def save_split(data: List[Dict], output_path: str) -> None:
    """Lưu dữ liệu dưới dạng JSON."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[DATA] Saved {len(data)} samples to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Prepare Vietnamese Legal data for multi-task fine-tuning"
    )
    parser.add_argument(
        "--hf_dataset", type=str, default=None,
        help="Tên HuggingFace dataset (vd: QuangTran276/new_reasoning)",
    )
    parser.add_argument(
        "--public_test", type=str, default=None,
        help="Tên HuggingFace Public Test dataset (vd: VLSP2025-LegalSML/Public-Test)",
    )
    parser.add_argument(
        "--input_dir", type=str, default="data/raw",
        help="Đường dẫn thư mục dữ liệu gốc (dùng nếu không có --hf_dataset)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="data/processed",
        help="Đường dẫn thư mục đầu ra",
    )
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()

    # =====================================================================
    # 1. Load training data
    # =====================================================================
    if args.hf_dataset:
        processed_data = load_hf_dataset(args.hf_dataset)
    else:
        raw_data = load_raw_data(args.input_dir)
        if len(raw_data) == 0:
            print(
                "[WARNING] Không tìm thấy dữ liệu. "
                "Hãy dùng --hf_dataset hoặc đặt file vào data/raw/"
            )
            return
        processed_data = [convert_local_sample(s) for s in raw_data]

    print(f"[DATA] Converted {len(processed_data)} samples to ChatML format")

    # Thống kê phân bố task
    task_counts = {}
    for item in processed_data:
        t = item.get("task_type", "unknown")
        task_counts[t] = task_counts.get(t, 0) + 1
    print(f"[DATA] Task distribution: {task_counts}")

    # =====================================================================
    # 2. Load Public Test (nếu có) hoặc split từ training data
    # =====================================================================
    if args.public_test:
        # Có Public Test → split training data thành train/val only
        public_test_data = load_public_test(args.public_test)

        # Split train data thành train + val (không cần test)
        val_ratio_adjusted = args.val_ratio / (args.train_ratio + args.val_ratio)
        train_ratio_adjusted = 1.0 - val_ratio_adjusted

        train_data, val_data, _ = split_data(
            processed_data,
            train_ratio=train_ratio_adjusted,
            val_ratio=val_ratio_adjusted,
            test_ratio=0.0,
            seed=args.seed,
        )
        test_data = public_test_data
        print(f"[DATA] Split: train={len(train_data)}, val={len(val_data)}")
        print(f"[DATA] Public Test: {len(test_data)} samples")
    else:
        # Không có Public Test → split 3 way
        train_data, val_data, test_data = split_data(
            processed_data,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )
        print(f"[DATA] Split: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")

    # =====================================================================
    # 3. Save
    # =====================================================================
    save_split(train_data, os.path.join(args.output_dir, "train.json"))
    save_split(val_data, os.path.join(args.output_dir, "val.json"))
    save_split(test_data, os.path.join(args.output_dir, "test.json"))

    print("[DATA]  Data preparation completed!")


if __name__ == "__main__":
    main()
