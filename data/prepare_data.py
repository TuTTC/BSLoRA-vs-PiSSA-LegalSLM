"""
Data Preparation Script
=======================
Tiền xử lý dữ liệu Pháp luật Tiếng Việt và chuyển sang Instruction format.

Usage:
    python data/prepare_data.py --input_dir data/raw --output_dir data/processed
"""

import os
import json
import argparse
import random
from pathlib import Path
from typing import List, Dict, Any

# ---------------------------------------------------------------------------
# Prompt template (Alpaca-style)
# ---------------------------------------------------------------------------
ALPACA_TEMPLATE = (
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n{output}"
)


# ---------------------------------------------------------------------------
# Cleaning utilities
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str:
    """Làm sạch văn bản: loại bỏ ký tự thừa, chuẩn hóa khoảng trắng."""
    import re
    # Loại bỏ khoảng trắng thừa
    text = re.sub(r"\s+", " ", text).strip()
    # Loại bỏ ký tự đặc biệt không mong muốn
    text = re.sub(r"[^\w\s.,;:!?()\"'àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐ/§\-–—]", "", text)
    return text


def convert_to_instruction_format(sample: Dict[str, Any]) -> Dict[str, str]:
    """
    Chuyển đổi một mẫu dữ liệu sang Instruction format.
    
    Trả về dict với keys: instruction, input, output, text.
    Cần được tùy chỉnh dựa trên cấu trúc dữ liệu gốc.
    """
    # =====================================================================
    # TODO: Tùy chỉnh logic chuyển đổi dựa trên cấu trúc dữ liệu thực tế
    # Ví dụ: dữ liệu QA pháp luật
    # =====================================================================
    instruction = sample.get("instruction", sample.get("question", ""))
    input_text = sample.get("input", sample.get("context", ""))
    output_text = sample.get("output", sample.get("answer", ""))

    instruction = clean_text(instruction)
    input_text = clean_text(input_text)
    output_text = clean_text(output_text)

    formatted_text = ALPACA_TEMPLATE.format(
        instruction=instruction,
        input=input_text,
        output=output_text,
    )

    return {
        "instruction": instruction,
        "input": input_text,
        "output": output_text,
        "text": formatted_text,
    }


# ---------------------------------------------------------------------------
# Data splitting
# ---------------------------------------------------------------------------
def split_data(
    data: List[Dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 3407,
) -> tuple:
    """Chia dữ liệu thành train/val/test."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Tỉ lệ train + val + test phải bằng 1.0"

    random.seed(seed)
    random.shuffle(data)

    n = len(data)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    return data[:train_end], data[train_end:val_end], data[val_end:]


# ---------------------------------------------------------------------------
# I/O
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
    parser = argparse.ArgumentParser(description="Prepare Vietnamese Legal data for fine-tuning")
    parser.add_argument("--input_dir", type=str, default="data/raw",
                        help="Đường dẫn thư mục dữ liệu gốc")
    parser.add_argument("--output_dir", type=str, default="data/processed",
                        help="Đường dẫn thư mục đầu ra")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()

    # 1. Load raw data
    raw_data = load_raw_data(args.input_dir)

    if len(raw_data) == 0:
        print("[WARNING] Không tìm thấy dữ liệu trong thư mục raw. "
              "Hãy đặt file .json hoặc .jsonl vào data/raw/")
        return

    # 2. Convert to instruction format
    processed_data = [convert_to_instruction_format(sample) for sample in raw_data]
    print(f"[DATA] Converted {len(processed_data)} samples to instruction format")

    # 3. Split data
    train_data, val_data, test_data = split_data(
        processed_data,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    print(f"[DATA] Split: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")

    # 4. Save
    save_split(train_data, os.path.join(args.output_dir, "train.json"))
    save_split(val_data, os.path.join(args.output_dir, "val.json"))
    save_split(test_data, os.path.join(args.output_dir, "test.json"))

    print("[DATA]  Data preparation completed!")


if __name__ == "__main__":
    main()
