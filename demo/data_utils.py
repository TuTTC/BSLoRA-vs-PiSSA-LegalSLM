# demo/data_utils.py
from datasets import load_dataset

def load_sample_data(num_samples: int = 20):
    """
    Tải dữ liệu từ VLSP2025 Public Test và format chuẩn như prepare_data.py
    Trả về dictionary chứa list các inputs cho từng task.
    """
    print("⏳ Đang tải dữ liệu Public Test từ HuggingFace để làm mẫu (Sample Data)...")
    dataset_name = "VLSP2025-LegalSML/Public-Test"
    samples = {"task1": [], "task2": [], "task3": []}

    try:
        # TASK 1: nli_questions
        ds_nli = load_dataset(dataset_name, "nli_questions", split="train", streaming=True)
        for i, sample in enumerate(ds_nli):
            if i >= num_samples: break
            user_input = (
                f"Điều luật: {sample['legal_document']}\n"
                f"Câu hỏi: {sample['specific_question']}\n"
                f"{sample['question']}"
            )
            samples["task1"].append(user_input)

        # TASK 2: multichoice_questions
        ds_mcq = load_dataset(dataset_name, "multichoice_questions", split="train", streaming=True)
        for i, sample in enumerate(ds_mcq):
            if i >= num_samples: break
            choices = sample["choices"]
            choices_str = "\n".join(f"{idx}. {c}" for idx, c in enumerate(choices))
            user_input = (
                f"Câu hỏi: {sample['question']}\n"
                f"Lựa chọn:\n{choices_str}"
            )
            samples["task2"].append(user_input)

        # TASK 3: syllogism_questions
        ds_syl = load_dataset(dataset_name, "syllogism_questions", split="train", streaming=True)
        for i, sample in enumerate(ds_syl):
            if i >= num_samples: break
            samples["task3"].append(sample["question"])

        print("✅ Đã tải xong dữ liệu mẫu!")
    except Exception as e:
        print(f"⚠️ Lỗi khi tải sample data: {e}")
        # Dữ liệu backup nếu mất mạng
        samples["task1"] = ["Điều luật: ...\nCâu hỏi: ...\nĐiều luật được cung cấp có thể dùng để trả lời câu hỏi trên hay không?"]
        samples["task2"] = ["Câu hỏi: ...\nLựa chọn:\n0. A\n1. B\n2. C\n3. D"]
        samples["task3"] = ["Người lao động tự ý bỏ việc 5 ngày liên tục thì có bị sa thải không?"]

    return samples