import json
import re
import os

def remove_thinking_tags(file_path):
    if not os.path.exists(file_path):
        print(f"⚠️ Không tìm thấy file: {file_path}")
        return

    print(f"⏳ Đang xử lý {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    # Regex bắt toàn bộ nội dung từ <think> đến </think>, bao gồm cả xuống dòng (DOTALL)
    pattern = re.compile(r'<think>.*?</think>\s*', re.DOTALL)
    count = 0

    for item in dataset:
        cleaned_in_this_item = False
        if isinstance(item, dict):
            # Quét càn qua toàn bộ các key (assistant, output, text...) vì data đã bị flatten
            for key, value in item.items():
                if isinstance(value, str) and '<think>' in value:
                    item[key] = re.sub(pattern, '', value).strip()
                    cleaned_in_this_item = True

        if cleaned_in_this_item:
            count += 1

    # Ghi đè lại file cũ
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"✅ Đã làm sạch {count} mẫu trong {file_path}\n")

if __name__ == "__main__":
    train_file = "data/processed/train.json"
    val_file = "data/processed/val.json"

    print("🧹 BẮT ĐẦU DỌN DẸP THẺ <think> CHO NON-THINKING MODE...")
    remove_thinking_tags(train_file)
    remove_thinking_tags(val_file)
    print("🎉 HOÀN TẤT! Dữ liệu đã sẵn sàng để train Baseline.")