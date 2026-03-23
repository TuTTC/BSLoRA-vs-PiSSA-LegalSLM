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

    # Regex bắt toàn bộ nội dung từ <think> đến </think>, bao gồm cả dấu xuống dòng (DOTALL)
    pattern = re.compile(r'<think>.*?</think>\s*', re.DOTALL)
    count = 0

    for item in dataset:
        # Dữ liệu thường nằm trong key 'messages' hoặc dạng list trực tiếp
        messages = item.get('messages', item) if isinstance(item, dict) else item
        
        if isinstance(messages, list):
            for turn in messages:
                if turn.get('role') == 'assistant' and '<think>' in turn.get('content', ''):
                    # Xóa phần think và strip khoảng trắng thừa
                    original_text = turn['content']
                    cleaned_text = re.sub(pattern, '', original_text).strip()
                    turn['content'] = cleaned_text
                    count += 1

    # Ghi đè lại file cũ
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Đã làm sạch {count} mẫu trong {file_path}\n")

# Đường dẫn đến các file data đã xử lý
train_file = "data/processed/train.json"
val_file = "data/processed/val.json"

remove_thinking_tags(train_file)
remove_thinking_tags(val_file)