import json

train_file_path = "data/processed/train.json"

try:
    with open(train_file_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)

    print(f"Đã tải {len(train_data)} mẫu từ: {train_file_path}\n")
    print("Một vài dữ liệu mẫu từ train_file:")
    for i, sample in enumerate(train_data[:3]): # Hiển thị 3 mẫu đầu tiên
        print(f"-- Mẫu {i+1} --")
        print(json.dumps(sample, indent=2, ensure_ascii=False))
        print("\n")
except FileNotFoundError:
    # Sửa lỗi: Thay dấu nháy kép bao quanh f-string bằng dấu nháy đơn để tránh xung đột
    print(f'Lỗi: Không tìm thấy file {train_file_path}. Hãy đảm bảo bạn đã chạy bước "Chuẩn bị dữ liệu" (Cell 5) trước đó.')
except json.JSONDecodeError:
    print(f"Lỗi: Không thể đọc file JSON từ {train_file_path}. Kiểm tra định dạng file.")

