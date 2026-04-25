import json
import argparse
import numpy as np
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    print("[LỖI] Thiếu thư viện scikit-learn. Vui lòng chạy: pip install scikit-learn")
    exit(1)

def extract_text_from_item(item):
    """Trích xuất toàn bộ text từ một item trong cục training data (xử lý đa format)"""
    text = ""
    # Nếu là chuẩn ChatML (có messages)
    if "messages" in item:
        for msg in item["messages"]:
            text += msg.get("content", "") + " "
    # Nếu là chuẩn ShareGPT (có conversations)
    elif "conversations" in item:
        for conv in item["conversations"]:
            text += conv.get("value", "") + " "
    # Nếu format phẳng (bình thường)
    elif "user_input" in item or "instruction" in item:
        text += item.get("user_input", item.get("instruction", "")) + " "
        text += item.get("reference", item.get("output", "")) + " "
    # Nếu là string
    elif isinstance(item, str):
        text = item
    else:
        # Gom góp tất cả chuỗi tìm được
        text = " ".join([str(v) for v in item.values() if isinstance(v, str)])
    return text.strip()

def main():
    parser = argparse.ArgumentParser(description="Đo độ tương đồng TF-IDF + Cosine Similarity giữa Train Data và Test Data")
    parser.add_argument("--train_file", type=str, default="data/processed/train.json", help="File dữ liệu huấn luyện của LegalSLM")
    parser.add_argument("--test_file", type=str, default="outputs-final/outputs/results/pissa_task3_detailed_responses_thinking.json", help="File chứa câu hỏi VLegal-Bench (vd: file output JSON của bạn)")
    parser.add_argument("--sample_train", type=int, default=15000, help="Lấy ngẫu nhiên N mẫu từ tập train để đo (tránh tràn RAM). Để -1 để lấy hết.")
    args = parser.parse_args()

    print(f"[*] Đang tải dữ liệu Test (VLegal-Bench) từ: {args.test_file}")
    with open(args.test_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    # Lấy ra các câu hỏi trong đề thi (giả định có trường user_input)
    test_texts = [item.get("user_input", "") + " " + item.get("reference", "") for item in test_data if isinstance(item, dict)]
    print(f"    -> Đã tải {len(test_texts)} mẫu Test.")

    print(f"[*] Đang tải dữ liệu Train (LegalSLM) từ: {args.train_file}")
    with open(args.train_file, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    print(f"    -> Đã tải tổng cộng {len(train_data)} mẫu Train.")

    # Rút gọn tập train nếu quá lớn
    if args.sample_train > 0 and args.sample_train < len(train_data):
        print(f"[*] Đang Lấy ngẫu nhiên {args.sample_train} mẫu từ tập Train để so sánh...")
        import random
        random.seed(42)
        train_data = random.sample(train_data, args.sample_train)
    
    train_texts = [extract_text_from_item(item) for item in train_data]

    print("[*] Đang xây dựng không gian vectors TF-IDF (Có thể mất 1-2 phút)...")
    vectorizer = TfidfVectorizer(max_df=0.85, min_df=2) # Loại bỏ từ xuất hiện ở >85% văn bản
    # Khớp vectorizer trên cả train và test để tạo không gian từ vựng chung
    all_texts = train_texts + test_texts
    vectorizer.fit(all_texts)

    train_vectors = vectorizer.transform(train_texts)
    test_vectors = vectorizer.transform(test_texts)

    print("[*] Đang tính toán ma trận độ tương đồng Cosine (Cosine Similarity)...")
    # Ma trận kết quả có dạng: [Số mẫu Test] x [Số mẫu Train]
    similarities = cosine_similarity(test_vectors, train_vectors)

    # Thống kê
    max_sims_per_test = np.max(similarities, axis=1) # Với mỗi câu test, tìm độ tương đồng với câu train giống nhất
    avg_max_sim = np.mean(max_sims_per_test)
    
    # Đếm tỷ lệ rò rỉ dữ liệu (Leakage) - ngưỡng 80% giống nhau (0.8) coi như bị leak/trùng lặp
    leakage_threshold = 0.8
    leaked_samples = np.sum(max_sims_per_test >= leakage_threshold)

    print("\n" + "="*60)
    print(" KẾT QUẢ ĐÁNH GIÁ ĐỘ TƯƠNG ĐỒNG DỮ LIỆU (DATA LEAKAGE / ALIGNMENT) ")
    print("="*60)
    print(f"- Số mẫu Test VLegal-Bench      : {len(test_texts)} mẫu")
    print(f"- Số mẫu Train LegalSLM so sánh : {len(train_texts)} mẫu")
    print(f"- Độ tương đồng trung bình lớn nhất (Avg Max Similarity): {avg_max_sim*100:.2f}%")
    print("  (Chỉ số này ở mức < 15% là bình thường, > 50% là có hiện tượng đạo văn/leakage)")
    print(f"- Số lượng câu thi Test bị trùng lặp sát với Train (>80%): {leaked_samples}/{len(test_texts)} câu thi ({(leaked_samples/len(test_texts))*100:.2f}%)")
    
    print("\n[PHÂN TÍCH CHO PAPER]")
    if avg_max_sim < 0.2:
        print("=> Tốt: Dữ liệu Train (LegalSLM) và Test (VLegal-Bench) hầu như độc lập. Kết quả điểm cao của mô hình chứng tỏ nó ĐÃ HIỂU tư duy luật, chứ KHÔNG PHẢI học vẹt trúng tủ.")
    elif avg_max_sim < 0.5:
        print("=> Chấp nhận được: Có sự tương đồng nhỏ về chủ đề/Luật, nhưng không bị trùng lắp trực tiếp đề thi.")
    else:
        print("=> Nguy hiểm (Data Leakage): Khả năng cao đề thi VLegal-Bench đã bị lọt vào tập huấn luyện của bạn. Cần loại bỏ các mẫu trùng này để đánh giá chính xác!")

if __name__ == "__main__":
    main()
