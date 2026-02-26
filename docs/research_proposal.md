# ĐỀ CƯƠNG NGHIÊN CỨU CHI TIẾT

**Tên đề tài:** So sánh hiệu năng của các phương pháp tinh chỉnh tham số hiệu quả (PEFT): DoRA, PiSSA và LoRA trên Mô hình ngôn ngữ lớn cho Tiếng Việt chuyên ngành Pháp luật.

**Quy mô thực hiện:** Nhóm 3 người  
**Thời gian dự kiến:** 8 tuần

---

## 1. Đặt vấn đề và Lý do chọn đề tài (Introduction & Motivation)

Sự phát triển của các Mô hình Ngôn ngữ Lớn (LLMs) đã mở ra nhiều ứng dụng tiềm năng. Tuy nhiên, việc tinh chỉnh toàn bộ tham số (Full Fine-tuning) đòi hỏi chi phí tính toán khổng lồ. Kỹ thuật LoRA (Low-Rank Adaptation) ra đời giúp giảm thiểu tài nguyên nhưng đôi khi gặp hạn chế trong việc bảo toàn khả năng suy luận trên các miền dữ liệu đặc thù phức tạp.

Gần đây, các cải tiến như **DoRA (ICML 2024)** và **PiSSA** đã chứng minh sự vượt trội trên ngôn ngữ tiếng Anh:

* **DoRA** phân tách trọng số thành hai thành phần độc lập: Hướng (Direction) và Độ lớn (Magnitude), giúp mô hình học các đặc trưng mới mà không làm xáo trộn quá lớn tri thức gốc.
* **PiSSA** sử dụng phân rã giá trị suy biến (SVD) để tối ưu hóa không gian khởi tạo, giúp tăng tốc độ hội tụ.

Tuy nhiên, hiệu quả của các phương pháp này trên một ngôn ngữ có tài nguyên trung bình (low/mid-resource) và có cấu trúc từ vựng ghép phức tạp như **Tiếng Việt chuyên ngành (Pháp luật)** vẫn là một khoảng trống nghiên cứu lớn. Đề tài này được thực hiện nhằm lấp đầy khoảng trống đó, đồng thời chứng minh tính khả thi của việc tinh chỉnh các mô hình LLM tiên tiến ngay trên phần cứng dân dụng (GPU 8GB–12GB) thông qua thư viện tối ưu hóa Unsloth.

## 2. Mục tiêu nghiên cứu (Research Objectives)

1. **Mục tiêu lý thuyết:** Khảo sát và phân tích cơ chế toán học cốt lõi của DoRA và PiSSA so với LoRA truyền thống.
2. **Mục tiêu thực nghiệm:** Xây dựng quy trình tinh chỉnh (Fine-tuning pipeline) trên dữ liệu Tiếng Việt chuyên ngành Pháp luật sử dụng thư viện Unsloth.
3. **Mục tiêu đánh giá:** So sánh hiệu năng của 3 phương pháp (LoRA, DoRA, PiSSA) về: Tốc độ hội tụ (Loss convergence), VRAM usage, Perplexity, ROUGE, BLEU.

## 3. Đối tượng và Phạm vi nghiên cứu (Scope of Study)

* **Mô hình nền tảng:** Llama-3.2-3B hoặc Qwen-2.5-7B
* **Dữ liệu:** VLSP2025-LegalSML / VLegal-Bench (chuẩn hóa sang Instruction format)
* **Phần cứng:** GPU VRAM 8GB–24GB

## 4. Phương pháp nghiên cứu (Methodology)

Thực nghiệm song song (Parallel Experimentation) với 3 nhánh:

* **Nhánh 1 (Baseline):** LoRA tiêu chuẩn: ΔW = BA
* **Nhánh 2 (DoRA):** W' = m · (W₀ + BA) / ‖W₀ + BA‖
* **Nhánh 3 (PiSSA):** Khởi tạo ma trận bằng SVD

## 5. Thiết lập thực nghiệm (Experimental Setup)

* **Công cụ:** PyTorch, HuggingFace Transformers, TRL, PEFT, Unsloth, Quantization 4-bit
* **Siêu tham số:** Đồng bộ r, α, batch size, lr, seq_length giữa 3 nhánh
* **Metrics:** Training/Val Loss, Perplexity, ROUGE-1/2/L, BLEU

## 6. Phân công công việc (Timeline)

* **Tuần 1–2:** Thu thập dữ liệu + Setup môi trường + Literature Review
* **Tuần 3–4:** Chạy LoRA baseline + Hyperparameter tuning
* **Tuần 5–6:** Chạy DoRA & PiSSA + WandB logging
* **Tuần 7:** Evaluation + Metrics + Biểu đồ so sánh
* **Tuần 8:** Báo cáo + Ablation Study

## 7. Kết quả dự kiến (Expected Outcomes)

1. **Source Code:** Pipeline hoàn chỉnh (data → train → eval) với Unsloth
2. **Model Checkpoints:** Adapter weights LoRA, DoRA, PiSSA cho luật Việt Nam
3. **Paper/Report:** Báo cáo phân tích chi tiết ưu/nhược điểm từng phương pháp
