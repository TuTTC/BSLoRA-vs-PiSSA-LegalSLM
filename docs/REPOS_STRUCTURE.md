# Giải thích cấu trúc Repository và Code chính

Tài liệu này cung cấp cái nhìn chi tiết về vai trò của từng thư mục, tập tin trong repository **CS431-DoRA-vs-PiSSA-LegalSLM** và giải thích luồng hoạt động của mã nguồn chính.

---

## 1. Cấu trúc Thư mục (Folder Structure)

| Thư mục | Vai trò |
| :--- | :--- |
| `configs/` | Chứa các tập tin cấu hình YAML. `base_config.yaml` giữ các tham số chung, trong khi các file như `lora_config.yaml`, `dora_config.yaml` định nghĩa riêng cho từng phương pháp PEFT. |
| `data/` | Quản lý dữ liệu tập huấn và kiểm thử. |
| &nbsp;&nbsp;├─ `raw/` | Dữ liệu gốc (chưa qua xử lý). |
| &nbsp;&nbsp;├─ `processed/` | Dữ liệu sau khi chạy tiền xử lý, sẵn sàng cho training. |
| &nbsp;&nbsp;└─ `prepare_data.py` | Script chính để chuyển đổi dữ liệu thô sang định dạng Alpaca hoặc ChatML. |
| `training/` | Chứa logic cốt lõi cho quá trình huấn luyện mô hình. |
| &nbsp;&nbsp;├─ `train.py` | **EntryPoint chính** của project để bắt đầu quá trình fine-tuning. |
| &nbsp;&nbsp;└─ `trainer_utils.py` | Các hàm bổ trợ: load model (Unsloth), áp dụng PEFT, và custom Trainer cho LoRA+. |
| `evaluation/` | Chứa các công cụ đánh giá mô hình sau khi huấn luyện. |
| &nbsp;&nbsp;├─ `evaluate.py` | Script chạy đánh giá trên tập test. |
| &nbsp;&nbsp;└─ `metrics.py` | Cài đặt các chỉ số: Perplexity (PPL), ROUGE, BLEU. |
| `utils/` | Các tiện ích hệ thống. |
| &nbsp;&nbsp;├─ `logger.py` | Quản lý log và tích hợp với Weights & Biases (WandB). |
| &nbsp;&nbsp;└─ `helpers.py` | Tiện ích về VRAM tracking, thiết lập random seed, kiểm tra thiết bị. |
| `notebooks/` | Chứa các file Jupyter Notebook để thử nghiệm hoặc phân tích nhanh. |
| `outputs/` | (Thường bị gitignore) Lưu trữ kết quả đầu ra: checkpoints, logs, và kết quả đánh giá. |
| `docs/` | Chứa các tài liệu hướng dẫn, đề cương nghiên cứu. |

---

## 2. Giải thích Code chính (`training/train.py`)

File `train.py` là trái tim của dự án. Dưới đây là luồng thực thi chính:

### Bước 1: Khởi tạo và Cấu hình
- **Đọc tham số**: Script nhận đường dẫn đến file config của phương pháp PEFT (LoRA/DoRA/PiSSA) qua command line.
- **Merge Config**: Hàm `load_config` kết hợp `base_config.yaml` với config cụ thể của phương pháp đó để tạo ra một cấu hình hoàn chỉnh.

### Bước 2: Load Model và Tokenizer
- Sử dụng thư viện **Unsloth** (thông qua `load_model` trong `trainer_utils.py`) để tải mô hình với tốc độ tối ưu và tiết kiệm VRAM (thường là ở định dạng 4-bit).

### Bước 3: Áp dụng kỹ thuật PEFT (Parameter-Efficient Fine-Tuning)
- Hàm `apply_peft` sẽ dựa vào cấu hình để "tiêm" các tham số học thêm vào mô hình:
    - **LoRA**: Thêm các ma trận rank thấp A và B.
    - **DoRA**: Tách biệt hướng (Direction) và độ lớn (Magnitude) để học ổn định hơn.
    - **PiSSA**: Sử dụng SVD để khởi tạo ma trận LoRA nhằm hội tụ nhanh hơn.
    - **BSLoRA**: (Phần nâng cao) Một kỹ thuật chia sẻ trọng số adapter giữa các lớp.

### Bước 4: Chuẩn bị Dữ liệu
- Dữ liệu JSON được load và chuyển đổi sang định dạng prompt (thông qua `format_prompts`) để mô hình hiểu được cấu trúc câu hỏi/trả lời.

### Bước 5: Huấn luyện với Custom Trainer
- Sử dụng `LoraPlusSFTTrainer`. Đây là phiên bản mở rộng của `SFTTrainer` từ thư viện TRL, cho phép sử dụng chiến thuật **LoRA+**:
    - Gán Learning Rate khác nhau cho ma trận A và ma trận B, giúp tối ưu hóa quá trình cập nhật trọng số hiệu quả hơn.

### Bước 6: Theo dõi và Lưu trữ
- **VRAM Tracking**: Theo dõi lượng bộ nhớ GPU tiêu thụ ở từng giai đoạn.
- **WandB**: Đẩy các thông số loss, learning rate lên dashboard trực tuyến.
- **Save**: Sau khi hoàn tất, lưu "Adapter weights" (chỉ nặng vài chục MB thay vì vài GB của model gốc).

---

## 3. Các file quan trọng khác

- **`requirements.txt`**: Danh sách tất cả thư viện cần thiết (`unsloth`, `transformers`, `trl`, `peft`, ...).
- **`setup.py`**: Cho phép cài đặt project như một package.
- **`pipeline_guide.md`**: Hướng dẫn chi tiết quy trình chạy từ đầu đến cuối (End-to-End).
