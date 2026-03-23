<!-- Banner -->
<p align='center'>
  <a href="https://www.uit.edu.vn/" title="Trường Đại học Công nghệ Thông tin" style="border: none;">
     <img src="https://i.imgur.com/WmMnSRt.png" alt="Trường Đại học Công nghệ Thông tin | University of Information Technology">
  </a>
</p>

<h1 align="center"><b>CÁC KỸ THUẬT HỌC SÂU VÀ ỨNG DỤNG - CS431</b></h1>

<h3 align="center">
  So sánh hiệu năng DoRA, PiSSA và LoRA trên LLM<br>
  cho Tiếng Việt chuyên ngành Pháp luật
</h3>

---

## 📋 Giới thiệu môn học

| Thông tin       | Chi tiết                         |
| --------------- | -------------------------------- |
| **Tên môn học** | Các kỹ thuật Học Sâu và Ứng dụng |
| **Mã môn học**  | CS431                            |
| **Mã lớp**      | CS431.Q22                        |
| **Giảng viên**  | TS. Nguyễn Vinh Tiệp             |

## 👥 Thành viên nhóm

|  STT  |   MSSV   | Họ và Tên           |   Chức vụ   | GitHub                                                     | Email                  |
| :---: | :------: | ------------------- | :---------: | ---------------------------------------------------------- | ---------------------- |
|   1   | 23521704 | Trần Thị Cẩm Tú     | Nhóm trưởng | [@TuTTC](https://github.com/TuTTC)                         | 23521704@gm.uit.edu.vn |
|   2   | 23521821 | Lê Ngọc Phương Thảo | Thành viên  | [@]()                                                      | 23521821@gm.uit.edu.vn |
|   3   | 23521193 | Đinh Hoàng Phúc     | Thành viên  | [@DinhHoangPhuc3010](https://github.com/DinhHoangPhuc3010) | 23521193@gm.uit.edu.vn |

---

## 🔬 Tổng quan đề tài

Nghiên cứu so sánh ba phương pháp tinh chỉnh tham số hiệu quả (PEFT) trên mô hình ngôn ngữ lớn cho tiếng Việt chuyên ngành Pháp luật:

| Phương pháp | Cơ chế                          | Đặc điểm                     |
| :---------: | ------------------------------- | ---------------------------- |
|  **LoRA**   | ΔW = BA (Low-Rank Adaptation)   | Baseline, giảm tài nguyên    |
|  **DoRA**   | Phân tách Direction & Magnitude | ICML 2024, học ổn định hướng |
|  **PiSSA**  | Khởi tạo SVD-based              | Tăng tốc hội tụ đầu training |

**Công cụ chính:** [Unsloth](https://github.com/unslothai/unsloth) + HuggingFace + TRL + Quantization 4-bit

## 📁 Cấu trúc dự án

```
CS431-DoRA-vs-PiSSA-LegalSLM/
│
├── configs/                     # Cấu hình thí nghiệm
│   ├── base_config.yaml         # Tham số chung
│   ├── lora_config.yaml         # LoRA (Baseline)
│   ├── dora_config.yaml         # DoRA
│   └── pissa_config.yaml        # PiSSA
│
├── data/                        # Dữ liệu
│   ├── raw/                     # Dữ liệu gốc
│   ├── processed/               # Dữ liệu đã xử lý
│   └── prepare_data.py          # Script tiền xử lý
│
├── training/                    # Huấn luyện
│   ├── train.py                 # Script chính
│   └── trainer_utils.py         # Load model, PEFT routing
│
├── evaluation/                  # Đánh giá
│   ├── evaluate.py              # Script đánh giá
│   └── metrics.py               # PPL, ROUGE, BLEU
│
├── utils/                       # Tiện ích
│   ├── logger.py                # Logging + WandB
│   └── helpers.py               # Seed, VRAM, config
│
├── notebooks/                   # Jupyter notebooks
├── outputs/                     # Kết quả (gitignored)
│   ├── checkpoints/             # Model checkpoints
│   ├── logs/                    # Training logs
│   └── results/                 # Evaluation results
│
├── docs/                        # Tài liệu
│   └── research_proposal.md     # Đề cương nghiên cứu
│
├── .gitignore
├── requirements.txt
├── setup.py
└── README.md
```

## 🚀 Hướng dẫn cài đặt

### 1. Clone repository

```bash
git clone https://github.com/TuTTC/CS431-DoRA-vs-PiSSA-LegalSLM.git
cd CS431-DoRA-vs-PiSSA-LegalSLM
```

### 2. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

> **Lưu ý:** Unsloth cần được cài đặt theo hướng dẫn chính thức tại [unsloth.ai](https://docs.unsloth.ai/get-started/installation) tùy theo phiên bản CUDA/PyTorch.

### 3. Chuẩn bị dữ liệu

Đặt dữ liệu gốc (`.json` / `.jsonl`) vào `data/raw/`, sau đó chạy:

```bash
python data/prepare_data.py --input_dir data/raw --output_dir data/processed
```

## 🏋️ Huấn luyện

Chạy từng nhánh thí nghiệm:

```bash
# LoRA (Baseline)
python training/train.py --peft_config configs/lora_config.yaml

# DoRA
python training/train.py --peft_config configs/dora_config.yaml

# PiSSA
python training/train.py --peft_config configs/pissa_config.yaml
```

## 📊 Đánh giá

```bash
# Đánh giá LoRA
python evaluation/evaluate.py --peft_config configs/lora_config.yaml

# Đánh giá DoRA
python evaluation/evaluate.py --peft_config configs/dora_config.yaml

# Đánh giá PiSSA
python evaluation/evaluate.py --peft_config configs/pissa_config.yaml
```

**Metrics đánh giá:**
- **Perplexity (PPL):** Đánh giá chất lượng ngôn ngữ
- **ROUGE-1/2/L:** Đánh giá độ trùng khớp n-gram
- **BLEU:** Đánh giá chất lượng văn bản sinh ra

## 📅 Timeline (8 tuần)

| Tuần  | Nội dung                                                |
| :---: | ------------------------------------------------------- |
|  1–2  | Thu thập dữ liệu + Setup môi trường + Literature Review |
|  3–4  | Chạy LoRA baseline + Hyperparameter tuning              |
|  5–6  | Chạy DoRA & PiSSA + WandB logging                       |
|   7   | Evaluation + Metrics + Biểu đồ so sánh                  |
|   8   | Báo cáo + Ablation Study                                |

---

<p align="center">
  <b>CS431 - Các kỹ thuật Học Sâu và Ứng dụng</b><br>
  Trường Đại học Công nghệ Thông tin - ĐHQG TP.HCM
</p>
