"""
Legal Text Preprocessor
========================
Tiền xử lý văn bản pháp lý theo đề xuất paper top-1 VLSP2025 (Section 3.2):
  1. Text cleaning: loại bỏ HTML tags, artifacts
  2. Normalization: chuẩn hóa Unicode, số điều/khoản
  3. Sentence segmentation: tách câu

Nguồn tham chiếu:
  - VLSP2025-LegalSML paper (Section 3.2: Data Collection & Preprocessing)

Usage:
    preprocessor = LegalPreprocessor()
    clean_texts = preprocessor.process("legal", raw_texts)
"""

import re
import unicodedata
from typing import List, Dict, Any, Optional


class LegalPreprocessor:
    """
    Tiền xử lý văn bản pháp lý Việt Nam.
    
    Hỗ trợ 2 loại corpus:
      - "legal": Văn bản luật gốc (codes, statutes, decrees, circulars)
      - "news": Bài báo & bình luận pháp luật
    """

    # Regex patterns
    HTML_TAG_RE = re.compile(r"<[^>]+>")
    MULTI_SPACE_RE = re.compile(r"[ \t]+")
    MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
    URL_RE = re.compile(r"https?://\S+")
    EMAIL_RE = re.compile(r"\S+@\S+\.\S+")

    # Legal-specific normalization
    DIEU_NORMALIZE_RE = re.compile(
        r"[Đđ]iều\s+(\d+)", re.IGNORECASE
    )
    KHOAN_NORMALIZE_RE = re.compile(
        r"[Kk]hoản\s+(\d+)", re.IGNORECASE
    )
    DIEM_NORMALIZE_RE = re.compile(
        r"[Đđ]iểm\s+([a-zđ])", re.IGNORECASE
    )

    # Sentence boundary patterns (Vietnamese)
    SENTENCE_SPLIT_RE = re.compile(
        r"(?<=[.!?;])\s+(?=[A-ZÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴĐ])"
    )

    def __init__(self, remove_urls: bool = True, remove_emails: bool = True):
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails

    # ------------------------------------------------------------------
    # Step 1: Text Cleaning
    # ------------------------------------------------------------------
    def clean_html(self, text: str) -> str:
        """Loại bỏ HTML tags và các artifacts không liên quan."""
        # Remove HTML tags
        text = self.HTML_TAG_RE.sub("", text)

        # Remove HTML entities
        text = text.replace("&nbsp;", " ")
        text = text.replace("&amp;", "&")
        text = text.replace("&lt;", "<")
        text = text.replace("&gt;", ">")
        text = text.replace("&quot;", '"')
        text = re.sub(r"&#?\w+;", "", text)

        # Remove URLs and emails if configured
        if self.remove_urls:
            text = self.URL_RE.sub("", text)
        if self.remove_emails:
            text = self.EMAIL_RE.sub("", text)

        return text.strip()

    # ------------------------------------------------------------------
    # Step 2: Text Normalization
    # ------------------------------------------------------------------
    def normalize(self, text: str) -> str:
        """
        Chuẩn hóa văn bản:
          - Unicode NFC normalization
          - Chuẩn hóa khoảng trắng
          - Chuẩn hóa format số điều/khoản/điểm
        """
        # Unicode NFC normalization
        text = unicodedata.normalize("NFC", text)

        # Collapse multiple spaces → single space
        text = self.MULTI_SPACE_RE.sub(" ", text)

        # Collapse multiple newlines → max 2
        text = self.MULTI_NEWLINE_RE.sub("\n\n", text)

        # Normalize legal references
        text = self.DIEU_NORMALIZE_RE.sub(r"Điều \1", text)
        text = self.KHOAN_NORMALIZE_RE.sub(r"Khoản \1", text)
        text = self.DIEM_NORMALIZE_RE.sub(r"Điểm \1", text)

        # Strip leading/trailing whitespace per line
        lines = [line.strip() for line in text.split("\n")]
        text = "\n".join(lines)

        return text.strip()

    # ------------------------------------------------------------------
    # Step 3: Sentence Segmentation
    # ------------------------------------------------------------------
    def segment_sentences(self, text: str) -> List[str]:
        """
        Tách câu tiếng Việt.
        
        Ưu tiên dùng underthesea nếu có, fallback sang regex.
        """
        try:
            from underthesea import sent_tokenize
            return sent_tokenize(text)
        except ImportError:
            # Fallback: regex-based segmentation
            sentences = self.SENTENCE_SPLIT_RE.split(text)
            return [s.strip() for s in sentences if s.strip()]

    # ------------------------------------------------------------------
    # Full Pipeline
    # ------------------------------------------------------------------
    def process_text(self, text: str) -> str:
        """Xử lý 1 văn bản: clean → normalize."""
        text = self.clean_html(text)
        text = self.normalize(text)
        return text

    def process(
        self,
        dataset_type: str,
        samples: List[Dict[str, Any]],
        text_field: str = "text",
    ) -> List[Dict[str, Any]]:
        """
        Pipeline tiền xử lý cho danh sách samples.

        Args:
            dataset_type: "legal" hoặc "news"
            samples: List[Dict] — mỗi dict có trường text_field
            text_field: Tên trường chứa văn bản gốc

        Returns:
            List[Dict] với trường "clean_text" và "sentences" được thêm vào
        """
        processed = []
        for i, sample in enumerate(samples):
            raw_text = sample.get(text_field, "")
            if not raw_text:
                continue

            clean_text = self.process_text(raw_text)

            if not clean_text or len(clean_text) < 10:
                continue

            result = {
                **sample,
                "clean_text": clean_text,
                "sentences": self.segment_sentences(clean_text),
                "source_type": dataset_type,
                "original_length": len(raw_text),
                "clean_length": len(clean_text),
            }
            processed.append(result)

            if (i + 1) % 10000 == 0:
                print(f"[PREPROCESS] Processed {i + 1}/{len(samples)} {dataset_type} samples")

        print(f"[PREPROCESS] {dataset_type}: {len(processed)}/{len(samples)} samples retained")
        return processed
