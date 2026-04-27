"""
Legal Document Chunker
=======================
Chia văn bản pháp lý thành chunks, chiến lược khác nhau cho 2 loại corpus:
  - Văn bản luật gốc (legal): Tách theo cấu trúc Điều/Khoản/Điểm
  - Bài báo pháp luật (news): Fixed-window với overlap

Chunking schema phân cấp theo:
  Document → Chapter → Section → Article → Clause → Point
  (Tham chiếu: Vietnamese Legal KG for RAG paper)

Usage:
    chunker = LegalChunker(chunk_size=512, chunk_overlap=64)
    chunks = chunker.chunk(texts, source_type="legal")
"""

import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class Chunk:
    """Một đoạn văn bản đã được chia nhỏ."""
    text: str
    chunk_id: str
    source_type: str            # "legal" | "news"
    source_document: str = ""
    article_id: str = ""        # e.g. "Điều 584"
    parent_chapter: str = ""    # e.g. "Chương XX"
    start_pos: int = 0
    end_pos: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "chunk_id": self.chunk_id,
            "source_type": self.source_type,
            "source_document": self.source_document,
            "article_id": self.article_id,
            "parent_chapter": self.parent_chapter,
            "start_pos": self.start_pos,
            "end_pos": self.end_pos,
            **self.metadata,
        }


class LegalChunker:
    """
    Chunker cho văn bản pháp lý Việt Nam.
    
    Chiến lược:
      - "legal": Tách theo cấu trúc Điều (Article) → giữ nguyên Khoản/Điểm
      - "news": Fixed-window chunking với overlap
    """

    # Regex patterns cho cấu trúc pháp lý
    CHAPTER_RE = re.compile(
        r"(Chương\s+[IVXLCDM\d]+[.:]*\s*[^\n]*)", re.IGNORECASE
    )
    ARTICLE_RE = re.compile(
        r"(Điều\s+\d+[a-z]?\.?\s*[^\n]*)", re.IGNORECASE
    )
    SECTION_RE = re.compile(
        r"(Mục\s+\d+\.?\s*[^\n]*)", re.IGNORECASE
    )

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        min_chunk_length: int = 50,
    ):
        """
        Args:
            chunk_size: Số ký tự tối đa mỗi chunk (cho news chunking)
            chunk_overlap: Số ký tự overlap giữa các chunk (cho news chunking)
            min_chunk_length: Bỏ qua chunk ngắn hơn giá trị này
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_length = min_chunk_length

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def chunk(
        self,
        samples: List[Dict[str, Any]],
        source_type: str = "legal",
        text_field: str = "clean_text",
    ) -> List[Chunk]:
        """
        Chia danh sách samples thành chunks.

        Args:
            samples: List[Dict] — output từ LegalPreprocessor.process()
            source_type: "legal" hoặc "news"
            text_field: Tên trường chứa văn bản đã tiền xử lý

        Returns:
            List[Chunk]
        """
        all_chunks = []
        for i, sample in enumerate(samples):
            text = sample.get(text_field, "")
            if not text or len(text) < self.min_chunk_length:
                continue

            doc_id = sample.get("id", f"{source_type}_{i}")

            if source_type == "legal":
                chunks = self._chunk_by_article(text, doc_id)
            else:
                chunks = self._chunk_with_overlap(text, doc_id, source_type="news")

            all_chunks.extend(chunks)

            if (i + 1) % 10000 == 0:
                print(f"[CHUNKER] Processed {i + 1}/{len(samples)} docs → {len(all_chunks)} chunks")

        print(f"[CHUNKER] {source_type}: {len(samples)} docs → {len(all_chunks)} chunks")
        return all_chunks

    # ------------------------------------------------------------------
    # Strategy 1: Legal Structure Chunking (cho văn bản luật gốc)
    # ------------------------------------------------------------------
    def _chunk_by_article(self, text: str, doc_id: str) -> List[Chunk]:
        """
        Tách văn bản luật theo cấu trúc Điều (Article).
        
        Mỗi Điều trở thành 1 chunk, giữ nguyên Khoản/Điểm bên trong.
        Nếu Điều quá dài (> 2 * chunk_size), chia thêm theo Khoản.
        """
        chunks = []

        # Tìm tất cả vị trí Chương
        chapters = list(self.CHAPTER_RE.finditer(text))
        current_chapter = ""

        # Tìm tất cả vị trí Điều
        articles = list(self.ARTICLE_RE.finditer(text))

        if not articles:
            # Không tìm thấy cấu trúc Điều → fallback fixed-window
            return self._chunk_with_overlap(text, doc_id, source_type="legal")

        for idx, match in enumerate(articles):
            start = match.start()
            end = articles[idx + 1].start() if idx + 1 < len(articles) else len(text)
            article_text = text[start:end].strip()
            article_title = match.group(0).strip()[:100]  # Truncate to avoid MemoryError

            # Xác định Chương chứa Điều này
            for ch in chapters:
                if ch.start() <= start:
                    current_chapter = ch.group(0).strip()[:100]

            if not article_text or len(article_text) < self.min_chunk_length:
                continue

            # Nếu Điều quá dài → chia theo Khoản
            if len(article_text) > self.chunk_size * 2:
                sub_chunks = self._split_long_article(
                    article_text, doc_id, article_title, current_chapter, start
                )
                chunks.extend(sub_chunks)
            else:
                chunk = Chunk(
                    text=article_text,
                    chunk_id=f"{doc_id}_art{idx}",
                    source_type="legal",
                    source_document=doc_id,
                    article_id=article_title,
                    parent_chapter=current_chapter,
                    start_pos=start,
                    end_pos=end,
                )
                chunks.append(chunk)

        return chunks

    def _split_long_article(
        self,
        article_text: str,
        doc_id: str,
        article_title: str,
        chapter: str,
        global_start: int,
    ) -> List[Chunk]:
        """Chia Điều dài thành sub-chunks theo Khoản."""
        khoan_re = re.compile(r"(\d+\.\s)", re.MULTILINE)
        parts = khoan_re.split(article_text)

        chunks = []
        current_text = ""
        sub_idx = 0

        for part in parts:
            if len(current_text) + len(part) > self.chunk_size and current_text:
                chunks.append(Chunk(
                    text=current_text.strip(),
                    chunk_id=f"{doc_id}_sub{sub_idx}",
                    source_type="legal",
                    source_document=doc_id,
                    article_id=article_title,
                    parent_chapter=chapter,
                    start_pos=global_start,
                    end_pos=global_start + len(current_text),
                ))
                sub_idx += 1
                current_text = ""
            current_text += part

        if current_text.strip() and len(current_text.strip()) >= self.min_chunk_length:
            chunks.append(Chunk(
                text=current_text.strip(),
                chunk_id=f"{doc_id}_{article_title}_sub{sub_idx}",
                source_type="legal",
                source_document=doc_id,
                article_id=article_title,
                parent_chapter=chapter,
                start_pos=global_start,
                end_pos=global_start + len(current_text),
            ))

        return chunks

    # ------------------------------------------------------------------
    # Strategy 2: Fixed-Window Chunking (cho bài báo pháp luật)
    # ------------------------------------------------------------------
    def _chunk_with_overlap(
        self,
        text: str,
        doc_id: str,
        source_type: str = "news",
    ) -> List[Chunk]:
        """
        Chia văn bản bằng sliding window.
        
        Window size = chunk_size, overlap = chunk_overlap.
        Cố gắng cắt ở ranh giới câu (dấu . ! ? ;).
        """
        chunks = []
        start = 0
        idx = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))

            # Cố gắng cắt ở ranh giới câu
            if end < len(text):
                boundary = self._find_sentence_boundary(text, start, end)
                if boundary > start:
                    end = boundary

            chunk_text = text[start:end].strip()

            if chunk_text and len(chunk_text) >= self.min_chunk_length:
                chunks.append(Chunk(
                    text=chunk_text,
                    chunk_id=f"{doc_id}_chunk{idx}",
                    source_type=source_type,
                    source_document=doc_id,
                    start_pos=start,
                    end_pos=end,
                ))
                idx += 1

            # Move window with overlap — guarantee forward progress
            next_start = end - self.chunk_overlap
            if next_start <= start:
                next_start = start + 1  # force at least 1 char advance
            start = next_start
            if end >= len(text):
                break

        return chunks

    def _find_sentence_boundary(self, text: str, start: int, end: int) -> int:
        """Tìm ranh giới câu gần nhất trước vị trí end."""
        # Tìm dấu câu cuối cùng trong khoảng [start, end]
        search_text = text[start:end]
        best = -1
        best_boundary = end
        for punct in [". ", ".\n", "! ", "? ", "; "]:
            last_pos = search_text.rfind(punct)
            if last_pos > best:
                best = last_pos
                best_boundary = start + last_pos + len(punct)
        return best_boundary if best > 0 else end
