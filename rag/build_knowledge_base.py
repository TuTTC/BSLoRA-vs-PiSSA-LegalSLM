"""
Build Knowledge Base — Offline Indexing Script
================================================
Xây dựng Hierarchical KG từ 2 bộ corpus:
  1. VLSP2025-LegalSML/legal-pretrain (văn bản luật gốc)
  2. VLSP2025-LegalSML/legal-pretrain-news (bài báo pháp luật)

Quy trình (chạy 1 lần):
  1. Load 2 dataset
  2. Preprocessing: Clean HTML → Normalize → Sentence segmentation
  3. Chunking: Luật → theo Điều/Khoản; News → fixed window
  4. Entity + Relation extraction (Gemini API hoặc regex fallback)
  5. Build Hierarchical KG (Layer 0 → GMM → Layer 1+)
  6. Community Detection + Reports
  7. Cache kết quả

Usage:
    python rag/build_knowledge_base.py --config configs/rag_config.yaml
    python rag/build_knowledge_base.py --config configs/rag_config.yaml --max_samples 100
"""

import os
import sys
import yaml
import argparse
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.preprocessor import LegalPreprocessor
from rag.chunker import LegalChunker
from rag.knowledge_graph import HiIndex


def load_datasets(config: dict, max_samples: int = None):
    """Load 2 bộ corpus từ HuggingFace."""
    from datasets import load_dataset

    kb_config = config.get("rag", {}).get("knowledge_bases", {})

    all_samples = {"legal": [], "news": []}

    # 1. Load legal corpus
    legal_name = kb_config.get("legal_corpus", "VLSP2025-LegalSML/legal-pretrain")
    print(f"\n[BUILD] Loading legal corpus: {legal_name}")
    try:
        ds_legal = load_dataset(legal_name, split="train")
        legal_samples = [dict(s) for s in ds_legal]
        if max_samples:
            legal_samples = legal_samples[:max_samples]
        all_samples["legal"] = legal_samples
        print(f"[BUILD] Legal corpus: {len(legal_samples)} samples")
    except Exception as e:
        print(f"[BUILD] Failed to load legal corpus: {e}")

    # 2. Load legal news
    news_name = kb_config.get("legal_news", "VLSP2025-LegalSML/legal-pretrain-news")
    print(f"\n[BUILD] Loading legal news: {news_name}")
    try:
        ds_news = load_dataset(news_name, split="train")
        news_samples = [dict(s) for s in ds_news]
        if max_samples:
            news_samples = news_samples[:max_samples]
        all_samples["news"] = news_samples
        print(f"[BUILD] Legal news: {len(news_samples)} samples")
    except Exception as e:
        print(f"[BUILD] Failed to load legal news: {e}")

    return all_samples


def setup_llm_client(config: dict):
    """Tạo LLM client cho entity extraction."""
    rag_config = config.get("rag", {})
    llm_name = rag_config.get("entity_extraction_model", "gemini")

    if llm_name == "gemini":
        try:
            import google.generativeai as genai

            api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                print("[BUILD] No Gemini API key found. Using regex fallback for entity extraction.")
                return None

            genai.configure(api_key=api_key)

            class GeminiClient:
                def __init__(self):
                    self.model = genai.GenerativeModel("gemini-2.0-flash")
                    self.call_count = 0

                def generate(self, prompt: str) -> str:
                    self.call_count += 1
                    # Rate limiting: 15 RPM for free tier
                    if self.call_count % 14 == 0:
                        print(f"[LLM] Rate limit pause (call #{self.call_count})...")
                        time.sleep(60)
                    
                    response = self.model.generate_content(prompt)
                    return response.text

            return GeminiClient()

        except ImportError:
            print("[BUILD] google-generativeai not installed. Using regex fallback.")
            return None
    else:
        print(f"[BUILD] Unsupported LLM: {llm_name}. Using regex fallback.")
        return None


def setup_embedding_model(config: dict):
    """Tạo embedding model."""
    rag_config = config.get("rag", {})
    model_name = rag_config.get(
        "embedding_model", "bkai-foundation-models/vietnamese-bi-encoder"
    )

    try:
        from sentence_transformers import SentenceTransformer
        print(f"[BUILD] Loading embedding model: {model_name}")
        model = SentenceTransformer(model_name)
        return model
    except Exception as e:
        print(f"[BUILD] Failed to load embedding model: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Build Hierarchical Knowledge Base")
    parser.add_argument(
        "--config", type=str, default="configs/rag_config.yaml",
        help="Path to RAG config YAML"
    )
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Giới hạn số samples mỗi dataset (dùng để test nhanh)"
    )
    parser.add_argument(
        "--skip_llm", action="store_true",
        help="Bỏ qua LLM, chỉ dùng regex cho entity extraction"
    )
    parser.add_argument(
        "--just_chunking", action="store_true",
        help="Chỉ load data, preprocess, và chunking rồi lưu ra file, bỏ qua bước graph/models"
    )
    args = parser.parse_args()

    # Load config
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    rag_config = config.get("rag", {})
    paths_config = config.get("paths", {})

    start_time = time.time()
    print("=" * 60)
    print("  Building Hierarchical Knowledge Base")
    print("=" * 60)

    # Fast Resume Check
    cache_dir = os.path.dirname(paths_config.get("kg_cache", "data/rag_cache/knowledge_graph.pkl"))
    checkpoint_path = Path(cache_dir) / "kg_layer0_checkpoint.pkl"
    all_chunks = []
    datasets = {"legal": [], "news": []}
    
    if checkpoint_path.exists():
        print(f"\n[BUILD] Detected checkpoint: {checkpoint_path}")
        print("[BUILD] Skipping Step 1-3 (Load/Preprocess/Chunk) and resuming from checkpoint...")
    else:
        # Step 1: Load datasets
        print("\n" + "=" * 60)
        print("  Step 1: Loading Datasets")
        print("=" * 60)
        datasets = load_datasets(config, max_samples=args.max_samples)

        total_samples = sum(len(v) for v in datasets.values())
        if total_samples == 0:
            print("[BUILD] Không có dữ liệu. Kiểm tra lại config.")
            return

        # Step 2: Preprocessing
        print("\n" + "=" * 60)
        print("  Step 2: Preprocessing")
        print("=" * 60)
        preprocessor = LegalPreprocessor()

        processed = {}
        for dtype, samples in datasets.items():
            if samples:
                # legal-pretrain dùng cột 'doc_content', legal-news dùng cột 'text'
                text_field = "doc_content" if dtype == "legal" else "text"
                processed[dtype] = preprocessor.process(dtype, samples, text_field=text_field)

        # Step 3: Chunking
        print("\n" + "=" * 60)
        print("  Step 3: Chunking")
        print("=" * 60)
        chunker = LegalChunker(
            chunk_size=rag_config.get("chunk_size", 512),
            chunk_overlap=rag_config.get("chunk_overlap", 64),
        )

        for dtype, samples in processed.items():
            if samples:
                chunks = chunker.chunk(samples, source_type=dtype)
                all_chunks.extend(chunks)

        print(f"[BUILD] Total chunks: {len(all_chunks)}")

    if args.just_chunking and not checkpoint_path.exists():
        print("\n" + "=" * 60)
        print("  Saving Chunks Only & Exiting (--just_chunking)")
        print("=" * 60)
        import json
        out_path = os.path.join(os.path.dirname(paths_config.get("kg_cache", "data/rag_cache/knowledge_graph.pkl")), "chunks.json")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        # Convert objects to dict
        chunks_data = [c.to_dict() for c in all_chunks]
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(chunks_data, f, ensure_ascii=False, indent=2)
        print(f"[BUILD] Saved {len(all_chunks)} chunks to {out_path}")
        print("=" * 60)
        return

    # Step 4: Setup models
    print("\n" + "=" * 60)
    print("  Step 4: Setup Models")
    print("=" * 60)
    llm_client = None if args.skip_llm else setup_llm_client(config)
    embedding_model = setup_embedding_model(config)

    # Step 5: Build Hierarchical KG
    print("\n" + "=" * 60)
    print("  Step 5: Building Hierarchical Knowledge Graph")
    print("=" * 60)
    hi_index = HiIndex(
        llm_client=llm_client,
        embedding_model=embedding_model,
        num_layers=rag_config.get("kg_layers", 2),
        cache_dir=os.path.dirname(paths_config.get("kg_cache", "data/rag_cache/knowledge_graph.pkl")),
    )

    hi_index.build(all_chunks)

    # Step 6: Save
    print("\n" + "=" * 60)
    print("  Step 6: Saving Knowledge Graph")
    print("=" * 60)
    kg_path = paths_config.get("kg_cache", "data/rag_cache/knowledge_graph.pkl")
    hi_index.save(kg_path)

    # Summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("  Build Complete!")
    print("=" * 60)
    print(f"  Total time:       {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Input samples:    {total_samples}")
    print(f"  Total chunks:     {len(all_chunks)}")
    print(f"  Total entities:   {len(hi_index.entities)}")
    print(f"  Total relations:  {len(hi_index.relations)}")
    print(f"  Total communities:{len(hi_index.communities)}")
    print(f"  KG saved to:      {kg_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
