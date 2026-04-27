"""
RAG Module - Hierarchical RAG for Vietnamese Legal QA
=====================================================
Inspired by HiRAG (arxiv 2503.10150), adapted for Vietnamese legal domain.

Modules:
    - preprocessor: Text cleaning, normalization, sentence segmentation
    - chunker: Legal-structure-aware chunking
    - knowledge_graph: Hierarchical KG construction (HiIndex)
    - retriever: 3-level retrieval (HiRetrieval)
    - pipeline: End-to-end RAG + PiSSA inference
"""

from .preprocessor import LegalPreprocessor
from .chunker import LegalChunker
from .knowledge_graph import HiIndex
from .retriever import HiRetriever
from .pipeline import HiRAGPipeline

__all__ = [
    "LegalPreprocessor",
    "LegalChunker",
    "HiIndex",
    "HiRetriever",
    "HiRAGPipeline",
]
