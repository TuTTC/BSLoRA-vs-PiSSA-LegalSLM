"""
HiRAG Pipeline — End-to-end RAG + PiSSA Inference
===================================================
Pipeline kết nối toàn bộ:
  1. HiRetriever truy xuất 3-level context
  2. Prompt assembly theo ChatML format
  3. PiSSA fine-tuned model sinh câu trả lời

Usage:
    pipeline = HiRAGPipeline.from_config("configs/rag_config.yaml")
    answer = pipeline.answer("Trách nhiệm bồi thường thiệt hại?")
"""

import os
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional

from .knowledge_graph import HiIndex
from .retriever import HiRetriever


# =========================================================================
# RAG-enhanced System Prompts (bổ sung context section)
# =========================================================================
RAG_SYSTEM_PROMPTS = {
    "task1": (
        "Bạn là một chuyên gia pháp luật Việt Nam. "
        "Nhiệm vụ của bạn là xác định xem một điều luật "
        "có thể được sử dụng để trả lời câu hỏi pháp lý cụ thể hay không. "
        "Sử dụng các nguồn tham chiếu được cung cấp bên dưới để hỗ trợ phân tích."
    ),
    "task2": (
        "Bạn là một chuyên gia pháp luật Việt Nam. "
        "Hãy trả lời câu hỏi trắc nghiệm sau dựa trên "
        "văn bản pháp luật được cung cấp và nguồn tham chiếu bổ sung."
    ),
    "task3": (
        "Bạn là một chuyên gia pháp luật Việt Nam. "
        "Hãy trả lời câu hỏi mở sau theo cấu trúc lập luận pháp lý chuyên sâu. "
        "Tham chiếu và trích dẫn chính xác các điều luật liên quan từ nguồn được cung cấp."
    ),
}


class HiRAGPipeline:
    """
    Pipeline end-to-end: Query → HiRetrieval → Prompt → PiSSA Model → Answer.

    Hỗ trợ 2 chế độ:
      - RAG mode: Có truy xuất context từ Hierarchical KG
      - Direct mode: Inference trực tiếp không RAG (dùng để so sánh)
    """

    def __init__(
        self,
        retriever: Optional[HiRetriever] = None,
        model: Any = None,
        tokenizer: Any = None,
        max_new_tokens: int = 1024,
        temperature: float = 0.1,
    ):
        self.retriever = retriever
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    @classmethod
    def from_config(cls, config_path: str) -> "HiRAGPipeline":
        """
        Load pipeline từ file config YAML.

        Args:
            config_path: Đường dẫn tới configs/rag_config.yaml
        """
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        rag_config = config.get("rag", {})
        paths_config = config.get("paths", {})

        # Step 1: Load embedding model
        embedding_model_name = rag_config.get(
            "embedding_model", "bkai-foundation-models/vietnamese-bi-encoder"
        )
        print(f"[Pipeline] Loading embedding model: {embedding_model_name}")
        try:
            from sentence_transformers import SentenceTransformer
            embedding_model = SentenceTransformer(embedding_model_name)
        except Exception as e:
            print(f"[Pipeline] Failed to load embedding model: {e}")
            embedding_model = None

        # Step 2: Load Knowledge Graph
        kg_path = paths_config.get("kg_cache", "data/rag_cache/knowledge_graph.pkl")
        if os.path.exists(kg_path):
            print(f"[Pipeline] Loading KG from: {kg_path}")
            kg = HiIndex.load(kg_path)
        else:
            print(f"[Pipeline] KG not found at {kg_path}. Run build_knowledge_base.py first.")
            kg = None

        # Step 3: Create Retriever
        if kg and embedding_model:
            retriever = HiRetriever(
                knowledge_graph=kg,
                embedding_model=embedding_model,
                top_n=rag_config.get("top_n_entities", 20),
                top_m=rag_config.get("top_m_community_keys", 5),
            )
        else:
            retriever = None

        # Step 4: Load PiSSA model + adapter
        model_path = rag_config.get("model_path", "")
        adapter_path = rag_config.get("adapter_path", "")
        model, tokenizer = cls._load_model(model_path, adapter_path)

        return cls(
            retriever=retriever,
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=rag_config.get("max_new_tokens", 1024),
            temperature=rag_config.get("temperature", 0.1),
        )

    @staticmethod
    def _load_model(model_path: str, adapter_path: str):
        """Load model + adapter (PiSSA / LoRA)."""
        if not model_path:
            print("[Pipeline] No model path specified")
            return None, None

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel

            print(f"[Pipeline] Loading base model: {model_path}")
            tokenizer = AutoTokenizer.from_pretrained(model_path)

            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                torch_dtype="auto",
            )

            if adapter_path and os.path.exists(adapter_path):
                print(f"[Pipeline] Loading adapter: {adapter_path}")
                model = PeftModel.from_pretrained(model, adapter_path)

            model.eval()
            print("[Pipeline] Model loaded successfully")
            return model, tokenizer

        except Exception as e:
            print(f"[Pipeline] Model load failed: {e}")
            return None, None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def answer(
        self,
        query: str,
        task_type: str = "task3",
        use_rag: bool = True,
    ) -> Dict[str, Any]:
        """
        Trả lời câu hỏi pháp lý.

        Args:
            query: Câu hỏi
            task_type: "task1", "task2", "task3"
            use_rag: True = dùng RAG, False = inference trực tiếp

        Returns:
            Dict chứa "answer", "context", "retrieval_info"
        """
        retrieval_result = None
        context = ""

        # Step 1: Retrieve context (nếu RAG mode)
        if use_rag and self.retriever is not None:
            retrieval_result = self.retriever.retrieve(query)
            context = retrieval_result["context"]

        # Step 2: Build prompt
        prompt = self._build_prompt(query, task_type, context)

        # Step 3: Generate answer
        if self.model is not None and self.tokenizer is not None:
            answer = self._generate(prompt)
        else:
            answer = "[Model chưa được load. Hãy chạy pipeline.from_config() với model path hợp lệ]"

        # Step 4: Post-process (loại bỏ <think> tags)
        answer = self._postprocess(answer)

        return {
            "answer": answer,
            "context": context,
            "retrieval_info": retrieval_result,
            "task_type": task_type,
            "rag_enabled": use_rag and self.retriever is not None,
        }

    def answer_batch(
        self,
        queries: List[Dict[str, str]],
        use_rag: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Batch inference.

        Args:
            queries: List[{"query": "...", "task_type": "task3"}]
        """
        results = []
        for i, q in enumerate(queries):
            result = self.answer(
                query=q["query"],
                task_type=q.get("task_type", "task3"),
                use_rag=use_rag,
            )
            results.append(result)

            if (i + 1) % 10 == 0:
                print(f"[Pipeline] Processed {i + 1}/{len(queries)} queries")

        return results

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _build_prompt(
        self, query: str, task_type: str, context: str
    ) -> str:
        """Xây dựng prompt theo ChatML format + RAG context."""
        system_prompt = RAG_SYSTEM_PROMPTS.get(task_type, RAG_SYSTEM_PROMPTS["task3"])

        if context:
            user_content = (
                f"[NGUỒN THAM CHIẾU TỪ HỆ THỐNG RAG]\n"
                f"{context}\n\n"
                f"[CÂU HỎI]\n{query}"
            )
        else:
            user_content = query

        # ChatML format (Qwen3)
        prompt = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{user_content}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        return prompt

    def _generate(self, prompt: str) -> str:
        """Gọi model sinh câu trả lời."""
        import torch

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
                top_p=0.95,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only the generated part
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    @staticmethod
    def _postprocess(answer: str) -> str:
        """Loại bỏ <think>...</think> tags nếu có."""
        if "</think>" in answer:
            answer = answer.split("</think>")[-1].strip()
        return answer.strip()
