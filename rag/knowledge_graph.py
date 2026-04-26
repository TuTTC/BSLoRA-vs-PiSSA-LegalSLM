"""
Hierarchical Knowledge Graph (HiIndex)
=======================================
Xây dựng Knowledge Graph phân cấp theo framework HiRAG (arxiv 2503.10150),
adapted cho domain pháp lý Việt Nam.

Quy trình HiIndex:
  1. Layer 0 (Basic KG): Entity extraction + Relation extraction từ chunks
  2. Layer 1+ (Summary): GMM clustering → LLM summarization → summary entities
  3. Communities: Leiden algorithm → Community reports

Entity Type Schema (có cơ sở học thuật):
  - LEGAL_DOC:     Văn bản pháp luật      (LKIF:Norm + Vuong:Law)
  - PROVISION:     Điều khoản              (Hierarchical KG schema)
  - LEGAL_SUBJECT: Chủ thể pháp lý        (LKIF:Legal Role + VLSP:PER/ORG)
  - COURT:         Cơ quan tư pháp         (Vuong:Court)
  - LEGAL_ACTION:  Hành vi pháp lý        (LKIF:Legal Action)
  - SANCTION:      Chế tài / Hậu quả      (LKIF:Norm deontic)
  - LEGAL_DOMAIN:  Lĩnh vực pháp lý       (Vuong:Domain)
  - TEMPORAL:      Thời gian / Hiệu lực   (VLSP:TIM + LKIF:Modification)

Nguồn học thuật:
  - HiRAG: arxiv 2503.10150
  - LKIF Core Ontology: Hoekstra et al. 2007 (Estrella Project)
  - VLSP 2021 NER: Linh et al. 2021
  - Vuong et al. 2023: Vietnamese Legal KG with Heterogeneous Graphs

Usage:
    hi_index = HiIndex(config)
    hi_index.build(chunks)
    hi_index.save("data/rag_cache/knowledge_graph.pkl")
"""

import json
import pickle
import hashlib
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field

import numpy as np

try:
    import networkx as nx
except ImportError:
    raise ImportError("networkx is required: pip install networkx>=3.0")

try:
    from sklearn.mixture import GaussianMixture
except ImportError:
    raise ImportError("scikit-learn is required: pip install scikit-learn>=1.3")


# =========================================================================
# Constants
# =========================================================================
ENTITY_TYPES = [
    "LEGAL_DOC",        # Văn bản pháp luật
    "PROVISION",        # Điều khoản
    "LEGAL_SUBJECT",    # Chủ thể pháp lý
    "COURT",            # Cơ quan tư pháp
    "LEGAL_ACTION",     # Hành vi pháp lý
    "SANCTION",         # Chế tài / Hậu quả
    "LEGAL_DOMAIN",     # Lĩnh vực pháp lý
    "TEMPORAL",         # Thời gian / Hiệu lực
]

# Meta summary entities (dùng để guide LLM khi tạo summary cho layer cao hơn)
META_SUMMARIES = {
    "LEGAL_DOC": "Các văn bản quy phạm pháp luật (Luật, Bộ luật, Nghị định, Thông tư, Quyết định)",
    "PROVISION": "Các điều, khoản, điểm cụ thể trong văn bản pháp luật",
    "LEGAL_SUBJECT": "Các chủ thể trong quan hệ pháp luật (cá nhân, tổ chức, cơ quan nhà nước)",
    "COURT": "Các cơ quan trong hệ thống tư pháp (tòa án, viện kiểm sát, cơ quan thi hành án)",
    "LEGAL_ACTION": "Các hành vi pháp lý (vi phạm, khiếu nại, tố cáo, ký kết, chuyển nhượng)",
    "SANCTION": "Các hình thức xử lý, chế tài (phạt tiền, phạt tù, bồi thường, tước quyền)",
    "LEGAL_DOMAIN": "Các lĩnh vực pháp lý (hình sự, dân sự, hành chính, lao động, đất đai)",
    "TEMPORAL": "Thông tin về thời gian, hiệu lực, thời hạn, thời hiệu",
}


@dataclass
class Entity:
    """Một thực thể trong Knowledge Graph."""
    name: str
    entity_type: str
    description: str = ""
    layer: int = 0              # 0 = basic, 1+ = summary
    source_chunks: List[str] = field(default_factory=list)
    embedding: Optional[np.ndarray] = None

    @property
    def entity_id(self) -> str:
        return hashlib.md5(f"{self.name}_{self.entity_type}".encode()).hexdigest()[:12]


@dataclass
class Relation:
    """Một quan hệ giữa 2 entity."""
    source_id: str
    target_id: str
    relation_type: str
    description: str = ""
    weight: float = 1.0


# =========================================================================
# Prompt Templates cho Entity/Relation Extraction
# =========================================================================
ENTITY_EXTRACTION_PROMPT = """Bạn là chuyên gia pháp luật Việt Nam. Trích xuất các thực thể pháp lý từ đoạn văn bản sau.

Các loại thực thể cần trích xuất:
- LEGAL_DOC: Tên văn bản pháp luật (Luật, Nghị định, Thông tư, etc.)
- PROVISION: Điều, Khoản, Điểm cụ thể
- LEGAL_SUBJECT: Chủ thể pháp lý (người, tổ chức, cơ quan)
- COURT: Cơ quan tư pháp (tòa án, viện kiểm sát)
- LEGAL_ACTION: Hành vi pháp lý (vi phạm, khiếu nại, ký kết, etc.)
- SANCTION: Chế tài, hình phạt, mức phạt
- LEGAL_DOMAIN: Lĩnh vực pháp lý
- TEMPORAL: Thời gian, hiệu lực, thời hạn

Văn bản:
\"\"\"
{text}
\"\"\"

Trả về JSON array, mỗi entity có format:
{{"name": "...", "type": "...", "description": "Mô tả ngắn gọn vai trò trong ngữ cảnh"}}

Chỉ trả về JSON array, không giải thích thêm."""


RELATION_EXTRACTION_PROMPT = """Bạn là chuyên gia pháp luật Việt Nam. Dựa trên danh sách thực thể và đoạn văn bản, hãy trích xuất các quan hệ pháp lý giữa chúng.

Thực thể:
{entities_json}

Văn bản:
\"\"\"
{text}
\"\"\"

Trả về JSON array, mỗi relation có format:
{{"source": "tên entity nguồn", "target": "tên entity đích", "relation": "mô tả quan hệ"}}

Chỉ trả về JSON array, không giải thích thêm."""


SUMMARY_PROMPT = """Bạn là chuyên gia pháp luật Việt Nam. Dựa trên nhóm thực thể sau, hãy tạo các thực thể tóm tắt (summary entities) ở cấp độ cao hơn.

Loại tóm tắt gợi ý: {meta_summary}

Danh sách thực thể trong nhóm:
{entities_list}

Tạo 1-3 summary entities đại diện cho nhóm này. 
Trả về JSON array:
{{"name": "tên tóm tắt", "type": "loại entity", "description": "mô tả khái quát nhóm"}}

Chỉ trả về JSON array, không giải thích thêm."""


COMMUNITY_REPORT_PROMPT = """Bạn là chuyên gia pháp luật Việt Nam. Viết báo cáo ngữ nghĩa cho cộng đồng thực thể pháp lý sau.

Các thực thể trong cộng đồng:
{entities_list}

Các quan hệ:
{relations_list}

Viết báo cáo ngắn gọn (3-5 câu) tóm tắt:
1. Chủ đề chính của cộng đồng
2. Các thực thể quan trọng nhất
3. Các mối quan hệ pháp lý chính

Chỉ viết báo cáo, không thêm tiêu đề hay format đặc biệt."""


# =========================================================================
# HiIndex — Main Class
# =========================================================================
class HiIndex:
    """
    Xây dựng Hierarchical Knowledge Graph theo HiRAG framework.
    
    Attributes:
        graph: NetworkX graph chứa KG
        entities: Dict[entity_id → Entity]
        communities: List[Dict] — community reports
    """

    def __init__(
        self,
        llm_client: Any = None,
        embedding_model: Any = None,
        num_layers: int = 2,
        num_gmm_components: int = 10,
        cache_dir: str = "data/rag_cache",
    ):
        """
        Args:
            llm_client: LLM client (Gemini, OpenAI, etc.) — cần có method generate(prompt) → str
            embedding_model: SentenceTransformer model — cần có method encode(texts) → np.ndarray
            num_layers: Số tầng hierarchical KG (paper khuyến nghị 2)
            num_gmm_components: Số components cho GMM clustering
            cache_dir: Thư mục cache
        """
        self.llm_client = llm_client
        self.embedding_model = embedding_model
        self.num_layers = num_layers
        self.num_gmm_components = num_gmm_components
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # KG state
        self.graph = nx.Graph()
        self.entities: Dict[str, Entity] = {}
        self.relations: List[Relation] = []
        self.communities: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def build(self, chunks: List[Any], batch_size: int = 10) -> None:
        """
        Xây dựng toàn bộ Hierarchical KG từ danh sách chunks.

        Args:
            chunks: List[Chunk] — output từ LegalChunker
            batch_size: Số chunks xử lý cùng lúc
        """
        print(f"[HiIndex] Building hierarchical KG from {len(chunks)} chunks...")
        print(f"[HiIndex] Layers: {self.num_layers}, GMM components: {self.num_gmm_components}")

        # Step 1: Layer 0 — Basic KG
        checkpoint_path = self.cache_dir / "kg_layer0_checkpoint.pkl"
        if checkpoint_path.exists():
            print("\n[HiIndex] === Step 1: Loading Layer 0 from Checkpoint ===")
            # Tạm thời load state từ checkpoint
            with open(checkpoint_path, "rb") as f:
                data = pickle.load(f)
            self.graph = data["graph"]
            self.entities = data["entities"]
            self.relations = data["relations"]
            print(f"[HiIndex] Loaded {len(self.entities)} entities from checkpoint")
        else:
            print("\n[HiIndex] === Step 1: Building Layer 0 (Basic KG) ===")
            self._build_layer_0(chunks, batch_size)
            print(f"[HiIndex] Layer 0: {len(self.entities)} entities, {len(self.relations)} relations")
            # Lưu checkpoint ngay để chống OOM
            self.save(str(checkpoint_path))

        # Step 2: Compute embeddings for all entities
        print("\n[HiIndex] === Step 2: Computing entity embeddings ===")
        self._compute_embeddings()

        # Step 3: Layer 1+ — Summary Entities
        for layer_idx in range(1, self.num_layers + 1):
            print(f"\n[HiIndex] === Step 3.{layer_idx}: Building Layer {layer_idx} (Summary) ===")
            self._build_summary_layer(layer_idx)

        # Step 4: Community Detection
        print("\n[HiIndex] === Step 4: Community Detection (Leiden) ===")
        self._detect_communities()

        # Step 5: Community Reports
        print("\n[HiIndex] === Step 5: Generating Community Reports ===")
        self._generate_community_reports()

        print(f"\n[HiIndex] Build complete!")
        print(f"[HiIndex] Total entities: {len(self.entities)}")
        print(f"[HiIndex] Total relations: {len(self.relations)}")
        print(f"[HiIndex] Total communities: {len(self.communities)}")

    def save(self, path: Optional[str] = None) -> str:
        """Lưu KG ra file pickle."""
        if path is None:
            path = str(self.cache_dir / "knowledge_graph.pkl")

        data = {
            "graph": self.graph,
            "entities": self.entities,
            "relations": self.relations,
            "communities": self.communities,
            "num_layers": self.num_layers,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

        print(f"[HiIndex] Saved KG to {path}")
        return path

    @classmethod
    def load(cls, path: str) -> "HiIndex":
        """Load KG từ file pickle."""
        with open(path, "rb") as f:
            data = pickle.load(f)

        instance = cls(num_layers=data["num_layers"])
        instance.graph = data["graph"]
        instance.entities = data["entities"]
        instance.relations = data["relations"]
        instance.communities = data["communities"]

        print(f"[HiIndex] Loaded KG: {len(instance.entities)} entities, "
              f"{len(instance.relations)} relations, {len(instance.communities)} communities")
        return instance

    # ------------------------------------------------------------------
    # Step 1: Layer 0 — Basic KG (Entity + Relation Extraction)
    # ------------------------------------------------------------------
    def _build_layer_0(self, chunks: List[Any], batch_size: int) -> None:
        """Trích xuất entity và relation từ mỗi chunk."""
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]

            for chunk in batch:
                chunk_text = chunk.text if hasattr(chunk, "text") else chunk.get("text", "")
                chunk_id = chunk.chunk_id if hasattr(chunk, "chunk_id") else chunk.get("chunk_id", "")

                # Extract entities
                entities = self._extract_entities(chunk_text)
                for ent in entities:
                    ent.source_chunks.append(chunk_id)
                    if ent.entity_id not in self.entities:
                        self.entities[ent.entity_id] = ent
                        self.graph.add_node(
                            ent.entity_id,
                            name=ent.name,
                            entity_type=ent.entity_type,
                            description=ent.description,
                            layer=0,
                        )

                # Extract relations
                if entities:
                    relations = self._extract_relations(chunk_text, entities)
                    for rel in relations:
                        self.relations.append(rel)
                        self.graph.add_edge(
                            rel.source_id,
                            rel.target_id,
                            relation_type=rel.relation_type,
                            description=rel.description,
                        )

            processed = min(i + batch_size, len(chunks))
            if processed % 100 == 0 or processed == len(chunks):
                print(f"[HiIndex] Layer 0 progress: {processed}/{len(chunks)} chunks")

    def _extract_entities(self, text: str) -> List[Entity]:
        """Gọi LLM để trích xuất entity từ text."""
        if self.llm_client is None:
            return self._extract_entities_regex(text)

        prompt = ENTITY_EXTRACTION_PROMPT.format(text=text[:2000])

        try:
            response = self.llm_client.generate(prompt)
            entities_data = self._parse_json_response(response)

            entities = []
            for item in entities_data:
                ent_type = item.get("type", "LEGAL_DOC")
                if ent_type not in ENTITY_TYPES:
                    ent_type = "LEGAL_DOC"

                entities.append(Entity(
                    name=item.get("name", ""),
                    entity_type=ent_type,
                    description=item.get("description", ""),
                    layer=0,
                ))
            return entities

        except Exception as e:
            print(f"[HiIndex] Entity extraction error: {e}")
            return self._extract_entities_regex(text)

    def _extract_entities_regex(self, text: str) -> List[Entity]:
        """Fallback: trích xuất entity bằng regex (không cần LLM)."""
        import re
        entities = []

        # LEGAL_DOC: Luật, Nghị định, Thông tư, etc.
        for match in re.finditer(
            r"(Luật|Bộ luật|Nghị định|Thông tư|Quyết định|Nghị quyết)\s+[^\n,.;]{5,80}",
            text, re.IGNORECASE
        ):
            entities.append(Entity(
                name=match.group(0).strip(),
                entity_type="LEGAL_DOC",
                description="Văn bản pháp luật",
                layer=0,
            ))

        # PROVISION: Điều X
        for match in re.finditer(r"Điều\s+\d+[a-z]?", text, re.IGNORECASE):
            entities.append(Entity(
                name=match.group(0).strip(),
                entity_type="PROVISION",
                description="Điều khoản",
                layer=0,
            ))

        # TEMPORAL: Dates
        for match in re.finditer(
            r"ngày\s+\d{1,2}[/\-]\d{1,2}[/\-]\d{4}|"
            r"năm\s+\d{4}|"
            r"có hiệu lực\s+[^\n,.;]{5,50}",
            text, re.IGNORECASE
        ):
            entities.append(Entity(
                name=match.group(0).strip(),
                entity_type="TEMPORAL",
                description="Thời gian/hiệu lực",
                layer=0,
            ))

        return entities

    def _extract_relations(self, text: str, entities: List[Entity]) -> List[Relation]:
        """Gọi LLM để trích xuất relations giữa các entity."""
        if self.llm_client is None or len(entities) < 2:
            return self._extract_relations_cooccurrence(entities)

        entities_json = json.dumps(
            [{"name": e.name, "type": e.entity_type} for e in entities],
            ensure_ascii=False
        )
        prompt = RELATION_EXTRACTION_PROMPT.format(
            entities_json=entities_json,
            text=text[:2000]
        )

        try:
            response = self.llm_client.generate(prompt)
            relations_data = self._parse_json_response(response)

            # Map entity names to IDs
            name_to_id = {e.name: e.entity_id for e in entities}

            relations = []
            for item in relations_data:
                src_name = item.get("source", "")
                tgt_name = item.get("target", "")
                if src_name in name_to_id and tgt_name in name_to_id:
                    relations.append(Relation(
                        source_id=name_to_id[src_name],
                        target_id=name_to_id[tgt_name],
                        relation_type=item.get("relation", "related_to"),
                        description=item.get("relation", ""),
                    ))
            return relations

        except Exception as e:
            print(f"[HiIndex] Relation extraction error: {e}")
            return self._extract_relations_cooccurrence(entities)

    def _extract_relations_cooccurrence(self, entities: List[Entity]) -> List[Relation]:
        """Fallback: Tạo relation dựa trên co-occurrence trong cùng chunk."""
        relations = []
        for i in range(len(entities)):
            for j in range(i + 1, min(i + 5, len(entities))):
                relations.append(Relation(
                    source_id=entities[i].entity_id,
                    target_id=entities[j].entity_id,
                    relation_type="co_occurs_with",
                    description=f"Xuất hiện cùng trong văn bản",
                ))
        return relations

    # ------------------------------------------------------------------
    # Step 2: Compute Embeddings
    # ------------------------------------------------------------------
    def _compute_embeddings(self) -> None:
        """Tính embedding cho tất cả entity."""
        if self.embedding_model is None:
            print("[HiIndex] No embedding model → skipping embeddings")
            return

        entity_texts = []
        entity_ids = []
        for eid, ent in self.entities.items():
            entity_texts.append(f"{ent.name}: {ent.description}")
            entity_ids.append(eid)

        if entity_texts:
            print(f"[HiIndex] Computing embeddings for {len(entity_texts)} entities...")
            
            # Check if final embeddings file already exists (from a previous successful run)
            final_emb_path = self.cache_dir / "embeddings.npy"
            if final_emb_path.exists():
                print(f"[HiIndex] Loading pre-computed embeddings from {final_emb_path}...")
                embeddings = np.load(str(final_emb_path))
                if len(embeddings) == len(entity_texts):
                    print(f"[HiIndex] Loaded {len(embeddings)} embeddings from cache")
                    for eid, emb in zip(entity_ids, embeddings):
                        self.entities[eid].embedding = emb
                        self.graph.nodes[eid]["embedding"] = emb
                    print(f"[HiIndex] Embeddings restored successfully")
                    return
                else:
                    print(f"[HiIndex] Cached embeddings size mismatch ({len(embeddings)} vs {len(entity_texts)}), recomputing...")

            chunk_size = 50000
            total_chunks = (len(entity_texts) + chunk_size - 1) // chunk_size
            
            all_embeddings = []
            
            for i in range(0, len(entity_texts), chunk_size):
                chunk_idx = i // chunk_size
                chunk_path = self.cache_dir / f"embeddings_chunk_{chunk_idx}.npy"
                
                if chunk_path.exists():
                    print(f"[HiIndex] Loading embedding chunk {chunk_idx + 1}/{total_chunks} from cache...")
                    chunk_embs = np.load(str(chunk_path))
                else:
                    print(f"[HiIndex] Processing embedding chunk {chunk_idx + 1}/{total_chunks}...")
                    chunk_texts = entity_texts[i:i + chunk_size]
                    chunk_embs = self.embedding_model.encode(
                        chunk_texts,
                        show_progress_bar=True,
                        batch_size=128,
                        convert_to_numpy=True
                    )
                    np.save(str(chunk_path), chunk_embs)
                    
                    # Cleanup memory
                    import gc
                    del chunk_texts
                    gc.collect()

                all_embeddings.append(chunk_embs)

            embeddings = np.vstack(all_embeddings)

            for eid, emb in zip(entity_ids, embeddings):
                self.entities[eid].embedding = emb
                self.graph.nodes[eid]["embedding"] = emb

            # Save final embeddings cache and delete chunks
            np.save(str(self.cache_dir / "embeddings.npy"), embeddings)
            for chunk_idx in range(total_chunks):
                chunk_path = self.cache_dir / f"embeddings_chunk_{chunk_idx}.npy"
                if chunk_path.exists():
                    chunk_path.unlink()
                    
            print(f"[HiIndex] Embeddings computed and cached")

    # ------------------------------------------------------------------
    # Step 3: Summary Layers (GMM Clustering → LLM Summarization)
    # ------------------------------------------------------------------
    def _build_summary_layer(self, layer_idx: int) -> None:
        """Xây dựng summary layer bằng GMM clustering + LLM summarization."""
        # Lấy entities từ layer trước
        prev_layer_entities = [
            e for e in self.entities.values() if e.layer == layer_idx - 1
        ]

        if len(prev_layer_entities) < 3:
            print(f"[HiIndex] Layer {layer_idx}: too few entities ({len(prev_layer_entities)}), skipping")
            return

        # Lấy embeddings
        embeddings = []
        entity_ids = []
        for ent in prev_layer_entities:
            if ent.embedding is not None:
                embeddings.append(ent.embedding)
                entity_ids.append(ent.entity_id)

        if len(embeddings) < 3:
            print(f"[HiIndex] Layer {layer_idx}: too few embeddings, skipping")
            return

        embeddings_array = np.array(embeddings)

        # GMM Clustering (theo paper HiRAG, Section 4.1)
        n_components = min(self.num_gmm_components, len(embeddings) // 2)
        n_components = max(2, n_components)

        if len(embeddings_array) > 5000:
            print(f"[HiIndex] Layer {layer_idx}: array too large ({len(embeddings_array)}). Using MiniBatchKMeans with {n_components} components...")
            from sklearn.cluster import MiniBatchKMeans
            kmeans = MiniBatchKMeans(
                n_clusters=n_components,
                random_state=3407,
                batch_size=1024,
                n_init="auto"
            )
            cluster_labels = kmeans.fit_predict(embeddings_array)
        else:
            print(f"[HiIndex] Layer {layer_idx}: GMM clustering with {n_components} components...")
            gmm = GaussianMixture(
                n_components=n_components,
                covariance_type="full",
                random_state=3407,
            )
            cluster_labels = gmm.fit_predict(embeddings_array)

        # Tạo summary entities cho mỗi cluster
        clusters: Dict[int, List[str]] = {}
        for eid, label in zip(entity_ids, cluster_labels):
            clusters.setdefault(label, []).append(eid)

        for cluster_id, member_ids in clusters.items():
            if len(member_ids) < 2:
                continue

            members = [self.entities[eid] for eid in member_ids]
            summary_entities = self._create_summary_entities(members, layer_idx)

            for summary in summary_entities:
                self.entities[summary.entity_id] = summary
                self.graph.add_node(
                    summary.entity_id,
                    name=summary.name,
                    entity_type=summary.entity_type,
                    description=summary.description,
                    layer=layer_idx,
                )
                # Connect summary to cluster members
                for member_id in member_ids:
                    self.graph.add_edge(
                        summary.entity_id,
                        member_id,
                        relation_type="summarizes",
                    )
                    self.relations.append(Relation(
                        source_id=summary.entity_id,
                        target_id=member_id,
                        relation_type="summarizes",
                    ))

        summary_count = sum(1 for e in self.entities.values() if e.layer == layer_idx)
        print(f"[HiIndex] Layer {layer_idx}: created {summary_count} summary entities")

    def _create_summary_entities(
        self, members: List[Entity], layer: int
    ) -> List[Entity]:
        """Tạo summary entities cho 1 cluster, dùng LLM hoặc heuristic."""
        if self.llm_client is None:
            return self._create_summary_heuristic(members, layer)

        # Determine dominant entity type
        type_counts: Dict[str, int] = {}
        for m in members:
            type_counts[m.entity_type] = type_counts.get(m.entity_type, 0) + 1
        dominant_type = max(type_counts, key=type_counts.get)
        meta = META_SUMMARIES.get(dominant_type, "Khái niệm pháp lý")

        entities_list = "\n".join(
            f"- {m.name} ({m.entity_type}): {m.description}" for m in members[:20]
        )

        prompt = SUMMARY_PROMPT.format(
            meta_summary=meta,
            entities_list=entities_list
        )

        try:
            response = self.llm_client.generate(prompt)
            summaries_data = self._parse_json_response(response)

            summaries = []
            for item in summaries_data:
                summaries.append(Entity(
                    name=item.get("name", f"Summary_{layer}"),
                    entity_type=item.get("type", dominant_type),
                    description=item.get("description", ""),
                    layer=layer,
                ))
            return summaries if summaries else self._create_summary_heuristic(members, layer)

        except Exception as e:
            print(f"[HiIndex] Summary generation error: {e}")
            return self._create_summary_heuristic(members, layer)

    def _create_summary_heuristic(
        self, members: List[Entity], layer: int
    ) -> List[Entity]:
        """Fallback: Tạo summary bằng heuristic (không cần LLM)."""
        type_counts: Dict[str, int] = {}
        for m in members:
            type_counts[m.entity_type] = type_counts.get(m.entity_type, 0) + 1
        dominant_type = max(type_counts, key=type_counts.get)

        # Tạo 1 summary entity đại diện cho cluster
        member_names = [m.name for m in members[:5]]
        summary_name = f"Nhóm: {', '.join(member_names[:3])}"
        if len(member_names) > 3:
            summary_name += f" (+{len(members) - 3} khác)"

        return [Entity(
            name=summary_name,
            entity_type=dominant_type,
            description=f"Summary gồm {len(members)} {dominant_type} entities",
            layer=layer,
        )]

    # ------------------------------------------------------------------
    # Step 4: Community Detection (Leiden Algorithm)
    # ------------------------------------------------------------------
    def _detect_communities(self) -> None:
        """Phát hiện communities trong KG bằng Leiden algorithm."""
        if len(self.graph.nodes) < 3:
            print("[HiIndex] Too few nodes for community detection")
            return

        try:
            import leidenalg
            import igraph as ig

            # Convert NetworkX → igraph
            ig_graph = ig.Graph.from_networkx(self.graph)

            # Run Leiden
            partition = leidenalg.find_partition(
                ig_graph,
                leidenalg.ModularityVertexPartition,
            )

            # Map back to entity IDs
            node_list = list(self.graph.nodes())
            self._community_partition = {}
            for comm_idx, community in enumerate(partition):
                for node_idx in community:
                    if node_idx < len(node_list):
                        self._community_partition[node_list[node_idx]] = comm_idx

            num_communities = len(partition)
            print(f"[HiIndex] Leiden detected {num_communities} communities")

        except ImportError:
            print("[HiIndex] leidenalg not installed → using Louvain fallback")
            try:
                from networkx.algorithms.community import louvain_communities
                communities = louvain_communities(self.graph, seed=3407)
                self._community_partition = {}
                for comm_idx, community in enumerate(communities):
                    for node_id in community:
                        self._community_partition[node_id] = comm_idx
                print(f"[HiIndex] Louvain detected {len(communities)} communities")
            except Exception as e:
                print(f"[HiIndex] Community detection failed: {e}")
                self._community_partition = {}

    # ------------------------------------------------------------------
    # Step 5: Community Reports
    # ------------------------------------------------------------------
    def _generate_community_reports(self) -> None:
        """Tạo community reports cho mỗi community."""
        if not hasattr(self, "_community_partition") or not self._community_partition:
            return

        # Group entities by community
        comm_members: Dict[int, List[str]] = {}
        for entity_id, comm_id in self._community_partition.items():
            comm_members.setdefault(comm_id, []).append(entity_id)

        # Pre-index relations by source_id for O(1) lookup
        relations_by_source: Dict[str, List[Relation]] = {}
        for r in self.relations:
            relations_by_source.setdefault(r.source_id, []).append(r)

        total_comms = len(comm_members)
        for idx, (comm_id, member_ids) in enumerate(comm_members.items()):
            member_set = set(member_ids)  # O(1) lookup instead of O(n)

            members = [
                self.entities[eid] for eid in member_ids
                if eid in self.entities
            ]
            if not members:
                continue

            # Get relations within community — now O(members) instead of O(all_relations)
            comm_relations = []
            for mid in member_ids:
                for r in relations_by_source.get(mid, []):
                    if r.target_id in member_set:
                        comm_relations.append(r)

            # Generate report
            report = self._generate_single_report(members, comm_relations)

            self.communities.append({
                "community_id": comm_id,
                "member_ids": member_ids,
                "member_count": len(members),
                "report": report,
                "entity_types": list(set(m.entity_type for m in members)),
            })

            if (idx + 1) % 50 == 0 or idx + 1 == total_comms:
                print(f"[HiIndex] Community reports: {idx + 1}/{total_comms}")

        print(f"[HiIndex] Generated {len(self.communities)} community reports")

        # Save community reports
        with open(self.cache_dir / "communities.json", "w", encoding="utf-8") as f:
            json.dump(self.communities, f, ensure_ascii=False, indent=2)

    def _generate_single_report(
        self, members: List[Entity], relations: List[Relation]
    ) -> str:
        """Tạo community report cho 1 community."""
        if self.llm_client is None:
            return self._generate_report_heuristic(members, relations)

        entities_list = "\n".join(
            f"- {m.name} ({m.entity_type}): {m.description}" for m in members[:30]
        )
        relations_list = "\n".join(
            f"- {r.description}" for r in relations[:20] if r.description
        )

        prompt = COMMUNITY_REPORT_PROMPT.format(
            entities_list=entities_list,
            relations_list=relations_list or "Không có quan hệ rõ ràng",
        )

        try:
            return self.llm_client.generate(prompt)
        except Exception as e:
            return self._generate_report_heuristic(members, relations)

    def _generate_report_heuristic(
        self, members: List[Entity], relations: List[Relation]
    ) -> str:
        """Fallback: Tạo community report bằng heuristic."""
        type_counts: Dict[str, int] = {}
        for m in members:
            type_counts[m.entity_type] = type_counts.get(m.entity_type, 0) + 1

        types_str = ", ".join(f"{k}({v})" for k, v in type_counts.items())
        names = ", ".join(m.name for m in members[:5])

        return (
            f"Cộng đồng gồm {len(members)} thực thể: {types_str}. "
            f"Các thực thể tiêu biểu: {names}. "
            f"Có {len(relations)} mối quan hệ nội bộ."
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_json_response(response: str) -> List[Dict]:
        """Parse JSON từ LLM response, xử lý các edge case."""
        response = response.strip()

        # Remove markdown code fences if present
        if response.startswith("```"):
            lines = response.split("\n")
            response = "\n".join(lines[1:-1]) if len(lines) > 2 else response

        # Try parsing directly
        try:
            result = json.loads(response)
            return result if isinstance(result, list) else [result]
        except json.JSONDecodeError:
            pass

        # Try finding JSON array in response
        import re
        match = re.search(r"\[.*\]", response, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        return []
