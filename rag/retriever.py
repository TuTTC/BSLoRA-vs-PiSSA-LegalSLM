"""
Hierarchical Retriever (HiRetrieval)
=====================================
Truy xuất 3 cấp độ kiến thức từ Hierarchical KG (theo HiRAG, Section 4.2):
  1. Local-level: Top-n entities gần nhất (cosine similarity)
  2. Global-level: Community reports chứa entities đã truy xuất
  3. Bridge-level: Shortest paths nối local ↔ global (reasoning paths)

Nguồn tham chiếu:
  - HiRAG: arxiv 2503.10150 (Section 4.2: Retrieval with Hierarchical Knowledge)

Usage:
    retriever = HiRetriever(kg, embedding_model)
    context = retriever.retrieve("Trách nhiệm bồi thường thiệt hại?")
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple

try:
    import networkx as nx
except ImportError:
    raise ImportError("networkx is required: pip install networkx>=3.0")


class HiRetriever:
    """
    Truy xuất kiến thức 3 cấp từ Hierarchical KG.

    Theo HiRAG Section 4.2:
      - Local: Sim(query, entity) → top-n entities
      - Global: Communities chứa local entities → community reports
      - Bridge: Shortest paths giữa key entities từ mỗi community
    """

    def __init__(
        self,
        knowledge_graph: Any,     # HiIndex instance
        embedding_model: Any,     # SentenceTransformer
        top_n: int = 20,          # Số entity local cần truy xuất
        top_m: int = 5,           # Số key entities mỗi community cho bridge
        max_context_length: int = 3000,  # Max ký tự context trả về
    ):
        self.kg = knowledge_graph
        self.embedding_model = embedding_model
        self.top_n = top_n
        self.top_m = top_m
        self.max_context_length = max_context_length

        # Pre-compute entity embeddings matrix
        self._build_index()

    def _build_index(self) -> None:
        """Xây dựng index cho retrieval nhanh."""
        self.entity_ids = []
        self.entity_embeddings = []

        for eid, entity in self.kg.entities.items():
            if entity.embedding is not None:
                self.entity_ids.append(eid)
                self.entity_embeddings.append(entity.embedding)

        if self.entity_embeddings:
            self.entity_embeddings = np.array(self.entity_embeddings)
            # Normalize for cosine similarity
            norms = np.linalg.norm(self.entity_embeddings, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)
            self.entity_embeddings_normalized = self.entity_embeddings / norms
            print(f"[HiRetriever] Built index with {len(self.entity_ids)} entities")
        else:
            self.entity_embeddings_normalized = np.array([])
            print("[HiRetriever] Warning: no entity embeddings available")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def retrieve(self, query: str) -> Dict[str, Any]:
        """
        Truy xuất 3-level context cho query.

        Args:
            query: Câu hỏi pháp lý

        Returns:
            Dict chứa:
              - "context": str — context string sẵn sàng nhúng vào prompt
              - "local_entities": List — top-n entities
              - "global_communities": List — matching community reports
              - "bridge_paths": List — reasoning paths
        """
        # Step 1: Local — top-n entities (Eq. 12 in paper)
        local_entities = self._retrieve_local(query)

        # Step 2: Global — community reports (Eq. 13 in paper)
        global_communities = self._retrieve_global(local_entities)

        # Step 3: Bridge — shortest paths (Eq. 14-16 in paper)
        bridge_paths = self._retrieve_bridge(local_entities, global_communities)

        # Assemble context
        context = self._assemble_context(
            local_entities, global_communities, bridge_paths
        )

        return {
            "context": context,
            "local_entities": local_entities,
            "global_communities": global_communities,
            "bridge_paths": bridge_paths,
        }

    # ------------------------------------------------------------------
    # Step 1: Local-level Retrieval
    # ------------------------------------------------------------------
    def _retrieve_local(self, query: str) -> List[Dict[str, Any]]:
        """
        Tìm top-n entity gần nhất bằng cosine similarity.
        (Equation 12 in HiRAG paper)
        """
        if len(self.entity_embeddings_normalized) == 0:
            return []

        # Encode query
        query_embedding = self.embedding_model.encode([query])[0]
        query_norm = np.linalg.norm(query_embedding)
        if query_norm > 0:
            query_embedding = query_embedding / query_norm

        # Compute cosine similarities
        similarities = np.dot(self.entity_embeddings_normalized, query_embedding)

        # Get top-n
        top_indices = np.argsort(similarities)[::-1][:self.top_n]

        results = []
        for idx in top_indices:
            eid = self.entity_ids[idx]
            entity = self.kg.entities[eid]
            results.append({
                "entity_id": eid,
                "name": entity.name,
                "entity_type": entity.entity_type,
                "description": entity.description,
                "score": float(similarities[idx]),
                "layer": entity.layer,
            })

        return results

    # ------------------------------------------------------------------
    # Step 2: Global-level Retrieval
    # ------------------------------------------------------------------
    def _retrieve_global(
        self, local_entities: List[Dict]
    ) -> List[Dict[str, Any]]:
        """
        Lấy community reports chứa các entity đã truy xuất.
        (Equation 13 in HiRAG paper)
        """
        # Collect entity IDs from local results
        local_ids = set(e["entity_id"] for e in local_entities)

        matching_communities = []
        for community in self.kg.communities:
            member_ids = set(community.get("member_ids", []))
            overlap = local_ids & member_ids

            if overlap:
                matching_communities.append({
                    "community_id": community["community_id"],
                    "report": community["report"],
                    "overlap_count": len(overlap),
                    "member_count": community["member_count"],
                    "entity_types": community.get("entity_types", []),
                    "overlapping_entities": list(overlap),
                })

        # Sort by overlap count (most relevant first)
        matching_communities.sort(key=lambda x: x["overlap_count"], reverse=True)

        return matching_communities

    # ------------------------------------------------------------------
    # Step 3: Bridge-level Retrieval
    # ------------------------------------------------------------------
    def _retrieve_bridge(
        self,
        local_entities: List[Dict],
        communities: List[Dict],
    ) -> List[Dict[str, Any]]:
        """
        Tìm shortest paths giữa key entities từ các community.
        (Equations 14-16 in HiRAG paper)
        """
        if not communities or len(communities) < 2:
            return []

        # Collect top-m key entities from each community (Eq. 14)
        key_entities = []
        for comm in communities[:5]:
            overlapping = comm.get("overlapping_entities", [])
            key_entities.extend(overlapping[:self.top_m])

        # Remove duplicates while preserving order
        seen = set()
        unique_keys = []
        for eid in key_entities:
            if eid not in seen:
                seen.add(eid)
                unique_keys.append(eid)

        # Find shortest paths between consecutive key entities (Eq. 15)
        paths = []
        for i in range(len(unique_keys) - 1):
            src = unique_keys[i]
            tgt = unique_keys[i + 1]

            if src in self.kg.graph and tgt in self.kg.graph:
                try:
                    path = nx.shortest_path(self.kg.graph, src, tgt)
                    path_info = self._describe_path(path)
                    paths.append(path_info)
                except nx.NetworkXNoPath:
                    continue
                except nx.NodeNotFound:
                    continue

        return paths

    def _describe_path(self, path: List[str]) -> Dict[str, Any]:
        """Mô tả reasoning path bằng text."""
        descriptions = []
        for node_id in path:
            if node_id in self.kg.entities:
                entity = self.kg.entities[node_id]
                descriptions.append(f"{entity.name} ({entity.entity_type})")

        # Get edge descriptions
        edge_descriptions = []
        for i in range(len(path) - 1):
            edge_data = self.kg.graph.get_edge_data(path[i], path[i + 1], default={})
            rel_type = edge_data.get("relation_type", "liên quan đến")
            edge_descriptions.append(rel_type)

        return {
            "path": path,
            "entities": descriptions,
            "relations": edge_descriptions,
            "description": " → ".join(descriptions),
        }

    # ------------------------------------------------------------------
    # Context Assembly
    # ------------------------------------------------------------------
    def _assemble_context(
        self,
        local_entities: List[Dict],
        communities: List[Dict],
        bridge_paths: List[Dict],
    ) -> str:
        """
        Ghép 3-level context thành 1 chuỗi sẵn sàng nhúng vào prompt.

        Format:
            [NGUỒN THAM CHIẾU]
            1. Thực thể liên quan: ...
            2. Ngữ cảnh tổng quát: ...
            3. Mối liên hệ: ...
        """
        parts = []

        # Local: Entity descriptions
        if local_entities:
            entity_desc = []
            for e in local_entities[:10]:  # Limit to top 10
                entity_desc.append(f"- {e['name']} ({e['entity_type']}): {e['description']}")
            parts.append(
                "### Thực thể pháp lý liên quan:\n" + "\n".join(entity_desc)
            )

        # Global: Community reports
        if communities:
            comm_desc = []
            for c in communities[:3]:  # Limit to top 3
                comm_desc.append(f"- {c['report']}")
            parts.append(
                "### Ngữ cảnh pháp lý tổng quát:\n" + "\n".join(comm_desc)
            )

        # Bridge: Reasoning paths
        if bridge_paths:
            path_desc = []
            for p in bridge_paths[:5]:  # Limit to top 5
                path_desc.append(f"- {p['description']}")
            parts.append(
                "### Mối liên hệ pháp lý:\n" + "\n".join(path_desc)
            )

        context = "\n\n".join(parts)

        # Truncate if too long
        if len(context) > self.max_context_length:
            context = context[:self.max_context_length] + "\n[... truncated]"

        return context
