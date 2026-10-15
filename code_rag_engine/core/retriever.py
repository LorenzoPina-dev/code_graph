import re
from typing import List, Dict, Any, Optional
from qdrant_client.http import models as qmodels
from .indexer import CodeGraphIndexer

class HybridRetriever:
    """
    Retriever ibrido avanzato per Code RAG:
    1. Query Rewriting & Technical Term Expansion:
       - Estrae parole chiave sintattiche, identificatori, camelCase e snake_case
       - Genera varianti di query tecnica
    2. Hybrid Search (Qdrant Dense + Sparse/BM25):
       - Esegue query densa per similarità semantica
       - Esegue query sparsa per keyword esatte
       - Fonde i risultati tramite Reciprocal Rank Fusion (RRF)
    3. Graph-Augmented Retrieval:
       - Per ciascun chunk rilevato, interroga il CPG (NetworkX)
       - Arricchisce il contesto con simboli chiamati o classi genitrici
    """

    def __init__(self, indexer: CodeGraphIndexer):
        self.indexer = indexer

    def expand_query(self, user_query: str) -> List[str]:
        """
        Estrae termini tecnici, scompone identificatori composti (camelCase, snake_case)
        e genera varianti di ricerca mirate.
        """
        queries = [user_query]
        tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", user_query)
        expanded_terms = set()

        for token in tokens:
            # Scomposizione camelCase (es. AuthService -> Auth, Service)
            camel_split = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)", token)
            if len(camel_split) > 1:
                expanded_terms.update([p.lower() for p in camel_split])
            
            # Scomposizione snake_case (es. verify_token_db -> verify, token, db)
            if "_" in token:
                snake_parts = token.split("_")
                expanded_terms.update([p.lower() for p in snake_parts if len(p) > 1])

        # Se ci sono termini tecnici espansi, crea una query di keyword specifica
        if expanded_terms:
            queries.append(" ".join(list(expanded_terms)))

        return queries

    def _reciprocal_rank_fusion(
        self,
        dense_results: List[Any],
        sparse_results: List[Any],
        k: int = 60
    ) -> Dict[str, Dict[str, Any]]:
        """
        Fonde e ordina i risultati delle ricerche dense e sparse tramite RRF.
        """
        fused_scores = {}

        for rank, hit in enumerate(dense_results):
            doc_id = hit.id
            if doc_id not in fused_scores:
                fused_scores[doc_id] = {
                    "score": 0.0,
                    "payload": hit.payload,
                    "dense_score": hit.score,
                    "sparse_score": 0.0
                }
            fused_scores[doc_id]["score"] += 1.0 / (k + rank + 1)

        for rank, hit in enumerate(sparse_results):
            doc_id = hit.id
            if doc_id not in fused_scores:
                fused_scores[doc_id] = {
                    "score": 0.0,
                    "payload": hit.payload,
                    "dense_score": 0.0,
                    "sparse_score": hit.score
                }
            fused_scores[doc_id]["score"] += 1.0 / (k + rank + 1)
            fused_scores[doc_id]["sparse_score"] = hit.score

        return fused_scores

    def retrieve(
        self,
        user_query: str,
        top_k: int = 5,
        expand_graph: bool = True,
        graph_depth: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Esegue la ricerca ibrida su Qdrant ed espande il contesto tramite il Grafo CPG.
        """
        expanded_queries = self.expand_query(user_query)
        search_query_text = " ".join(expanded_queries)

        # 1. Genera vettori di query
        dense_vec = list(self.indexer.dense_model.embed([search_query_text]))[0].tolist()
        sparse_vec_obj = list(self.indexer.sparse_model.embed([search_query_text]))[0]

        q_sparse = qmodels.SparseVector(
            indices=sparse_vec_obj.indices.tolist(),
            values=sparse_vec_obj.values.tolist()
        )

        # 2. Esegui ricerca densa su Qdrant
        dense_hits = self.indexer.client.query_points(
            collection_name=self.indexer.collection_name,
            query=dense_vec,
            using="dense",
            limit=top_k * 2
        ).points

        # 3. Esegui ricerca sparsa (BM25) su Qdrant
        sparse_hits = self.indexer.client.query_points(
            collection_name=self.indexer.collection_name,
            query=q_sparse,
            using="sparse",
            limit=top_k * 2
        ).points

        # 4. RRF Fusion
        fused = self._reciprocal_rank_fusion(dense_hits, sparse_hits)
        sorted_docs = sorted(fused.values(), key=lambda x: x["score"], reverse=True)[:top_k]

        results = []
        for doc in sorted_docs:
            payload = doc["payload"]
            node_id = payload.get("node_id")

            graph_neighbors = []
            if expand_graph and node_id:
                graph_neighbors = self.indexer.get_neighbors_context(node_id, depth=graph_depth)

            results.append({
                "id": node_id,
                "file": payload.get("file_path"),
                "symbol": payload.get("symbol_name"),
                "symbol_type": payload.get("symbol_type"),
                "start_line": payload.get("start_line"),
                "end_line": payload.get("end_line"),
                "content": payload.get("code_content"),
                "docstring": payload.get("docstring"),
                "dependencies": payload.get("dependencies", []),
                "calls": payload.get("calls", []),
                "score": doc["score"],
                "graph_neighbors": graph_neighbors
            })

        return results