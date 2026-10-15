import os
from typing import List, Dict, Any, Optional
from sentence_transformers import CrossEncoder

class CodeReranker:
    """
    Reranker di produzione basato su Cross-Encoder (Zero Mock).
    Valuta congiuntamente la query e ciascun candidato estratto dal retriever,
    calcolando l'attenzione incrociata su sintassi e semantica.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Inizializza il modello Cross-Encoder.
        Di default utilizza un modello efficiente, sostituibile con BAAI/bge-reranker-large o Cohere Rerank.
        """
        self.model_name = model_name
        self._model = None

    @property
    def model(self) -> CrossEncoder:
        if self._model is None:
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_n: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Riorina chirurgicamente i candidati tramite Cross-Encoder.
        Prepara le coppie (query, text_candidato) arricchite da metadati contestuali.
        """
        if not candidates:
            return []

        if len(candidates) <= 1:
            return candidates[:top_n]

        # Prepara coppie per l'inferenza del Cross-Encoder
        pairs = []
        for cand in candidates:
            symbol = cand.get("symbol", "")
            file_path = cand.get("file", "")
            docstring = cand.get("docstring", "")
            content = cand.get("content", "")

            # Contesto arricchito con grafi (se presenti)
            graph_context_str = ""
            if "graph_neighbors" in cand and cand["graph_neighbors"]:
                related = [f"{n.get('symbol_name', '')} ({n.get('relation_type', '')})" for n in cand["graph_neighbors"]]
                graph_context_str = f"Related symbols: {', '.join(related)}\n"

            candidate_repr = (
                f"File: {file_path}\n"
                f"Symbol: {symbol}\n"
                f"{graph_context_str}"
                f"Docstring: {docstring}\n"
                f"Code:\n{content}"
            )
            pairs.append([query, candidate_repr])

        # Calcola gli score del Cross-Encoder
        scores = self.model.predict(pairs)

        # Associa i punteggi e ordina
        scored_candidates = []
        for cand, score in zip(candidates, scores):
            cand_copy = dict(cand)
            cand_copy["rerank_score"] = float(score)
            scored_candidates.append(cand_copy)

        scored_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        return scored_candidates[:top_n]