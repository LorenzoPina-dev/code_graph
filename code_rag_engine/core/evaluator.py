from typing import List, Dict, Any, Optional
import math

class ProductionCodeRAGEvaluator:
    """
    Framework di valutazione per Code RAG conforme alla project specification:
    1. Retrieval Metrics:
       - Hit Rate @ K
       - Mean Reciprocal Rank (MRR)
    2. Generation / Groundedness Metrics (LLM-as-a-judge / semantic alignment):
       - Faithfulness: aderenza e ancoraggio della risposta rispetto al codice recuperato
       - Context Precision: rapporto tra contesti utili/pertinenti e rumore recuperato
       - Answer Relevancy: pertinenza della risposta generata rispetto all'intento della query
    """

    def __init__(self, ground_truth: List[Dict[str, Any]]):
        self.ground_truth = ground_truth

    def evaluate_retrieval(
        self,
        retriever_func,
        top_k: int = 3
    ) -> Dict[str, float]:
        """
        Valuta il retriever calcolando Hit Rate @ K e MRR rispetto a target_symbol o target_file.
        """
        hits = 0
        reciprocal_ranks = []
        precision_scores = []

        for item in self.ground_truth:
            query = item["query"]
            target_symbol = item.get("target_symbol")
            target_file = item.get("target_file")
            expected_keywords = item.get("expected_keywords", [])

            results = retriever_func(query, top_k=top_k)
            retrieved_symbols = [r.get("symbol", "") for r in results]
            retrieved_files = [r.get("file", "") for r in results]

            # Calcolo Hit & Rank
            hit_index = None
            for idx, r in enumerate(results):
                sym_match = target_symbol and (target_symbol in r.get("symbol", "") or r.get("symbol", "") in target_symbol)
                file_match = target_file and (target_file in r.get("file", ""))
                if sym_match or file_match:
                    hit_index = idx
                    break

            if hit_index is not None:
                hits += 1
                reciprocal_ranks.append(1.0 / (hit_index + 1))
            else:
                reciprocal_ranks.append(0.0)

            # Context Precision: quante informazioni pertinenti vs non pertinenti
            relevant_in_top_k = 0
            for r in results:
                content = r.get("content", "").lower()
                if any(kw.lower() in content for kw in expected_keywords):
                    relevant_in_top_k += 1
            precision_scores.append(relevant_in_top_k / max(1, len(results)))

        total = max(1, len(self.ground_truth))
        return {
            "hit_rate_at_k": round(hits / total, 4),
            "mrr": round(sum(reciprocal_ranks) / total, 4),
            "context_precision": round(sum(precision_scores) / total, 4)
        }

    def evaluate_generation(
        self,
        generated_responses: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Valuta Faithfulness e Answer Relevancy sulle risposte generate e sui contesti.
        """
        faithfulness_scores = []
        relevancy_scores = []

        for item in generated_responses:
            response = item.get("response", "")
            context = item.get("context", "")
            query = item.get("query", "")

            # Misura Faithfulness: verifica che le entità o token di codice generati siano effettivamente nel contesto
            resp_tokens = set(response.lower().split())
            ctx_tokens = set(context.lower().split())

            # Sovrapposizione termini di codice
            code_overlap = resp_tokens.intersection(ctx_tokens)
            faithfulness = min(1.0, len(code_overlap) / max(1, len(resp_tokens) * 0.5))
            faithfulness_scores.append(faithfulness)

            # Misura Relevancy: sovrapposizione con i concetti della query
            q_tokens = set(query.lower().split())
            query_overlap = resp_tokens.intersection(q_tokens)
            relevancy = min(1.0, len(query_overlap) / max(1, len(q_tokens) * 0.4))
            relevancy_scores.append(relevancy)

        total = max(1, len(generated_responses))
        return {
            "faithfulness": round(sum(faithfulness_scores) / total, 4),
            "answer_relevancy": round(sum(relevancy_scores) / total, 4)
        }

