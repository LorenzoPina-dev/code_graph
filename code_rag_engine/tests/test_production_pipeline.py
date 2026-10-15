import os
import pytest
from core.chunker import ASTCodeChunker
from core.indexer import CodeGraphIndexer
from core.retriever import HybridRetriever
from core.reranker import CodeReranker
from core.evaluator import ProductionCodeRAGEvaluator

SAMPLE_CODE = '''
class AuthService:
    """Modulo di autenticazione utente."""
    def authenticate(self, username: str, token: str) -> bool:
        """Verifica la validità del token utente."""
        if not token:
            return False
        return self.verify_token_db(token)

    def verify_token_db(self, token: str) -> bool:
        """Controllo nel database dei token di sessione."""
        return token == "secret_valid_token"

class OrderProcessor:
    """Modulo di gestione ordini e carrello."""
    def calculate_total(self, items: list, discount: float = 0.0) -> float:
        """Calcola totale applicando sconti."""
        total = sum(i['price'] for i in items)
        return total * (1.0 - discount)
'''

@pytest.fixture(scope="module")
def setup_engine():
    chunker = ASTCodeChunker()
    chunks = chunker.chunk_file("services/core.py", SAMPLE_CODE)
    
    indexer = CodeGraphIndexer(collection_name="pytest_suite_collection", location=":memory:")
    indexer.add_repository_chunks("services/core.py", chunks)
    
    retriever = HybridRetriever(indexer)
    reranker = CodeReranker()
    
    return {
        "chunker": chunker,
        "indexer": indexer,
        "retriever": retriever,
        "reranker": reranker,
        "chunks": chunks
    }

def test_ast_chunking_integrity(setup_engine):
    chunks = setup_engine["chunks"]
    symbols = [c.symbol_name for c in chunks]
    assert "AuthService" in symbols
    assert "AuthService.authenticate" in symbols
    assert "AuthService.verify_token_db" in symbols
    assert "OrderProcessor.calculate_total" in symbols

    # Verifica docstring
    auth_chunk = next(c for c in chunks if c.symbol_name == "AuthService.authenticate")
    assert "validità del token" in auth_chunk.docstring

def test_qdrant_cpg_indexing(setup_engine):
    indexer = setup_engine["indexer"]
    count = indexer.client.count(collection_name=indexer.collection_name).count
    assert count == len(setup_engine["chunks"])
    assert indexer.graph.number_of_nodes() == len(setup_engine["chunks"])
    assert indexer.graph.number_of_edges() > 0

def test_hybrid_retrieval_and_graph(setup_engine):
    retriever = setup_engine["retriever"]
    query = "Come viene verificato il token per l'autenticazione?"
    results = retriever.retrieve(query, top_k=2, expand_graph=True)
    
    assert len(results) > 0
    top_hit = results[0]
    assert "auth" in top_hit["symbol"].lower() or "token" in top_hit["content"].lower()
    assert "graph_neighbors" in top_hit

def test_cross_encoder_rerank(setup_engine):
    retriever = setup_engine["retriever"]
    reranker = setup_engine["reranker"]
    
    query = "Come si calcolano gli sconti sul totale del carrello ordini?"
    candidates = retriever.retrieve(query, top_k=4, expand_graph=False)
    
    reranked = reranker.rerank(query, candidates, top_n=1)
    assert len(reranked) == 1
    assert "calculate_total" in reranked[0]["symbol"] or "OrderProcessor" in reranked[0]["symbol"]

def test_evaluation_metrics_suite(setup_engine):
    retriever = setup_engine["retriever"]
    
    ground_truth = [
        {
            "query": "autenticazione tramite token utente",
            "target_symbol": "AuthService.authenticate",
            "expected_keywords": ["token", "verify_token_db"]
        },
        {
            "query": "calcolo sconti e totale ordine",
            "target_symbol": "OrderProcessor.calculate_total",
            "expected_keywords": ["discount", "total"]
        }
    ]

    evaluator = ProductionCodeRAGEvaluator(ground_truth)
    metrics = evaluator.evaluate_retrieval(lambda q, top_k: retriever.retrieve(q, top_k=top_k), top_k=2)

    assert metrics["hit_rate_at_k"] >= 0.5
    assert metrics["mrr"] > 0.0
    assert metrics["context_precision"] > 0.0

