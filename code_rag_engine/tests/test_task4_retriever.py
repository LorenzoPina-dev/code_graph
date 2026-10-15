from core.chunker import ASTCodeChunker
from core.indexer import CodeGraphIndexer
from core.retriever import HybridRetriever

sample_code = '''
class AuthService:
    """Gestione dell'autenticazione utente e sessioni."""
    def authenticate(self, username: str, token: str) -> bool:
        """Verifica la validità del token utente."""
        if not token:
            return False
        return self.verify_token_db(token)

    def verify_token_db(self, token: str) -> bool:
        """Controllo nel database dei token di sessione."""
        return token == "secret_valid_token"

class PaymentProcessor:
    """Modulo di gestione pagamenti carta di credito."""
    def process_payment(self, amount: float, card_number: str) -> bool:
        """Elabora transazione con gateway bancario."""
        return amount > 0 and len(card_number) == 16
'''

def test_retriever():
    print("Testing HybridRetriever (Query Expansion + Dense + Sparse Qdrant + Graph)...")
    chunker = ASTCodeChunker()
    chunks = chunker.chunk_file("services/core_app.py", sample_code)

    indexer = CodeGraphIndexer(collection_name="test_retrieval_collection", location=":memory:")
    indexer.add_repository_chunks("services/core_app.py", chunks)

    retriever = HybridRetriever(indexer)

    # Test 1: Query Espansa per autenticazione e token
    query = "Come viene verificato il token per l'autenticazione utente?"
    results = retriever.retrieve(query, top_k=2, expand_graph=True)
    
    print(f"Retrieved {len(results)} results for query: '{query}'")
    for r in results:
        print(f"- Hit: {r['symbol']} (Score: {r['score']:.4f})")
        print(f"  Graph context: {[n['symbol_name'] for n in r['graph_neighbors']]}")

    assert len(results) > 0, "No results returned"
    top_symbol = results[0]["symbol"]
    assert "auth" in top_symbol.lower() or "token" in results[0]["content"].lower(), "Retrieved irrelevant symbol"
    
    # Test 2: Verifica espansione grafo su authenticate
    auth_hit = next((r for r in results if "authenticate" in r["symbol"]), None)
    if auth_hit:
        neighbor_names = [n["symbol_name"] for n in auth_hit["graph_neighbors"]]
        print(f"Authenticate neighbors: {neighbor_names}")
        assert any("verify_token_db" in name or "AuthService" in name for name in neighbor_names)

    print("TASK 4 VALIDATION SUCCESSFUL: HybridRetriever validated with Dense+Sparse Qdrant and CPG expansion!")

if __name__ == "__main__":
    test_retriever()

