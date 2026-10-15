from core.chunker import ASTCodeChunker
from core.indexer import CodeGraphIndexer

sample_code = '''
class AuthService:
    """Gestione dell'autenticazione."""
    def authenticate(self, username: str, token: str) -> bool:
        """Verifica la validità del token utente."""
        if not token:
            return False
        return self.verify_token_db(token)

    def verify_token_db(self, token: str) -> bool:
        """Verifica diretta sul database."""
        return token == "secret_valid_token"
'''

def test_indexer():
    print("Testing CodeGraphIndexer with Qdrant and CPG...")
    chunker = ASTCodeChunker()
    chunks = chunker.chunk_file("services/auth.py", sample_code)
    
    indexer = CodeGraphIndexer(location=":memory:")
    indexer.add_repository_chunks("services/auth.py", chunks)
    
    # 1. Verifica Qdrant
    points_count = indexer.client.count(collection_name=indexer.collection_name).count
    print(f"Points in Qdrant collection: {points_count}")
    assert points_count == len(chunks), f"Expected {len(chunks)} points, found {points_count}"
    
    # 2. Verifica CPG Nodes & Edges
    print(f"Nodes in CPG: {indexer.graph.number_of_nodes()}")
    print(f"Edges in CPG: {indexer.graph.number_of_edges()}")
    assert indexer.graph.number_of_nodes() == len(chunks)
    
    # Verifica che authenticate sia collegato a verify_token_db o a AuthService
    auth_node = "services/auth.py::AuthService.authenticate"
    neighbors = indexer.get_neighbors_context(auth_node)
    print(f"Neighbors for {auth_node}: {[n['symbol_name'] for n in neighbors]}")
    assert len(neighbors) > 0, "Expected at least one neighbor relation (calls or contains)"
    
    print("TASK 3 VALIDATION SUCCESSFUL: Qdrant Dense+Sparse index and CPG graph relations verified!")

if __name__ == "__main__":
    test_indexer()

