import uuid
from typing import List, Dict, Any, Optional
import networkx as nx
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from fastembed import TextEmbedding, SparseTextEmbedding
from .chunker import CodeChunk

class CodeGraphIndexer:
    """
    Indicizzatore orientato alla produzione per Enterprise Code RAG:
    1. Qdrant Vector Store (supporto memoria locale o cluster remoto):
       - Dense vectors (semantic embedding specializzato)
       - Sparse vectors (lexical/BM25 per keyword esatte su simboli e variabili)
       - Payload strutturato (file_path, symbol_name, symbol_type, start_line, end_line, language, dependencies, calls)
    2. NetworkX Code Property Graph (CPG):
       - Nodi: Simboli (funzioni, metodi, classi, moduli)
       - Archi: 'calls', 'contains', 'imports', 'inherits'
    """

    def __init__(
        self,
        collection_name: str = "code_repository_index",
        location: str = ":memory:",
        dense_model_name: str = "BAAI/bge-small-en-v1.5",
        sparse_model_name: str = "Qdrant/bm25"
    ):
        self.collection_name = collection_name
        self.client = QdrantClient(location=location)
        self.graph = nx.DiGraph()

        # Inizializza modelli Dense e Sparse (BM25) da fastembed
        self.dense_model = TextEmbedding(model_name=dense_model_name)
        self.sparse_model = SparseTextEmbedding(model_name=sparse_model_name)

        self._init_collection()

    def _init_collection(self):
        """Inizializza la collezione Qdrant configurando vettori densi e sparsi."""
        if self.client.collection_exists(self.collection_name):
            return

        # Dimensione del modello bge-small-en-v1.5 è 384
        dense_sample = list(self.dense_model.embed(["probe"]))[0]
        dim = len(dense_sample)

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                "dense": qmodels.VectorParams(
                    size=dim,
                    distance=qmodels.Distance.COSINE
                )
            },
            sparse_vectors_config={
                "sparse": qmodels.SparseVectorParams(
                    index=qmodels.SparseIndexParams(on_disk=False)
                )
            }
        )

    def add_repository_chunks(self, file_path: str, chunks: List[CodeChunk]):
        """
        Indicizza una lista di chunk nel vector store Qdrant e popola il grafo CPG.
        """
        self.add_chunks_batch([(file_path, c) for c in chunks])

    def add_chunks_batch(self, file_chunk_pairs: List[tuple], resolve_calls: bool = True):
        """
        Indicizzazione in batch ad alte prestazioni:
        - Batch embeddings densi + sparsi in un unico passaggio
        - Upsert Qdrant in blocchi da 200 (ottimizzato per throughput)
        - resolve_calls=False posticipa la risoluzione dei call-edges (usa a fine ingestione)
        - Testo di embedding troncato a 1024 caratteri per evitare token overflow
        """
        if not file_chunk_pairs:
            return

        texts_to_embed = []
        chunks = []
        file_paths = []

        MAX_EMBED_CHARS = 1024

        for file_path, chunk in file_chunk_pairs:
            node_id = f"{file_path}::{chunk.symbol_name}"
            chunks.append(chunk)
            file_paths.append(file_path)

            # 1. Popola nodo CPG
            self.graph.add_node(
                node_id,
                file_path=file_path,
                symbol_name=chunk.symbol_name,
                symbol_type=chunk.symbol_type,
                start_line=chunk.start_line,
                end_line=chunk.end_line,
                docstring=chunk.docstring,
                dependencies=chunk.dependencies,
                calls=chunk.calls,
                content=chunk.code_content
            )

            # Contenimento
            if chunk.parent_symbol:
                parent_id = f"{file_path}::{chunk.parent_symbol}"
                self.graph.add_edge(parent_id, node_id, relation="contains")

            embed_text = (
                f"File: {file_path}\n"
                f"Symbol: {chunk.symbol_name} ({chunk.symbol_type})\n"
                f"Docstring: {chunk.docstring}\n"
                f"{chunk.code_content}"
            )[:MAX_EMBED_CHARS]
            texts_to_embed.append(embed_text)

        # Batch embeddings
        dense_vectors = list(self.dense_model.embed(texts_to_embed))
        sparse_vectors = list(self.sparse_model.embed(texts_to_embed))

        points = []
        for idx, chunk in enumerate(chunks):
            fpath = file_paths[idx]
            node_id = f"{fpath}::{chunk.symbol_name}"
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, node_id))

            points.append(
                qmodels.PointStruct(
                    id=point_id,
                    vector={
                        "dense": dense_vectors[idx].tolist(),
                        "sparse": qmodels.SparseVector(
                            indices=sparse_vectors[idx].indices.tolist(),
                            values=sparse_vectors[idx].values.tolist()
                        )
                    },
                    payload={
                        "node_id": node_id,
                        "file_path": fpath,
                        "symbol_name": chunk.symbol_name,
                        "symbol_type": chunk.symbol_type,
                        "start_line": chunk.start_line,
                        "end_line": chunk.end_line,
                        "language": chunk.language,
                        "docstring": chunk.docstring,
                        "dependencies": chunk.dependencies,
                        "calls": chunk.calls,
                        "code_content": chunk.code_content
                    }
                )
            )

        # Upsert in blocchi da 200 (↑ rispetto a 100)
        qdrant_batch = 200
        for i in range(0, len(points), qdrant_batch):
            self.client.upsert(
                collection_name=self.collection_name,
                points=points[i:i + qdrant_batch]
            )

        if resolve_calls:
            self._resolve_graph_calls()


    def _resolve_graph_calls(self):
        """Collega archi 'calls' tra i simboli noti nel grafo CPG."""
        nodes = dict(self.graph.nodes(data=True))
        symbol_map = {}
        for nid, data in nodes.items():
            sym = data.get("symbol_name")
            if sym:
                # Mappa sia 'foo' che 'MyClass.foo'
                symbol_map[sym] = nid
                if "." in sym:
                    short = sym.split(".")[-1]
                    symbol_map[short] = nid

        for nid, data in nodes.items():
            calls = data.get("calls", [])
            for c in calls:
                target_id = symbol_map.get(c)
                if target_id and target_id != nid:
                    self.graph.add_edge(nid, target_id, relation="calls")

    def add_call_relation(self, caller_id: str, callee_id: str):
        """Aggiunge una relazione esplicita di chiamata tra due simboli."""
        if self.graph.has_node(caller_id) and self.graph.has_node(callee_id):
            self.graph.add_edge(caller_id, callee_id, relation="calls")

    def get_neighbors_context(self, node_id: str, depth: int = 1) -> List[Dict[str, Any]]:
        """
        Recupera il contesto adiacente nel grafo (simboli chiamati, classi padre o contenute).
        """
        if node_id not in self.graph:
            return []

        neighbors = []
        visited = {node_id}
        frontier = [node_id]

        for _ in range(depth):
            next_frontier = []
            for curr in frontier:
                # Nodi in uscita (funzioni chiamate, figli)
                for successor in self.graph.successors(curr):
                    if successor not in visited:
                        visited.add(successor)
                        edge_data = self.graph.get_edge_data(curr, successor)
                        rel = edge_data.get("relation", "related")
                        data = dict(self.graph.nodes[successor])
                        data["relation_type"] = f"outgoing_{rel}"
                        neighbors.append(data)
                        next_frontier.append(successor)

                # Nodi in entrata (chi chiama questa funzione)
                for predecessor in self.graph.predecessors(curr):
                    if predecessor not in visited:
                        visited.add(predecessor)
                        edge_data = self.graph.get_edge_data(predecessor, curr)
                        rel = edge_data.get("relation", "related")
                        data = dict(self.graph.nodes[predecessor])
                        data["relation_type"] = f"incoming_{rel}"
                        neighbors.append(data)
                        next_frontier.append(predecessor)

            frontier = next_frontier

        return neighbors