"""
Pipeline specializzata per ingerire interi dataset di codice da Hugging Face / librerie
e costruire l'intero Code Property Graph (CPG) e indice vettoriale Qdrant.
"""

from typing import Dict, Any, Optional
from datasets import load_dataset
from .chunker import ASTCodeChunker
from .indexer import CodeGraphIndexer
from .visualizer import Graph3DVisualizer

class DatasetIngestionPipeline:
    def __init__(self, indexer: CodeGraphIndexer, chunker: Optional[ASTCodeChunker] = None):
        self.indexer = indexer
        self.chunker = chunker or ASTCodeChunker()

    def ingest_humaneval(
        self,
        split: str = "test",
        max_samples: Optional[int] = None,
        batch_size: int = 50
    ) -> Dict[str, Any]:
        """
        Scarica ed ingerisce l'intero dataset 'openai/openai_humaneval' (tutti i 164 task di default),
        costruendo l'albero sintattico AST, i nodi e le relazioni di chiamata e contenimento nel CPG,
        e indicizzando vettorialmente in Qdrant.
        """
        print(f"[*] Ingestione dataset 'openai/openai_humaneval' (split: {split})...")
        dataset = load_dataset("openai/openai_humaneval", split=split)
        
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))

        total_items = len(dataset)
        print(f"[*] Totale campioni da processare: {total_items}")

        all_file_chunk_pairs = []
        task_entries = []

        for idx, item in enumerate(dataset):
            task_id = item["task_id"]
            prompt = item["prompt"]
            canonical_solution = item["canonical_solution"]
            full_code = prompt + "\n" + canonical_solution
            file_path = f"humaneval/task_{idx:03d}_{item['entry_point']}.py"

            # 1. Parsing AST
            chunks = self.chunker.chunk_file(file_path, full_code)
            for c in chunks:
                all_file_chunk_pairs.append((file_path, c))

            task_entries.append({
                "task_id": task_id,
                "file_path": file_path,
                "entry_point": item["entry_point"],
                "chunks_count": len(chunks)
            })

            # Ingestione incrementale in batch per efficienza di memoria e throughput
            if len(all_file_chunk_pairs) >= batch_size:
                self.indexer.add_chunks_batch(all_file_chunk_pairs)
                all_file_chunk_pairs = []
                print(f"  -> Progresso: {idx + 1}/{total_items} task elaborati...")

        # Elabora l'eventuale rimanenza
        if all_file_chunk_pairs:
            self.indexer.add_chunks_batch(all_file_chunk_pairs)

        # Risoluzione finale delle relazioni globali nel CPG
        self.indexer._resolve_graph_calls()

        stats = {
            "dataset_name": "openai/openai_humaneval",
            "total_tasks_ingested": total_items,
            "qdrant_points": self.indexer.client.count(collection_name=self.indexer.collection_name).count,
            "cpg_nodes": self.indexer.graph.number_of_nodes(),
            "cpg_edges": self.indexer.graph.number_of_edges()
        }
        print(f"[*] Ingestione completata: {stats}")
        return stats

    def export_3d_graph(self, output_html_path: str = "cpg_3d_knowledge_graph.html") -> str:
        """Genera e salva la visualizzazione 3D interattiva del grafo CPG costruito."""
        visualizer = Graph3DVisualizer(self.indexer.graph)
        return visualizer.export_html(output_html_path)

