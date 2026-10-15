"""
Script eseguibile per:
1. Ingerire l'intero dataset OpenAI HumanEval (164 problemi reali)
2. Costruire l'intero Code Property Graph (CPG) con nodi e archi semantici
3. Popolare Qdrant con vettori densi e sparsi
4. Esportare e aprire il visualizzatore 3D interattivo
"""

import sys
import webbrowser
from pathlib import Path
from core.indexer import CodeGraphIndexer
from core.dataset_ingestion import DatasetIngestionPipeline

def main():
    print("=" * 75)
    print("   DATASET FULL INGESTION & 3D CODE PROPERTY GRAPH EXPLORER   ")
    print("=" * 75)

    # Inizializza l'indexer di produzione Qdrant in memoria
    indexer = CodeGraphIndexer(collection_name="humaneval_full_knowledge_graph", location=":memory:")
    pipeline = DatasetIngestionPipeline(indexer=indexer)

    # Esegui ingestione completa di tutto il dataset
    stats = pipeline.ingest_humaneval(split="test")

    print("\n[+] Statistiche Grafo di Conoscenza Costruito:")
    print(f"  * Nodi CPG estratti (Funzioni, Classi, Moduli) : {stats['cpg_nodes']}")
    print(f"  * Archi CPG (Calls, Contains, Dependencies)    : {stats['cpg_edges']}")
    print(f"  * Punti indicizzati in Qdrant (Dense+Sparse)   : {stats['qdrant_points']}")

    # Genera la visualizzazione 3D interattiva
    output_html = "cpg_3d_knowledge_graph.html"
    saved_path = pipeline.export_3d_graph(output_html)
    print(f"\n[+] Visualizzatore 3D generato con successo: {saved_path}")

    print("\n[*] Per esplorare il grafo 3D interattivo, apri il file:")
    print(f"    file:///{saved_path.replace(chr(92), '/')}")

    # Apri automaticamente nel browser predefinito se richiesto
    if "--open" in sys.argv:
        webbrowser.open(Path(saved_path).as_uri())

if __name__ == "__main__":
    main()

