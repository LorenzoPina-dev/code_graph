"""
CLI Universale: Ingerisce da URL GitHub, Cartella Locale o Nome Dataset
e apre automaticamente il visualizzatore 3D interattivo del grafo.

Esempi di utilizzo:
  1. Repository GitHub:
     python ingest_source.py https://github.com/psf/requests.git
  
  2. Percorso Locale:
     python ingest_source.py ./core
     python ingest_source.py D:/mio_progetto

  3. Dataset da libreria (Hugging Face):
     python ingest_source.py openai/openai_humaneval --limit 20
     python ingest_source.py humaneval
"""

import sys
import argparse
import webbrowser
from pathlib import Path
from core.indexer import CodeGraphIndexer
from core.universal_ingestion import UniversalIngestionPipeline

def main():
    parser = argparse.ArgumentParser(
        description="Ingerisce qualsiasi sorgente (GitHub, Path locale o Dataset) e genera il visualizzatore 3D."
    )
    parser.add_argument(
        "source",
        type=str,
        help="URL GitHub (es. https://github.com/...), percorso cartella locale, o nome dataset (es. 'openai/openai_humaneval')"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Numero massimo di task/campioni da processare se la sorgente è un dataset (es. --limit 30)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="cpg_3d_knowledge_graph.html",
        help="Nome del file HTML 3D di output (default: cpg_3d_knowledge_graph.html)"
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Non aprire automaticamente il browser al termine"
    )

    args = parser.parse_args()

    print("=" * 75)
    print("      UNIVERSAL INGESTION & 3D CODE PROPERTY GRAPH EXPLORER      ")
    print("=" * 75)
    print(f"[*] Sorgente indicata : {args.source}")

    # Inizializza indexer e pipeline universale
    indexer = CodeGraphIndexer(collection_name="universal_code_rag_cpg", location=":memory:")
    pipeline = UniversalIngestionPipeline(indexer=indexer)

    # Ingestione con auto-routing
    try:
        stats = pipeline.ingest(source=args.source, max_samples=args.limit)
    except Exception as e:
        print(f"\n[!] Errore durante l'ingestione della sorgente: {e}")
        sys.exit(1)

    print("\n" + "=" * 75)
    print("                     INGESTIONE COMPLETATA                       ")
    print("=" * 75)
    print(f" * Tipo Sorgente Rilevato : {stats.get('source_type')}")
    print(f" * Nodi CPG nel Grafo     : {stats.get('cpg_nodes')}")
    print(f" * Archi CPG Relazionali  : {stats.get('cpg_edges')}")
    print(f" * Punti Vettoriali Qdrant: {stats.get('qdrant_points')}")
    print("=" * 75)

    # Esportazione 3D HTML
    html_path = pipeline.export_and_view_3d(args.output)
    print(f"\n[+] Visualizzatore 3D generato: {html_path}")
    print(f"[*] File pronto per la navigazione: file:///{html_path.replace(chr(92), '/')}")

    # Apertura nel browser
    if not args.no_open:
        print("[*] Apertura visualizzatore 3D nel browser predefinito...")
        webbrowser.open(Path(html_path).as_uri())

if __name__ == "__main__":
    main()

