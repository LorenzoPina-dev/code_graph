"""
Script CLI interattivo per fare domande al Grafo di Conoscenza (CPG) e al Vector Store.
Permette di:
1. Chiedere statistiche aggregate (es. 'quante funzioni ci sono?')
2. Tracciare il flusso e la gestione di una funzione specifica (es. 'come viene gestita authenticate?')
3. Porre qualsiasi domanda in linguaggio naturale con ricerca ibrida e reranking
"""

import sys
from core.indexer import CodeGraphIndexer
from core.dataset_ingestion import DatasetIngestionPipeline
from core.graph_query import KnowledgeGraphQueryEngine

def main():
    print("=" * 75)
    print("    KNOWLEDGE GRAPH & CODE RAG QUERY INTERFACE (ZERO MOCK)")
    print("=" * 75)

    # 1. Inizializzazione e caricamento rapido dell'indice
    indexer = CodeGraphIndexer(collection_name="interactive_kg_query", location=":memory:")
    pipeline = DatasetIngestionPipeline(indexer=indexer)

    # Se passato un argomento limite (default 25 task per avvio rapido in CLI, o 'all' per 164)
    sample_limit = 25
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        sample_limit = int(sys.argv[1])
    elif len(sys.argv) > 1 and sys.argv[1].lower() == "all":
        sample_limit = None

    print(f"\n[*] Caricamento ed indicizzazione dataset ({sample_limit if sample_limit else 'TUTTI i 164'} task)...")
    pipeline.ingest_humaneval(split="test", max_samples=sample_limit)

    query_engine = KnowledgeGraphQueryEngine(indexer=indexer)
    stats = query_engine.get_summary_statistics()

    print(f"\n[+] Grafo di Conoscenza Pronto!")
    print(f"    - Nodi indicizzati : {stats['total_nodes']} ({stats['counts_by_type']})")
    print(f"    - Relazioni        : {stats['total_edges']} ({stats['relationships_by_type']})")

    print("\n" + "-" * 75)
    print("Puoi fare domande come:")
    print("  1. 'Quante funzioni ci sono nel grafo?'")
    print("  2. 'Come viene gestita la funzione has_close_elements?' (o qualsiasi simbolo)")
    print("  3. 'Quale funzione filtra i numeri negativi?'")
    print("  (Scrivi 'esci' o 'exit' per terminare)")
    print("-" * 75 + "\n")

    # Se è stata fornita una domanda come argomento CLI
    if len(sys.argv) > 2:
        cli_query = " ".join(sys.argv[2:])
        print(f"Domanda: {cli_query}\n")
        res = query_engine.answer_question(cli_query)
        print(res["answer"])
        return

    # Modalità interattiva REPL
    while True:
        try:
            query = input("Domanda al Grafo > ").strip()
            if not query:
                continue
            if query.lower() in ["esci", "exit", "quit", "q"]:
                print("Chiusura interfaccia.")
                break

            result = query_engine.answer_question(query)
            print("\n" + result["answer"] + "\n" + "-" * 75 + "\n")

        except (KeyboardInterrupt, EOFError):
            print("\nChiusura sessione.")
            break

if __name__ == "__main__":
    main()

