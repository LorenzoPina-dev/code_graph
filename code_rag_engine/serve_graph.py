"""
Server locale: visualizzatore 3D del Code Property Graph + agente che, quando fai una
domanda, evidenzia nel grafo la parte rilevante per rispondere.

Esempi:
  python serve_graph.py https://github.com/LorenzoPina-dev/Gestione_dispensa
  python serve_graph.py D:/mio_progetto --port 8800
  python serve_graph.py ./core --no-llm

Spiegazioni discorsive con LLM (opzionale): imposta ANTHROPIC_API_KEY (e, se vuoi,
CODE_RAG_LLM_MODEL). Senza chiave l'agente usa solo ricerca ibrida + grafo.

Il server ascolta solo su 127.0.0.1: non ha autenticazione e espone il codice indicizzato.
"""

import argparse
import json
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from core.agent import GraphQuestionAgent, build_default_llm
from core.indexer import CodeGraphIndexer
from core.universal_ingestion import UniversalIngestionPipeline
from core.visualizer import Graph3DVisualizer

MAX_BODY_BYTES = 64 * 1024


def make_handler(html: str, agent: GraphQuestionAgent):
    lock = threading.Lock()   # il client Qdrant in memoria e i modelli non sono pensati per l'accesso concorrente
    page = html.encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, payload: dict) -> None:
            self._send(code, json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                       "application/json; charset=utf-8")

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(200, page, "text/html; charset=utf-8")
            elif self.path == "/health":
                self._json(200, {"status": "ok"})
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):
            try:
                length = int(self.headers.get("Content-Length", "0"))
                # leggo sempre il corpo (se di dimensione ragionevole) prima di rispondere,
                # altrimenti su alcuni sistemi (Windows) la connessione puo' essere resettata
                raw = self.rfile.read(length) if 0 < length <= MAX_BODY_BYTES else b""
                if self.path != "/api/ask":
                    self._json(404, {"error": "not found"})
                    return
                if length <= 0 or length > MAX_BODY_BYTES:
                    self._json(400, {"error": "corpo della richiesta mancante o troppo grande"})
                    return
                body = json.loads(raw.decode("utf-8"))
                if not isinstance(body, dict):
                    self._json(400, {"error": "il corpo deve essere un oggetto JSON"})
                    return
                question = str(body.get("question", "")).strip()
                if not question:
                    self._json(400, {"error": "domanda vuota"})
                    return
                with lock:
                    result = agent.ask(question)
                self._json(200, result)
            except (json.JSONDecodeError, UnicodeDecodeError):
                self._json(400, {"error": "JSON non valido"})
            except Exception as e:   # non far cadere il server per un errore su una singola domanda
                print(f"[!] Errore su /api/ask: {e!r}")
                self._json(500, {"error": str(e)})

        def log_message(self, fmt, *args):
            pass   # silenzia il log per richiesta

    return Handler


def main():
    parser = argparse.ArgumentParser(
        description="Ingerisce una sorgente e serve il grafo 3D con l'agente di domande."
    )
    parser.add_argument("source", help="URL GitHub, cartella locale o nome dataset")
    parser.add_argument("--limit", type=int, default=None, help="Max campioni se la sorgente è un dataset")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-open", action="store_true", help="Non aprire il browser")
    parser.add_argument("--no-llm", action="store_true", help="Non usare l'LLM anche se la chiave è presente")
    parser.add_argument("--no-rerank", action="store_true", help="Salta il reranker (più veloce, meno preciso)")
    args = parser.parse_args()

    print("=" * 75)
    print("        CODE GRAPH 3D + AGENTE DI DOMANDE")
    print("=" * 75)
    print(f"[*] Sorgente: {args.source}")

    indexer = CodeGraphIndexer(collection_name="serve_graph_cpg", location=":memory:")
    pipeline = UniversalIngestionPipeline(indexer=indexer)
    try:
        pipeline.ingest(source=args.source, max_samples=args.limit)
    except Exception as e:
        print(f"[!] Errore durante l'ingestione: {e}")
        sys.exit(1)
    print(f"[+] Grafo: {indexer.graph.number_of_nodes()} nodi, {indexer.graph.number_of_edges()} archi, "
          f"{indexer.client.count(collection_name=indexer.collection_name).count} punti vettoriali")

    llm = None if args.no_llm else build_default_llm()
    print(f"[*] Sintesi LLM: {'attiva (' + llm.model + ')' if llm else 'non attiva (solo ricerca ibrida + grafo)'}")

    agent = GraphQuestionAgent(indexer, llm=llm, use_reranker=not args.no_rerank)
    print("[*] Caricamento modelli di ricerca (la prima volta può richiedere un po')...")
    agent.warmup()

    html = Graph3DVisualizer(indexer.graph).render_html(agent_endpoint="/api/ask")

    server = None
    port = args.port
    for candidate in range(args.port, args.port + 10):
        try:
            server = ThreadingHTTPServer(("127.0.0.1", candidate), make_handler(html, agent))
            port = candidate
            break
        except OSError:
            continue
    if server is None:
        print(f"[!] Nessuna porta libera tra {args.port} e {args.port + 9}.")
        sys.exit(1)

    url = f"http://127.0.0.1:{port}/"
    print(f"\n[+] Pronto: {url}   (Ctrl+C per terminare)")
    if not args.no_open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nChiusura server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
