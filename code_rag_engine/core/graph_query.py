"""
Modulo di interrogazione del Grafo di Conoscenza (Graph Query Engine).
Fornisce due modalità complementari:
1. Query Strutturali Dirette (Analytics & Traceability):
   - Conteggio aggregato per tipologia (funzioni, metodi, classi, moduli)
   - Ricerca e ispezione dei metadati di un simbolo
   - Tracciamento del flusso di chiamata (Call Graph / Execution Flow: chi chiama chi, upstream e downstream)
   - Analisi di impatto / dipendenze
2. Query in Linguaggio Naturale (Hybrid Graph RAG):
   - Interpreta domande generiche o specifiche
   - Recupera i nodi rilevanti da Qdrant e i relativi cammini nel grafo CPG
   - Sintetizza la risposta con il flusso architetturale
"""

from typing import Dict, Any, List, Optional
import networkx as nx
from .indexer import CodeGraphIndexer
from .retriever import HybridRetriever
from .reranker import CodeReranker
from .flow_trace import (
    FLOW_RELS, edge_rels, flow_direction, format_flow, has_flow_edges, has_flow_intent,
    pick_start_nodes, trace_flow,
)

# Parole di domanda che non sono nomi di simboli (evita di usarle come punto di partenza di un flusso)
_QUESTION_WORDS = {
    "cosa", "come", "dove", "quale", "quali", "succede", "accade", "dopo", "prima", "poi", "funziona",
    "flusso", "post", "pagina", "schermata", "funzione", "classe", "della", "dello", "nella", "sulla",
    "what", "happens", "after", "before", "does", "work", "flow", "page", "this", "that", "with", "from",
}

class KnowledgeGraphQueryEngine:
    def __init__(self, indexer: CodeGraphIndexer, reranker: Optional[CodeReranker] = None):
        self.indexer = indexer
        self.graph: nx.DiGraph = indexer.graph
        self.retriever = HybridRetriever(indexer)
        self.reranker = reranker or CodeReranker()

    # =========================================================================
    # 1. QUERY STRUTTURALI & ANALYTICS
    # =========================================================================

    def get_summary_statistics(self) -> Dict[str, Any]:
        """Restituisce il riepilogo quantitativo dei simboli e delle relazioni nel grafo."""
        stats = {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "counts_by_type": {},
            "relationships_by_type": {}
        }

        # Conteggio per tipologia di nodo (function, class, method, module)
        for _, data in self.graph.nodes(data=True):
            stype = data.get("symbol_type", "unknown")
            stats["counts_by_type"][stype] = stats["counts_by_type"].get(stype, 0) + 1

        # Conteggio per tipologia di arco (calls, contains, imports, e i flussi: renders, api_call, ...).
        # Un arco puo' portare piu' relazioni insieme: si conta ciascuna.
        for _, _, data in self.graph.edges(data=True):
            for rel in (edge_rels(data) or {"related"}):
                stats["relationships_by_type"][rel] = stats["relationships_by_type"].get(rel, 0) + 1

        return stats

    def find_symbol_nodes(self, symbol_name: str) -> List[str]:
        """Trova gli ID dei nodi corrispondenti a un nome di simbolo (anche parziale o case-insensitive)."""
        matches = []
        name_lower = symbol_name.lower()
        for nid, data in self.graph.nodes(data=True):
            sym = data.get("symbol_name", "").lower()
            if sym == name_lower or sym.endswith(f".{name_lower}"):
                matches.append(nid)
            elif name_lower in sym:
                matches.append(nid)
        return matches

    def trace_execution_flow(self, symbol_name: str, max_depth: int = 2) -> Dict[str, Any]:
        """
        Traccia il flusso di esecuzione e di gestione per un dato simbolo:
        - Downstream: chi viene chiamato da questa funzione (callees)
        - Upstream: chi chiama questa funzione (callers)
        - Gerarchia: classe genitrice o metodi contenuti (contains)
        """
        matches = self.find_symbol_nodes(symbol_name)
        if not matches:
            return {
                "found": False,
                "message": f"Simbolo '{symbol_name}' non trovato nel grafo di conoscenza."
            }

        target_node = matches[0]
        node_data = dict(self.graph.nodes[target_node])

        # Chi chiama questa funzione (Upstream / Callers)
        callers = []
        for pred in self.graph.predecessors(target_node):
            edge = self.graph.get_edge_data(pred, target_node)
            if "calls" in edge_rels(edge):
                callers.append({
                    "id": pred,
                    "symbol": self.graph.nodes[pred].get("symbol_name", pred),
                    "file": self.graph.nodes[pred].get("file_path", "")
                })

        # Chi viene chiamato da questa funzione (Downstream / Callees)
        callees = []
        for succ in self.graph.successors(target_node):
            edge = self.graph.get_edge_data(target_node, succ)
            if "calls" in edge_rels(edge):
                callees.append({
                    "id": succ,
                    "symbol": self.graph.nodes[succ].get("symbol_name", succ),
                    "file": self.graph.nodes[succ].get("file_path", "")
                })

        # Contesto di contenimento (es. classe di appartenenza)
        container = None
        for pred in self.graph.predecessors(target_node):
            edge = self.graph.get_edge_data(pred, target_node)
            if "contains" in edge_rels(edge):
                container = {
                    "id": pred,
                    "symbol": self.graph.nodes[pred].get("symbol_name", pred)
                }

        return {
            "found": True,
            "node_id": target_node,
            "symbol_name": node_data.get("symbol_name"),
            "symbol_type": node_data.get("symbol_type"),
            "file_path": node_data.get("file_path"),
            "lines": f"L{node_data.get('start_line')}-L{node_data.get('end_line')}",
            "docstring": node_data.get("docstring"),
            "container_class": container,
            "calls_made": node_data.get("calls", []),
            "callees_in_graph": callees,
            "callers_in_graph": callers,
            "dependencies": node_data.get("dependencies", []),
            "code_snippet": node_data.get("content")
        }

    # =========================================================================
    # 1b. FLUSSI: cosa succede PRIMA / DOPO un simbolo (UI, redirect, API, eventi)
    # =========================================================================

    def trace_flows(self, symbol_name: str, direction: str = "both", max_depth: int = 4,
                    max_steps: int = 40, max_starts: int = 2, exact_only: bool = False) -> Dict[str, Any]:
        """
        Segue gli archi di flusso (renders, navigates_to, redirects_to, api_call, handled_by, proxied_by,
        publishes, consumes + calls) a partire da un simbolo, in avanti ('after'), all'indietro ('before')
        o in entrambe le direzioni ('both'). Il punto di partenza e' scelto preferendo la UI
        (componente) agli endpoint e alle funzioni.
        """
        starts = pick_start_nodes(self.graph, symbol_name, limit=max_starts, exact_only=exact_only)
        if not starts:
            return {"found": False, "message": f"Simbolo '{symbol_name}' non trovato nel grafo di conoscenza."}
        dirs = ["before", "after"] if direction == "both" else [direction]
        sections = []
        for start in starts:
            for d in dirs:
                steps = trace_flow(self.graph, start, d, max_depth=max_depth, max_steps=max_steps)
                sections.append({
                    "start": start,
                    "symbol": self.graph.nodes[start].get("symbol_name", start),
                    "direction": d,
                    "steps": steps,
                    "text": format_flow(self.graph, steps, start),
                })
        return {"found": True, "sections": sections}

    def _answer_flow_question(self, question: str) -> Optional[Dict[str, Any]]:
        """
        Risponde a 'cosa succede dopo X', 'come si arriva a X', 'come funziona X' quando X ha archi di
        flusso. Restituisce None se la domanda non e' sui flussi o se X non ne ha (si ricade sugli altri intenti).
        """
        if not has_flow_intent(question):
            return None
        direction = flow_direction(question)
        words = [w.strip("?,.:;\"'()") for w in question.split()]
        words = [w for w in words if len(w) > 3 and w.lower() not in _QUESTION_WORDS]
        for exact_only in (True, False):   # prima i nomi esatti, poi le corrispondenze parziali
            for w in words:
                flow = self.trace_flows(w, direction, exact_only=exact_only)
                if not flow["found"]:
                    continue
                if not any(s["rel"] in FLOW_RELS for sec in flow["sections"] for s in sec["steps"]):
                    continue
                titles = {"before": "Cosa succede PRIMA (chi porta a)", "after": "Cosa succede DOPO"}
                parts = [f"### Flusso per '{w}' — archi tipizzati (renders, navigates_to, redirects_to, api_call, "
                         f"handled_by, publishes, calls)"]
                for sec in flow["sections"]:
                    body = sec["text"] if sec["steps"] else f"{sec['symbol']}\n  (nessun collegamento trovato)"
                    parts.append(f"**{titles[sec['direction']]} `{sec['symbol']}`**\n```\n{body}\n```")
                return {"type": "flow_answer", "answer": "\n\n".join(parts), "flow_details": flow}
        return None

    # =========================================================================
    # 2. QUERY IN LINGUAGGIO NATURALE (GRAPH-AUGMENTED ANSWERING)
    # =========================================================================

    def answer_question(self, user_question: str) -> Dict[str, Any]:
        """
        Risponde a domande sia quantitative che funzionali/architetturali.
        Riconosce automaticamente intenti statistici (es. 'quante funzioni...')
        oppure esegue il Graph RAG completo con flusso di chiamata.
        """
        q_lower = user_question.lower()

        # Intento 1: Domande statistiche, di panoramica o ricognizione (Overview / Global)
        overview_keywords = [
            "cosa è presente", "cosa contiene", "cosa c'è", "panoramica", "riassunto",
            "overview", "numero", "quante", "quanti", "conteggio", "quante funzioni",
            "totale", "elenco", "quali funzioni"
        ]
        if any(term in q_lower for term in overview_keywords):
            stats = self.get_summary_statistics()
            
            # Campiona i simboli principali per categoria
            sample_symbols = []
            for nid, data in list(self.graph.nodes(data=True))[:8]:
                sym = data.get("symbol_name")
                stype = data.get("symbol_type", "funzione")
                sample_symbols.append(f"`{sym}` ({stype})")

            summary_text = (
                f"### Panoramica del Grafo di Conoscenza (Repository Overview):\n"
                f"- **Totale Componenti Indicizzate**: {stats['total_nodes']} nodi logici\n"
                f"- **Ripartizione Simboli**:\n"
            )
            for k, v in stats["counts_by_type"].items():
                summary_text += f"  * {k.capitalize()}: {v}\n"

            summary_text += f"- **Relazioni CPG Mappate**: {stats['total_edges']} archi relazionali\n"
            for rel, count in stats["relationships_by_type"].items():
                summary_text += f"  * {rel}: {count}\n"

            if sample_symbols:
                summary_text += f"\n**Esempi di moduli/funzioni presenti**:\n{', '.join(sample_symbols)} ...\n"

            summary_text += (
                f"\n*Puoi chiedere il dettaglio di una funzione specifica (es. 'Come funziona has_close_elements?'), "
                f"il flusso delle chiamate o fare una ricerca concettuale.*"
            )

            return {
                "type": "statistical_answer",
                "answer": summary_text,
                "raw_stats": stats
            }

        # Intento 1b: flussi ("cosa succede dopo il login?", "come si arriva a X", "come funziona X")
        if has_flow_edges(self.graph):
            flow_answer = self._answer_flow_question(user_question)
            if flow_answer:
                return flow_answer

        # Intento 2: Ispezione specifica di un simbolo o del flusso
        # Estrai possibili simboli citati nella domanda
        words = [w.strip("?,.:;\"'") for w in user_question.split()]
        for w in words:
            if len(w) > 3 and self.find_symbol_nodes(w):
                flow = self.trace_execution_flow(w)
                if flow.get("found"):
                    answer_text = (
                        f"### Analisi del Flusso per '{flow['symbol_name']}' ({flow['symbol_type'].upper()}):\n"
                        f"- File: `{flow['file_path']}` ({flow['lines']})\n"
                        f"- Docstring: {flow['docstring'] or 'Non specificata'}\n"
                    )
                    if flow['container_class']:
                        answer_text += f"- Classe genitrice: `{flow['container_class']['symbol']}`\n"
                    
                    if flow['callees_in_graph']:
                        callee_names = [f"`{c['symbol']}`" for c in flow['callees_in_graph']]
                        answer_text += f"- Funzioni collegate chiamate direttamente: {', '.join(callee_names)}\n"
                    elif flow['calls_made']:
                        answer_text += f"- Chiamate interne rilevate nel codice: {', '.join(flow['calls_made'])}\n"
                    else:
                        answer_text += "- Nessuna chiamata esterna effettuata.\n"

                    if flow['callers_in_graph']:
                        caller_names = [f"`{c['symbol']}`" for c in flow['callers_in_graph']]
                        answer_text += f"- Chiamato upstream da: {', '.join(caller_names)}\n"

                    answer_text += f"\n```python\n{flow['code_snippet']}\n```"

                    return {
                        "type": "flow_trace_answer",
                        "answer": answer_text,
                        "flow_details": flow
                    }

        # Intento 3: Ricerca ibrida avanzata Dense + Sparse + CPG Rerank
        candidates = self.retriever.retrieve(user_question, top_k=5, expand_graph=True)
        reranked = self.reranker.rerank(user_question, candidates, top_n=2)

        if not reranked:
            return {
                "type": "retrieval_answer",
                "answer": "Nessun frammento rilevante trovato nel grafo di conoscenza per la domanda fornita.",
                "contexts": []
            }

        top = reranked[0]
        answer_text = (
            f"### Simbolo più rilevante: `{top['symbol']}` ({top.get('symbol_type', 'function')})\n"
            f"- File: `{top['file']}` (Linee {top['start_line']}-{top['end_line']})\n"
            f"- Punteggio Rerank: {top['rerank_score']:.4f}\n"
            f"- Docstring: {top.get('docstring') or 'N/A'}\n"
        )
        if top.get("graph_neighbors"):
            rels = [f"`{n['symbol_name']}` ({n.get('relation_type', '')})" for n in top['graph_neighbors']]
            answer_text += f"- Relazioni CPG correlate: {', '.join(rels)}\n"

        answer_text += f"\nImplementazione:\n```python\n{top['content']}\n```"

        return {
            "type": "retrieval_answer",
            "answer": answer_text,
            "contexts": reranked
        }

