"""
Attraversamento dei FLUSSI nel Code Property Graph.

Il linker (flow_extractor.FlowLinker) aggiunge al grafo archi tipizzati tra simboli, schermate,
endpoint ed eventi. Questo modulo li legge per rispondere a domande come
"cosa succede dopo il login?", "come si arriva a questa pagina?", "come funziona il login?".

E' volutamente senza dipendenze (solo il grafo NetworkX), cosi' puo' essere usato sia dal motore di
query (graph_query.py) sia dall'agente (agent.py) e testato senza Qdrant ne' Tree-sitter.

Note sul modello dati
---------------------
- Tra due nodi puo' esserci UN solo arco NetworkX che porta piu' relazioni:
  `relations` = lista (es. ["calls", "renders"]), `relation` = la prima, `meta[rel]` = dettagli
  (riga, `via`, `value`, `method`, `path`...). Per questo si usa sempre `edge_rels()`.
- Gli archi di flusso partono dal simbolo piu' interno che contiene la riga: il chunker non crea nodi
  per le funzioni annidate, quindi un handler dentro un componente e' attribuito al componente.
"""

import re
from typing import Any, Dict, List, Optional, Set

import networkx as nx

# Deve restare allineato a flow_extractor.FLOW_RELATIONS (non importato qui per non richiedere Tree-sitter).
FLOW_RELS = frozenset({
    "renders", "navigates_to", "redirects_to", "api_call",
    "proxied_by", "handled_by", "publishes", "consumes",
})
FOLLOW = FLOW_RELS | {"calls"}

# Ordine di presentazione degli archi uscenti da uno stesso nodo. Le righe di codice non sono
# confrontabili tra file diversi (un `navigates_to` ha la riga del setter, che sta in un altro file
# rispetto a un `api_call`), quindi si ordina prima per tipo: di norma la UI naviga DOPO la chiamata API.
REL_ORDER = {
    "calls": 0, "api_call": 1, "publishes": 2, "handled_by": 3, "proxied_by": 3,
    "consumes": 3, "renders": 4, "navigates_to": 5, "redirects_to": 5,
}
DETAIL_KEYS = ("value", "via", "method", "path", "target", "upstream", "prefix")

_BEFORE_RE = re.compile(
    r"\b(prima|precede\w*|da dove|chi (?:porta|chiama|arriva|rimanda|apre|usa)|come si arriva|"
    r"arriva(?:no)? a|before|upstream|who (?:calls|opens|redirects|uses)|how (?:do|does) .{0,30}(?:get|reach))\b",
    re.I,
)
_AFTER_RE = re.compile(
    r"\b(dopo|poi|segue|successiv\w*|succede|accade|avviene|redirect\w*|porta a|va a|"
    r"then|after|next|happens|downstream)\b|\bpost\b(?!\s*/)",
    re.I,
)
_BOTH_RE = re.compile(
    r"\b(come funziona|flusso|workflow|end[- ]to[- ]end|completo|intero|how (?:does|do) .{0,40}work|flow)\b", re.I
)


# ----------------------------------------------------------------------------- utilita'
def edge_rels(data: Optional[Dict[str, Any]]) -> Set[str]:
    """Insieme delle relazioni portate da un arco (gestisce gli archi con piu' relazioni)."""
    data = data or {}
    rels = data.get("relations") or [data.get("relation")]
    return {r for r in rels if r}


def has_rel(graph: nx.DiGraph, u: str, v: str, rel: str) -> bool:
    return rel in edge_rels(graph.get_edge_data(u, v))


def has_flow_edges(graph: nx.DiGraph) -> bool:
    return any(edge_rels(d) & FLOW_RELS for _, _, d in graph.edges(data=True))


def flow_direction(question: str) -> str:
    """'after' | 'before' | 'both', in base alla formulazione della domanda (default: both)."""
    before, after = bool(_BEFORE_RE.search(question)), bool(_AFTER_RE.search(question))
    if _BOTH_RE.search(question) or before == after:
        return "both"
    return "before" if before else "after"


def has_flow_intent(question: str) -> bool:
    """True se la domanda chiede di un flusso (prima / dopo / come funziona / workflow ...)."""
    return bool(_BEFORE_RE.search(question) or _AFTER_RE.search(question) or _BOTH_RE.search(question))


def descendants(graph: nx.DiGraph, nid: str) -> Set[str]:
    """Nodi contenuti (via `contains`): i metodi di una classe, ecc."""
    seen: Set[str] = set()
    stack = [nid]
    while stack:
        cur = stack.pop()
        for _, v, d in graph.out_edges(cur, data=True):
            if "contains" in edge_rels(d) and v not in seen:
                seen.add(v)
                stack.append(v)
    return seen


def label(graph: nx.DiGraph, nid: str) -> str:
    return graph.nodes[nid].get("symbol_name") or nid


# ----------------------------------------------------------------------------- attraversamento
def trace_flow(
    graph: nx.DiGraph,
    start: str,
    direction: str = "after",
    max_depth: int = 4,
    max_steps: int = 40,
    max_calls_per_node: int = 5,
    hub_degree: int = 15,
) -> List[Dict[str, Any]]:
    """
    Visita in profondita' (pre-ordine, quindi leggibile come albero) i flussi che partono da `start`
    (direction='after') o che arrivano a `start` (direction='before').

    Ogni passo: {"src", "dst", "rel", "meta", "depth", "seen"}. `seen=True` indica un nodo gia'
    visitato (ciclo, es. Login <-> Register): l'arco e' riportato ma non si riscende.
    Le `calls` sono limitate (max per nodo, niente utility con fan-in alto) per non sommergere il flusso.
    """
    if start not in graph:
        return []
    steps: List[Dict[str, Any]] = []
    visited = {start}

    def neighbours(node: str):
        if direction == "after":
            # il nodo + cio' che contiene: le azioni stanno spesso nei metodi/handler interni
            scope = [node, *descendants(graph, node)]
            raw = [(s, v, d) for s in scope for _, v, d in graph.out_edges(s, data=True)]
        else:
            raw = [(u, node, d) for u, _, d in graph.in_edges(node, data=True)]
        out = []
        for u, v, d in raw:
            for rel in edge_rels(d) & FOLLOW:
                out.append((u, v, rel, (d.get("meta") or {}).get(rel, {})))
        out.sort(key=lambda e: (REL_ORDER.get(e[2], 9), e[3].get("line", 0)))
        return out

    def walk(node: str, depth: int) -> None:
        calls = 0
        for u, v, rel, meta in neighbours(node):
            if len(steps) >= max_steps:
                return
            nxt = v if direction == "after" else u
            if nxt == node:
                continue
            if rel == "calls":
                if calls >= max_calls_per_node or graph.in_degree(nxt) > hub_degree:
                    continue
                calls += 1
            seen = nxt in visited
            steps.append({"src": u, "dst": v, "rel": rel, "meta": meta, "depth": depth, "seen": seen})
            if not seen:
                visited.add(nxt)
                if depth + 1 < max_depth:
                    walk(nxt, depth + 1)

    walk(start, 0)
    return steps


def other_end(step: Dict[str, Any], direction: str) -> str:
    """Il nodo 'nuovo' di un passo: la destinazione andando avanti, la sorgente andando indietro."""
    return step["dst"] if direction == "after" else step["src"]


def step_detail(step: Dict[str, Any]) -> str:
    meta = step.get("meta") or {}
    return ", ".join(f"{k}={meta[k]}" for k in DETAIL_KEYS if meta.get(k) not in (None, ""))


def format_flow(graph: nx.DiGraph, steps: List[Dict[str, Any]], start: Optional[str] = None) -> str:
    """Rappresentazione testuale ad albero, adatta a essere letta da una persona o da un LLM."""
    lines: List[str] = [label(graph, start)] if start else []
    for s in steps:
        detail = step_detail(s)
        lines.append(
            f"{'  ' * (s['depth'] + 1)}{label(graph, s['src'])} "
            f"--{s['rel']}{f' [{detail}]' if detail else ''}--> {label(graph, s['dst'])}"
            f"{'  (già visto)' if s['seen'] else ''}"
        )
    return "\n".join(lines)


def pick_start_nodes(graph: nx.DiGraph, name: str, limit: int = 2, exact_only: bool = False) -> List[str]:
    """
    Nodi che corrispondono a `name`, in ordine di utilita' per tracciare un flusso:
    prima le corrispondenze esatte, poi UI > endpoint/schermate > funzioni > resto.
    Con `exact_only` le corrispondenze parziali (sottostringa) vengono ignorate.
    """
    low = name.lower()
    rank = {"component": 0, "screen": 1, "endpoint": 1, "function": 2, "method": 2, "class": 3}
    exact, partial = [], []
    for nid, d in graph.nodes(data=True):
        sym = (d.get("symbol_name") or "").lower()
        if not sym or d.get("symbol_type") == "module":
            continue
        base = sym.split(".")[-1].split("/")[-1].split(" ")[-1]
        if sym == low or base == low:
            exact.append(nid)
        elif low in sym:
            partial.append(nid)

    def touches_flow(nid: str) -> bool:
        edges = [d for _, _, d in graph.out_edges(nid, data=True)] + [d for _, _, d in graph.in_edges(nid, data=True)]
        return any(edge_rels(d) & FLOW_RELS for d in edges)

    def key(nid: str):
        d = graph.nodes[nid]
        return (not touches_flow(nid), rank.get(d.get("symbol_type"), 4), len(d.get("symbol_name") or ""))

    pool = exact or ([] if exact_only else partial)
    return sorted(pool, key=key)[:limit]
