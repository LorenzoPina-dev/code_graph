"""
Agente di domande sul Code Property Graph.

Data una domanda in linguaggio naturale individua la porzione di grafo che serve per
rispondere, così il visualizzatore 3D può evidenziarla:

1. SEED (nodi "rilevanti"):
   - simboli / file citati esplicitamente nella domanda (es. `validateUser`, `auth.ts`)
   - ricerca ibrida Qdrant (dense + sparse/RRF) + reranking Cross-Encoder
     (i file di test vengono ignorati, salvo che la domanda parli di test)
2. ESPANSIONE sul grafo (profondità 1, con tetto per nodo e totale):
   chiamati (callee), chiamanti (caller), classe/modulo contenitore, membri.
3. RISPOSTA:
   - con ANTHROPIC_API_KEY: un LLM legge il codice dei candidati, scrive la spiegazione
     e sceglie/ordina i simboli davvero utili (ordine di lettura consigliato)
   - senza LLM (o in caso di errore): riepilogo deterministico dei simboli e delle relazioni

Il risultato ha la forma:
    {"question", "mode": "llm"|"retrieval"|"overview"|"empty", "answer",
     "focus": [{"id", "role", "rank", "reason", "via"}], "stats": {...}}
dove "id" è l'identificativo del nodo nel grafo (stesso usato dal visualizzatore).
"""

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .indexer import CodeGraphIndexer
from .retriever import HybridRetriever
from .reranker import CodeReranker

_STOPWORDS = {
    "the", "and", "for", "how", "does", "what", "which", "where", "with", "this", "that",
    "from", "into", "are", "you", "can", "del", "della", "dello", "dei", "delle", "degli",
    "che", "come", "cosa", "quale", "quali", "dove", "per", "con", "una", "uno", "gli",
    "nel", "nella", "nei", "sono", "funziona", "funzionano", "viene", "vengono", "fanno",
    "fare", "questo", "questa", "codice", "funzione", "classe", "metodo", "modulo", "file",
    "quando", "chi", "perche", "tra", "fra", "sul", "sulla", "alla", "allo", "alle", "dal",
    "dalla", "dallo", "dalle",
}

_TEST_PATH_RE = re.compile(
    r"(^|/)(tests?|__tests__|specs?)(/|$)|\.(test|spec)\.[a-z0-9]+$|(^|/)test_[^/]*$", re.I
)
_FILE_TOKEN_RE = re.compile(
    r"[\w./\\-]+\.(?:py|ts|tsx|js|jsx|mjs|cjs|java|cs|go|rs|cpp|hpp|cc|cxx|ino|c|h)\b", re.I
)
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_OVERVIEW_RE = re.compile(
    r"\b(quante|quanti|panoramica|overview|riassunto|how many|conteggio|totale)\b", re.I
)
_NOISE_SYMBOLS = {"anonymous", "anonymousclass"}

_LLM_SYSTEM = (
    "Sei un assistente che aiuta a capire rapidamente una codebase. Ti vengono dati una domanda "
    "e dei simboli candidati estratti da un grafo del codice (con ruolo, file, righe e codice). "
    "Usa SOLO le informazioni fornite: se non bastano, dillo. Rispondi nella lingua della domanda, "
    "in modo conciso (al massimo ~8 frasi), citando i simboli per nome tra backtick. "
    "Nel campo `relevant` metti i NUMERI dei simboli necessari per capire la risposta, nell'ordine "
    "in cui conviene leggerli (dal punto d'ingresso al dettaglio), escludendo quelli non pertinenti. "
    "Rispondi ESCLUSIVAMENTE con un oggetto JSON: {\"answer\": \"...\", \"relevant\": [1, 3, 2]}"
)


@dataclass
class _Seed:
    node_id: str
    reason: str


class AnthropicSynthesizer:
    """Wrapper minimale sull'SDK Anthropic (opzionale). Legge ANTHROPIC_API_KEY dall'ambiente."""

    def __init__(self, model: Optional[str] = None, max_tokens: int = 1500):
        import anthropic  # import locale: la dipendenza resta opzionale

        self.client = anthropic.Anthropic()
        self.model = model or os.getenv("CODE_RAG_LLM_MODEL", "claude-sonnet-5-5")
        self.max_tokens = max_tokens

    def complete(self, system: str, user: str) -> str:
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")


def build_default_llm() -> Optional[AnthropicSynthesizer]:
    """Restituisce il sintetizzatore LLM se la chiave API è configurata, altrimenti None."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        return None
    try:
        return AnthropicSynthesizer()
    except Exception as e:  # SDK non installato o configurazione errata
        print(f"[!] LLM non disponibile ({e}). Uso solo ricerca ibrida + grafo.")
        return None


class GraphQuestionAgent:
    ROLE_ORDER = {"match": 0, "callee": 1, "caller": 1, "container": 2, "member": 3}

    def __init__(
        self,
        indexer: CodeGraphIndexer,
        reranker: Optional[CodeReranker] = None,
        llm: Optional[Any] = None,
        use_reranker: bool = True,
        top_k: int = 5,
        max_nodes: int = 30,
        per_seed_neighbors: int = 6,
        rerank_margin: float = 5.0,
        code_chars: int = 1500,
    ):
        self.indexer = indexer
        self.graph = indexer.graph
        self.retriever = HybridRetriever(indexer)
        self.reranker = reranker if reranker is not None else (CodeReranker() if use_reranker else None)
        self.llm = llm
        self.top_k = top_k
        self.max_nodes = max_nodes
        self.per_seed = per_seed_neighbors
        self.rerank_margin = rerank_margin   # scarta i seed con rerank_score < migliore - margine
        self.code_chars = code_chars
        self._by_name: Dict[str, List[str]] = {}
        self.refresh_index()

    # ------------------------------------------------------------------ utilità
    def refresh_index(self) -> None:
        """Indice nome-simbolo -> nodi, per riconoscere i simboli citati nella domanda."""
        self._by_name = {}
        for nid, data in self.graph.nodes(data=True):
            name = data.get("symbol_name") or ""
            if not name or name.lower() in _NOISE_SYMBOLS or data.get("symbol_type") == "module":
                continue
            keys = {name.lower()}
            for sep in (".", "::"):
                if sep in name:
                    keys.add(name.split(sep)[-1].lower())
            for k in keys:
                self._by_name.setdefault(k, []).append(nid)

    def warmup(self) -> None:
        """Carica i modelli (embedding e reranker) prima della prima domanda."""
        try:
            self.retriever.retrieve("warmup", top_k=1, expand_graph=False)
            if self.reranker is not None:
                _ = self.reranker.model
        except Exception as e:
            print(f"[!] Warm-up non riuscito ({e}): i modelli verranno caricati alla prima domanda.")

    @staticmethod
    def _is_test_path(path: Optional[str]) -> bool:
        return bool(_TEST_PATH_RE.search((path or "").replace("\\", "/")))

    def _name(self, nid: str) -> str:
        return self.graph.nodes[nid].get("symbol_name") or nid

    def _is_noise(self, nid: str, include_tests: bool) -> bool:
        data = self.graph.nodes[nid]
        if (data.get("symbol_name") or "").lower() in _NOISE_SYMBOLS:
            return True
        return not include_tests and self._is_test_path(data.get("file_path"))

    def _relation(self, u: str, v: str) -> Optional[str]:
        return (self.graph.get_edge_data(u, v) or {}).get("relation")

    # ------------------------------------------------------------------ seed
    def _explicit_seeds(self, question: str, include_tests: bool) -> List[_Seed]:
        seeds: List[_Seed] = []
        seen = set()

        # a) file citati (es. "validation.ts", "src/auth.py")
        for tok in _FILE_TOKEN_RE.findall(question):
            t = tok.replace("\\", "/").lower()
            in_file = [
                (data.get("start_line") or 0, nid)
                for nid, data in self.graph.nodes(data=True)
                if (data.get("file_path") or "").replace("\\", "/").lower().endswith(t)
                and (data.get("symbol_name") or "").lower() not in _NOISE_SYMBOLS
            ]
            for _, nid in sorted(in_file)[:8]:
                if nid not in seen:
                    seen.add(nid)
                    seeds.append(_Seed(nid, f"file citato nella domanda: {tok}"))

        # b) simboli citati per nome (si ignorano nomi troppo ambigui)
        for tok in _IDENT_RE.findall(question):
            low = tok.lower()
            if len(tok) < 3 or low in _STOPWORDS:
                continue
            ids = self._by_name.get(low) or []
            if not ids or len(ids) > 4:
                continue
            for nid in ids:
                if nid not in seen and not self._is_noise(nid, include_tests):
                    seen.add(nid)
                    seeds.append(_Seed(nid, "simbolo citato nella domanda"))

        return seeds[: self.top_k]

    def _semantic_seeds(self, question: str, include_tests: bool) -> List[_Seed]:
        candidates = self.retriever.retrieve(question, top_k=self.top_k * 4, expand_graph=False)
        candidates = [c for c in candidates if c.get("id") in self.graph]
        if not include_tests:
            filtered = [c for c in candidates if not self._is_test_path(c.get("file"))]
            candidates = filtered or candidates
        if not candidates:
            return []

        ranked = candidates[: self.top_k]
        if self.reranker is not None and len(candidates) > 1:
            try:
                ranked = self.reranker.rerank(question, candidates, top_n=self.top_k)
                if ranked and "rerank_score" in ranked[0]:
                    best = ranked[0]["rerank_score"]
                    ranked = [r for r in ranked if r["rerank_score"] >= best - self.rerank_margin]
            except Exception as e:
                print(f"[!] Reranker non disponibile ({e}): uso l'ordinamento ibrido.")
                ranked = candidates[: self.top_k]
        return [_Seed(c["id"], "affinità semantica con la domanda") for c in ranked]

    # ------------------------------------------------------------------ espansione
    def _expand(self, seeds: List[_Seed], include_tests: bool) -> Dict[str, Dict[str, Any]]:
        nodes: Dict[str, Dict[str, Any]] = {}
        for rank, s in enumerate(seeds, 1):
            nodes[s.node_id] = {"id": s.node_id, "role": "match", "rank": rank,
                                "reason": s.reason, "via": None}

        degree = self.graph.degree
        for s in seeds:
            nid = s.node_id
            succ = list(self.graph.successors(nid))
            pred = list(self.graph.predecessors(nid))
            plan = [
                ("callee", [v for v in succ if self._relation(nid, v) == "calls"], "chiamato da «{n}»"),
                ("caller", [u for u in pred if self._relation(u, nid) == "calls"], "chiama «{n}»"),
                ("container", [u for u in pred if self._relation(u, nid) == "contains"], "contiene «{n}»"),
                ("member", [v for v in succ if self._relation(nid, v) == "contains"], "definito in «{n}»"),
            ]
            for role, ids, template in plan:
                ids = [i for i in ids if i in self.graph and not self._is_noise(i, include_tests)]
                ids.sort(key=lambda i: degree(i), reverse=True)
                added = 0
                for other in ids:
                    if other in nodes:
                        continue
                    if len(nodes) >= self.max_nodes or added >= self.per_seed:
                        break
                    nodes[other] = {"id": other, "role": role, "rank": None,
                                    "reason": template.format(n=self._name(nid)), "via": nid}
                    added += 1
        return nodes

    # ------------------------------------------------------------------ risposte
    def _describe(self, nodes: Dict[str, Dict[str, Any]]) -> str:
        seeds = sorted((n for n in nodes.values() if n["role"] == "match"),
                       key=lambda n: n.get("rank") or 0)
        if not seeds:
            return "Non ho trovato simboli rilevanti per questa domanda nel grafo."
        lines = [f"Ho evidenziato {len(nodes)} simboli collegati alla domanda. I più pertinenti:"]
        for n in seeds:
            d = self.graph.nodes[n["id"]]
            doc = (d.get("docstring") or "").strip().splitlines()
            doc_txt = f" — {doc[0][:120]}" if doc and doc[0] else ""
            lines.append(f"{n['rank']}. `{self._name(n['id'])}` ({d.get('symbol_type')}) — "
                         f"{d.get('file_path')}:{d.get('start_line')}-{d.get('end_line')}{doc_txt}")
        for n in seeds[:3]:
            callees = [self._name(m["id"]) for m in nodes.values()
                       if m["via"] == n["id"] and m["role"] == "callee"]
            callers = [self._name(m["id"]) for m in nodes.values()
                       if m["via"] == n["id"] and m["role"] == "caller"]
            parts = []
            if callees:
                parts.append("chiama " + ", ".join(f"`{c}`" for c in callees))
            if callers:
                parts.append("è chiamato da " + ", ".join(f"`{c}`" for c in callers))
            if parts:
                lines.append(f"`{self._name(n['id'])}` " + "; ".join(parts) + ".")
        return "\n".join(lines)

    @staticmethod
    def _parse_json(raw: str) -> Optional[Dict[str, Any]]:
        raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
        start, end = raw.find("{"), raw.rfind("}")
        if start == -1 or end <= start:
            return None
        try:
            data = json.loads(raw[start:end + 1])
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            return None

    def _refine_with_llm(self, question: str, nodes: Dict[str, Dict[str, Any]]):
        ordered = sorted(nodes.values(),
                         key=lambda n: (self.ROLE_ORDER[n["role"]], n.get("rank") or 999))
        items = []
        for i, n in enumerate(ordered, 1):
            d = self.graph.nodes[n["id"]]
            limit = self.code_chars if n["role"] == "match" else 400
            items.append(
                f"[{i}] {d.get('symbol_name')} ({d.get('symbol_type')}) — "
                f"{d.get('file_path')}:{d.get('start_line')}-{d.get('end_line')} — "
                f"ruolo: {n['role']} ({n['reason']})\n"
                f"{(d.get('docstring') or '')[:200]}\n{(d.get('content') or '')[:limit]}"
            )
        user = f"Domanda: {question}\n\nSimboli candidati:\n\n" + "\n\n".join(items)
        raw = self.llm.complete(_LLM_SYSTEM, user)

        data = self._parse_json(raw)
        if not data:
            return raw.strip(), nodes   # testo libero: tengo tutti i nodi
        answer = str(data.get("answer", "")).strip() or raw.strip()

        picked: List[int] = []
        for x in data.get("relevant", []) or []:
            if str(x).isdigit() and 1 <= int(x) <= len(ordered) and int(x) not in picked:
                picked.append(int(x))
        if not picked:
            return answer, nodes
        refined = {}
        for pos, idx in enumerate(picked, 1):
            n = dict(ordered[idx - 1])
            n["rank"] = pos   # ordine di lettura suggerito
            refined[n["id"]] = n
        return answer, refined

    def _overview(self) -> Dict[str, Any]:
        counts: Dict[str, int] = {}
        for _, d in self.graph.nodes(data=True):
            t = d.get("symbol_type", "unknown")
            counts[t] = counts.get(t, 0) + 1
        rels: Dict[str, int] = {}
        for _, _, d in self.graph.edges(data=True):
            r = d.get("relation", "related")
            rels[r] = rels.get(r, 0) + 1
        text = (f"Il grafo contiene {self.graph.number_of_nodes()} nodi "
                f"({', '.join(f'{k}: {v}' for k, v in sorted(counts.items()))}) e "
                f"{self.graph.number_of_edges()} archi "
                f"({', '.join(f'{k}: {v}' for k, v in sorted(rels.items()))}). "
                f"Per vedere una parte del grafo, chiedi di una funzionalità o cita un simbolo.")
        return self._result("", "overview", text, {}, {"counts": counts, "relations": rels})

    def _result(self, question, mode, answer, nodes, stats=None) -> Dict[str, Any]:
        focus = sorted(
            nodes.values(),
            key=lambda n: (n.get("rank") is None, n.get("rank") or 0, self.ROLE_ORDER[n["role"]]),
        )
        out_stats = {"nodes": len(focus), "matches": sum(1 for n in focus if n["role"] == "match")}
        out_stats.update(stats or {})
        return {
            "question": question,
            "mode": mode,
            "answer": answer,
            "focus": [{"id": n["id"], "role": n["role"], "rank": n.get("rank"),
                       "reason": n.get("reason"), "via": n.get("via")} for n in focus],
            "stats": out_stats,
        }

    # ------------------------------------------------------------------ API pubblica
    def ask(self, question: str) -> Dict[str, Any]:
        question = (question or "").strip()
        if not question:
            return self._result(question, "empty", "Scrivi una domanda sul codice.", {})

        include_tests = "test" in question.lower()
        seeds = self._explicit_seeds(question, include_tests)

        if not seeds and _OVERVIEW_RE.search(question):
            res = self._overview()
            res["question"] = question
            return res

        seen = {s.node_id for s in seeds}
        for s in self._semantic_seeds(question, include_tests):
            if len(seeds) >= self.top_k:
                break
            if s.node_id not in seen:
                seen.add(s.node_id)
                seeds.append(s)

        if not seeds:
            return self._result(question, "retrieval",
                                "Non ho trovato nulla di rilevante nel grafo per questa domanda.", {})

        nodes = self._expand(seeds, include_tests)

        if self.llm is not None:
            try:
                answer, refined = self._refine_with_llm(question, nodes)
                return self._result(question, "llm", answer, refined)
            except Exception as e:
                print(f"[!] Sintesi LLM fallita ({e}): uso il riepilogo deterministico.")

        return self._result(question, "retrieval", self._describe(nodes), nodes)
