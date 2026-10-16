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
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .indexer import CodeGraphIndexer
from .retriever import HybridRetriever
from .reranker import CodeReranker
from .flow_trace import (
    FLOW_RELS, edge_rels, flow_direction, format_flow, has_flow_edges, has_flow_intent,
    has_rel, label as flow_label, other_end, step_detail, trace_flow,
)

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
_CODE_REQUEST_RE = re.compile(
    r"\b(codice|implementa|implementare|query|sql|funzione|metodo|classe|"
    r"registrare|registrazione|account|utente|login|auth|endpoint|api|"
    r"create|register|signup|sign[\s_-]?up)\b", re.I
)
_DOMAIN_TERMS = {
    "ocr": {"ocr", "receipt", "paper", "scan", "scanner", "document", "enhance", "recognizer"},
    "auth": {"auth", "login", "logout", "account", "utente", "user", "register", "signup",
             "password", "token", "identity", "keycloak"},
    "data": {"data", "dataset", "query", "sql", "database", "postgres", "repository", "table"},
}

_LLM_SYSTEM = (
    "Sei un assistente che aiuta a capire rapidamente una codebase. Ti vengono dati una domanda "
    "e dei simboli candidati estratti da un grafo del codice (con ruolo, file, righe e codice). "
    "Usa SOLO le informazioni fornite: se non bastano, dillo. Rispondi nella lingua della domanda, "
    "in modo conciso (al massimo ~8 frasi), citando i simboli per nome tra backtick. "
    "Nel campo `relevant` metti i NUMERI dei simboli necessari per capire la risposta, nell'ordine "
    "in cui conviene leggerli (dal punto d'ingresso al dettaglio), escludendo quelli non pertinenti. "
    "Se nel messaggio c'è la sezione «Flussi», descrive archi tipizzati tra simboli: renders = un componente ne "
    "mostra un altro; navigates_to = navigazione UI (value = schermata di destinazione); redirects_to = redirect; "
    "api_call = chiamata HTTP verso un endpoint; handled_by = l'endpoint è implementato da quella funzione; "
    "proxied_by = passa dal gateway; publishes/consumes = evento di dominio; calls = chiamata di funzione. "
    "Usali per spiegare cosa succede PRIMA e DOPO, nell'ordine in cui avviene. "
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


class OpenAICompatibleSynthesizer:
    """Client minimale per provider con endpoint /chat/completions compatibile OpenAI."""

    def __init__(self, api_key: str, base_url: str, model: str, max_tokens: int = 1500):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/") + "/"
        self.model = model
        self.max_tokens = max_tokens

    def complete(self, system: str, user: str) -> str:
        payload = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.1,
            "max_tokens": self.max_tokens,
        }).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}chat/completions",
            data=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "Mozilla/5.0 (Compatible; CodeGraphAgent/1.0)",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            try:
                detail = exc.read().decode("utf-8", errors="replace")[:500]
            except Exception:
                detail = ""
            detail = re.sub(r"(?i)(authorization\s*:\s*bearer\s+)[^\s\"']+", r"\1[redacted]", detail)
            suffix = f" - {detail}" if detail else ""
            raise RuntimeError(
                f"Provider LLM HTTP {exc.code} ({exc.reason}){suffix}"
            ) from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise RuntimeError(f"Provider LLM non raggiungibile: {exc}") from exc
        try:
            return str(data["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("Risposta del provider LLM in formato inatteso") from exc


def build_default_llm() -> Optional[Any]:
    """Costruisce il provider configurato, oppure disabilita l'LLM se manca una chiave."""
    provider = os.getenv("CODE_RAG_LLM_PROVIDER", "").strip().lower()
    if provider in {"openai-compatible", "openai_compatible", "groq", "openrouter"}:
        api_key = os.getenv("CODE_RAG_LLM_API_KEY", "").strip()
        base_url = os.getenv("CODE_RAG_LLM_BASE_URL", "").strip()
        model = os.getenv("CODE_RAG_LLM_MODEL", "").strip()
        print(api_key)
        print(base_url)
        print(model)
        if not api_key or not base_url or not model:
            print("[!] Configurazione LLM incompleta: servono CODE_RAG_LLM_API_KEY, "
                  "CODE_RAG_LLM_BASE_URL e CODE_RAG_LLM_MODEL.")
            return None
        try:
            return OpenAICompatibleSynthesizer(api_key, base_url, model)
        except Exception as e:
            print(f"[!] LLM non disponibile ({e}). Uso solo ricerca ibrida + grafo.")
            return None

    if not os.getenv("ANTHROPIC_API_KEY"):
        return None
    try:
        return AnthropicSynthesizer()
    except Exception as e:  # SDK non installato o configurazione errata
        print(f"[!] LLM non disponibile ({e}). Uso solo ricerca ibrida + grafo.")
        return None


class GraphQuestionAgent:
    ROLE_ORDER = {"match": 0, "callee": 1, "caller": 1, "next": 1, "previous": 1, "container": 2, "member": 3}

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
        flow_extra_nodes: int = 24,
        flow_depth: int = 4,
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
        self.flow_extra_nodes = flow_extra_nodes   # nodi aggiuntivi ammessi per i flussi (prima/dopo)
        self.flow_depth = flow_depth
        self.history: List[Dict[str, str]] = []
        self._last_focus_ids: List[str] = []
        self._last_prompt_context_ids: List[str] = []
        self._last_prompt_omitted = 0
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

    def _has_rel(self, u: str, v: str, rel: str) -> bool:
        """Un arco puo' portare piu' relazioni insieme (es. calls + renders): si controlla l'insieme."""
        return has_rel(self.graph, u, v, rel)

    @staticmethod
    def _query_terms(question: str) -> set:
        terms = set()
        for token in _IDENT_RE.findall(question.lower()):
            if len(token) < 3 or token in _STOPWORDS:
                continue
            terms.add(token)
            for suffix in ("are", "ere", "ire", "zione", "mento", "ità", "ing", "ed"):
                if len(token) > len(suffix) + 3 and token.endswith(suffix):
                    terms.add(token[:-len(suffix)])
        aliases = {
            "registrare": {"register", "registration", "signup", "sign_up", "create"},
            "registrazione": {"register", "registration", "signup", "sign_up"},
            "account": {"user", "auth", "identity"},
            "utente": {"user", "account", "auth"},
            "nuovo": {"new", "create"},
        }
        for term in list(terms):
            terms.update(aliases.get(term, set()))
        return terms

    def _is_code_request(self, question: str) -> bool:
        return bool(_CODE_REQUEST_RE.search(question))

    def _candidate_relevance(self, question: str, candidate: Dict[str, Any]) -> int:
        terms = self._query_terms(question)
        searchable = " ".join(str(candidate.get(key) or "") for key in (
            "symbol", "file", "docstring", "content", "dependencies", "calls"
        )).lower()
        return sum(1 for term in terms if term in searchable)

    def _is_wrong_domain(self, question: str, candidate: Dict[str, Any]) -> bool:
        terms = self._query_terms(question)
        path = " ".join(str(candidate.get(key) or "") for key in ("symbol", "file", "docstring")).lower()
        for domain, markers in _DOMAIN_TERMS.items():
            if any(marker in terms for marker in markers):
                continue
            if domain == "ocr" and any(marker in path for marker in markers):
                return True
        return False

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

        # c) rotte HTTP citate (es. "/identity/me", "POST /api/v1/items"): nodi `endpoint` del livello flussi
        for tok in re.findall(r"(?<![\w.:])/[\w{}\-./]*\w", question):
            path = re.sub(r"^/api(?:/v\d+)?(?=/|$)", "", tok) or "/"
            path = re.sub(r"\{[^}]*\}|:[A-Za-z_]\w*", "{}", path)
            for nid, data in self.graph.nodes(data=True):
                if data.get("symbol_type") == "endpoint" and data.get("path_norm") == path and nid not in seen:
                    seen.add(nid)
                    seeds.append(_Seed(nid, f"rotta HTTP citata nella domanda: {tok}"))

        return seeds[: self.top_k]

    def _semantic_seeds(self, question: str, include_tests: bool) -> List[_Seed]:
        candidates = self.retriever.retrieve(question, top_k=self.top_k * 4, expand_graph=False)
        candidates = [c for c in candidates if c.get("id") in self.graph]
        if not include_tests:
            filtered = [c for c in candidates if not self._is_test_path(c.get("file"))]
            candidates = filtered or candidates
        if self._is_code_request(question):
            relevant = [c for c in candidates if self._candidate_relevance(question, c) > 0
                        and not self._is_wrong_domain(question, c)]
            if relevant:
                candidates = relevant
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
        return [_Seed(
            c["id"],
            "corrispondenza lessicale/semantica con la richiesta"
            if self._candidate_relevance(question, c) else "affinità semantica con la domanda",
        ) for c in ranked]

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
                ("callee", [v for v in succ if self._has_rel(nid, v, "calls")], "chiamato da «{n}»"),
                ("caller", [u for u in pred if self._has_rel(u, nid, "calls")], "chiama «{n}»"),
                ("container", [u for u in pred if self._has_rel(u, nid, "contains")], "contiene «{n}»"),
                ("member", [v for v in succ if self._has_rel(nid, v, "contains")], "definito in «{n}»"),
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

    # ------------------------------------------------------------------ flussi (prima / dopo)
    def _flow_expand(self, question: str, seeds: List[_Seed], nodes: Dict[str, Dict[str, Any]],
                     include_tests: bool) -> str:
        """
        Aggiunge ai nodi i FLUSSI dei seed: cosa li precede e cosa li segue lungo gli archi tipizzati
        (renders, navigates_to, redirects_to, api_call, handled_by, proxied_by, publishes, consumes, calls).
        Cosi' "come funziona il login" include la pagina di registrazione raggiungibile, le chiamate API,
        il backend che le gestisce e i redirect. Modifica `nodes` e restituisce l'albero testuale dei passi
        (da mostrare nella risposta e da dare all'LLM). Vuoto se i seed non hanno archi di flusso.
        """
        if not seeds or not has_flow_edges(self.graph):
            return ""
        intent = has_flow_intent(question)
        dirs = ["before", "after"] if flow_direction(question) == "both" else [flow_direction(question)]
        depth, max_steps = (self.flow_depth, 40) if intent else (2, 12)   # senza intento: solo contesto leggero
        budget = self.flow_extra_nodes
        sections: List[str] = []
        shown_steps = set()

        for s in seeds[:2]:
            for d in dirs:
                steps = trace_flow(self.graph, s.node_id, d, max_depth=depth, max_steps=max_steps)
                if not any(st["rel"] in FLOW_RELS for st in steps):
                    continue   # solo chiamate di funzione: ci pensa gia' _expand
                display_steps = []
                for st in steps:
                    other = other_end(st, d)
                    step_key = (d, st["src"], st["dst"], st["rel"])
                    if step_key not in shown_steps:
                        shown_steps.add(step_key)
                        display_steps.append(st)
                    if other in nodes or other not in self.graph or self._is_noise(other, include_tests):
                        continue
                    if budget <= 0:
                        break
                    detail = step_detail(st)
                    detail = f", {detail}" if detail else ""
                    if d == "after":
                        reason = f"raggiunto da «{flow_label(self.graph, st['src'])}» ({st['rel']}{detail})"
                        role, via = "next", st["src"]
                    else:
                        reason = f"porta a «{flow_label(self.graph, st['dst'])}» ({st['rel']}{detail})"
                        role, via = "previous", st["dst"]
                    nodes[other] = {"id": other, "role": role, "rank": None, "reason": reason, "via": via}
                    budget -= 1
                if display_steps:
                    title = "dopo" if d == "after" else "prima di"
                    sections.append(f"Flusso {title} «{flow_label(self.graph, s.node_id)}»:\n"
                                    f"{format_flow(self.graph, display_steps, s.node_id)}")
        return "\n\n".join(sections)

    # ------------------------------------------------------------------ risposte
    def _describe(self, nodes: Dict[str, Dict[str, Any]], flow_text: str = "") -> str:
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
        if flow_text:
            lines.append("\n" + flow_text)
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

    def _refine_with_llm(self, question: str, nodes: Dict[str, Dict[str, Any]], flow_text: str = ""):
        ordered = sorted(nodes.values(),
                         key=lambda n: (self.ROLE_ORDER[n["role"]], n.get("rank") or 999))
        if self._is_code_request(question):
            match_ids = {n["id"] for n in ordered if n["role"] == "match"}
            focused = []
            for n in ordered:
                if n["role"] == "match" or n.get("via") in match_ids:
                    focused.append(n)
                    continue
                if n["role"] == "container" and self._candidate_relevance(question, {
                        "symbol": self._name(n["id"]),
                        "file": self.graph.nodes[n["id"]].get("file_path"),
                        "docstring": self.graph.nodes[n["id"]].get("docstring"),
                    }) > 0:
                    focused.append(n)
            if focused:
                ordered = focused
        max_chars = max(8000, int(os.getenv("CODE_RAG_LLM_MAX_INPUT_CHARS", "18000")))
        flow_limit = min(3500, max_chars // 5)
        reserved = len(question) + len(flow_text[:flow_limit]) + 1200
        remaining = max_chars - reserved
        items = []
        included_ids = []
        prompt_nodes = []
        omitted = []
        for n in ordered:
            d = self.graph.nodes[n["id"]]
            # I match ricevono codice; i vicini non pertinenti vengono rappresentati
            # con metadati e docstring, evitando di inviare decine di corpi duplicati.
            limit = self.code_chars if n["role"] == "match" else 160
            item = (
                f"{d.get('symbol_name')} ({d.get('symbol_type')}) — "
                f"{d.get('file_path')}:{d.get('start_line')}-{d.get('end_line')} — "
                f"ruolo: {n['role']} ({n['reason']})\n"
                f"{(d.get('docstring') or '')[:160]}\n{(d.get('content') or '')[:limit]}"
            )
            if len(item) > remaining and items:
                omitted.append(n)
                continue
            prompt_nodes.append(n)
            prompt_number = len(prompt_nodes)
            item = f"[{prompt_number}] {item}"
            items.append(item)
            included_ids.append(n["id"])
            remaining -= len(item)
        if omitted:
            by_role: Dict[str, int] = {}
            for n in omitted:
                by_role[n["role"]] = by_role.get(n["role"], 0) + 1
            summary = ", ".join(f"{count} {role}" for role, count in sorted(by_role.items()))
            items.append(f"[riassunto contesto omesso] {len(omitted)} nodi esclusi per budget ({summary}). "
                         "Sono stati mantenuti nel grafo ma non inviati al modello.")
        self._last_prompt_context_ids = included_ids
        self._last_prompt_omitted = len(omitted)
        user = f"Domanda: {question}\n\nSimboli candidati:\n\n" + "\n\n".join(items)
        if self.history:
            previous = self.history[-3:]
            user = ("Conversazione precedente (usala solo per risolvere riferimenti come "
                    "«questo» o «la funzione precedente»):\n" +
                    "\n".join(f"- Utente: {turn['question']}\n  Assistente: {turn['answer'][:500]}"
                              for turn in previous) + "\n\n" + user)
        if flow_text:
            user += "\n\nFlussi (archi tipizzati tra i simboli):\n\n" + flow_text[:flow_limit]
        raw = self.llm.complete(_LLM_SYSTEM, user)

        data = self._parse_json(raw)
        if not data:
            return raw.strip(), nodes   # testo libero: tengo tutti i nodi
        answer = str(data.get("answer", "")).strip() or raw.strip()

        picked: List[int] = []
        for x in data.get("relevant", []) or []:
            if str(x).isdigit() and 1 <= int(x) <= len(prompt_nodes) and int(x) not in picked:
                picked.append(int(x))
        if not picked:
            return answer, nodes
        refined = {}
        for pos, idx in enumerate(picked, 1):
            n = dict(prompt_nodes[idx - 1])
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
            for r in (edge_rels(d) or {"related"}):
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
            "trace": {
                "question": question,
                "seed_nodes": [],
                "context_nodes": [],
                "used_nodes": [],
                "navigated_edges": [],
                "steps": [],
            },
        }

    def _trace(self, question: str, seeds: List[_Seed], nodes: Dict[str, Dict[str, Any]],
               refined: Optional[Dict[str, Dict[str, Any]]] = None,
               flow_text: str = "", llm_status: Optional[str] = None) -> Dict[str, Any]:
        """Restituisce una traccia auditabile del percorso seguito dall'agente."""
        context_ids = list(nodes)
        prompt_ids = self._last_prompt_context_ids or context_ids
        used_ids = list((refined or nodes))
        seed_ids = [s.node_id for s in seeds if s.node_id in self.graph]
        prompt_context = []
        for nid in prompt_ids:
            if nid not in self.graph or nid not in nodes:
                continue
            data = self.graph.nodes[nid]
            prompt_context.append({
                "id": nid,
                "name": self._name(nid),
                "role": nodes[nid].get("role"),
                "file": data.get("file_path", ""),
                "start_line": data.get("start_line", 0),
                "end_line": data.get("end_line", 0),
                "content_excerpt": (data.get("content") or "")[:self.code_chars],
            })
        navigated = []
        seen_edges = set()
        for source in context_ids:
            for target in self.graph.successors(source):
                if target not in nodes:
                    continue
                data = self.graph.get_edge_data(source, target) or {}
                relations = sorted(edge_rels(data))
                key = (source, target, tuple(relations))
                if key in seen_edges:
                    continue
                seen_edges.add(key)
                navigated.append({
                    "source": source,
                    "target": target,
                    "source_name": self._name(source),
                    "target_name": self._name(target),
                    "relations": relations,
                    "meta": data.get("meta", {}),
                })
        steps = [
            {"tool": "explicit_seed_search", "status": "completed", "count": len(seed_ids)},
            {"tool": "hybrid_retrieval", "status": "completed",
             "count": len(seeds), "description": "Ricerca semantica e lessicale"},
            {"tool": "graph_expansion", "status": "completed",
             "count": len(context_ids), "description": "Vicini, chiamanti, chiamati e contenitori"},
        ]
        if flow_text:
            steps.append({"tool": "flow_trace", "status": "completed",
                          "count": len(navigated), "description": "Flussi UI/API/eventi"})
        if llm_status:
            steps.append({"tool": "llm_synthesis", "status": llm_status,
                          "count": len(used_ids) if llm_status == "completed" else 0,
                          "description": "Sintesi vincolata al contesto"})
        return {
            "question": question,
            "seed_nodes": seed_ids,
            "context_nodes": context_ids,
            "used_nodes": used_ids,
            "prompt_context": prompt_context,
            "flow_context": flow_text[:6000],
            "prompt_omitted": self._last_prompt_omitted,
            "navigated_edges": navigated,
            "steps": steps,
        }

    # ------------------------------------------------------------------ API pubblica
    def ask(self, question: str) -> Dict[str, Any]:
        question = (question or "").strip()
        if not question:
            return self._result(question, "empty", "Scrivi una domanda sul codice.", {})

        include_tests = "test" in question.lower()
        seeds = self._explicit_seeds(question, include_tests)
        if not seeds and self.history and self._last_focus_ids and re.search(
                r"\b(questo|questa|precedente|poi|dopo|prima|it|this|that|previous)\b",
                question, re.I):
            seeds = [_Seed(nid, "contesto della conversazione precedente")
                     for nid in self._last_focus_ids if nid in self.graph][:self.top_k]

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
            result = self._result(question, "retrieval",
                                  "Non ho trovato nulla di rilevante nel grafo per questa domanda.", {})
            result["trace"] = {"question": question, "seed_nodes": [], "context_nodes": [],
                               "used_nodes": [], "navigated_edges": [],
                               "steps": [{"tool": "graph_search", "status": "completed", "count": 0}]}
            self.history.append({"question": question, "answer": result["answer"]})
            return result

        nodes = self._expand(seeds, include_tests)
        flow_text = self._flow_expand(question, seeds, nodes, include_tests)

        if self.llm is not None:
            try:
                answer, refined = self._refine_with_llm(question, nodes, flow_text)
                result = self._result(question, "llm", answer, refined)
                result["trace"] = self._trace(question, seeds, nodes, refined, flow_text, "completed")
                result["conversation"] = {"turn": len(self.history) + 1,
                                          "previous_turns": len(self.history)}
                self._last_focus_ids = list(refined)
                self.history.append({"question": question, "answer": answer})
                return result
            except Exception as e:
                print(f"[!] Sintesi LLM fallita ({e}): uso il riepilogo deterministico.")

        answer = self._describe(nodes, flow_text)
        result = self._result(question, "retrieval", answer, nodes)
        result["trace"] = self._trace(
            question, seeds, nodes, nodes, flow_text,
            "failed" if self.llm is not None else None,
        )
        result["conversation"] = {"turn": len(self.history) + 1,
                                  "previous_turns": len(self.history)}
        self._last_focus_ids = list(nodes)
        self.history.append({"question": question, "answer": answer})
        return result
