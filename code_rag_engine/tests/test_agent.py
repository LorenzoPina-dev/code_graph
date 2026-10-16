"""
Test dell'agente di domande (core/agent.py).

Non scaricano modelli e non usano Qdrant: il retriever, il reranker e l'LLM sono sostituiti
da doppioni deterministici, così si verifica solo la logica dell'agente
(scelta dei seed, filtro test, espansione sul grafo, sintesi LLM e fallback).
"""

import json
from types import SimpleNamespace

import networkx as nx
import pytest

from core.agent import GraphQuestionAgent


# ----------------------------------------------------------------------------- grafo di prova
def _add(g, file, sym, typ="function", calls=(), line=1):
    nid = f"{file}::{sym}"
    g.add_node(nid, file_path=file, symbol_name=sym, symbol_type=typ, start_line=line,
               end_line=line + 3, docstring=f"doc {sym}", dependencies=[], calls=list(calls),
               content=f"code of {sym}")
    return nid


@pytest.fixture()
def toy():
    g = nx.DiGraph()
    ids = SimpleNamespace(
        login=_add(g, "src/auth.ts", "login", calls=["verifyToken", "hash"]),
        verify=_add(g, "src/auth.ts", "verifyToken"),
        hash=_add(g, "src/crypto.ts", "hash"),
        svc=_add(g, "src/auth.ts", "AuthService", "class"),
        refresh=_add(g, "src/auth.ts", "AuthService.refresh", "method"),
        route=_add(g, "src/routes.ts", "loginRoute", calls=["login"]),
        test=_add(g, "src/auth.test.ts", "testLogin", calls=["login"]),
        anon=_add(g, "src/auth.ts", "anonymous"),
    )
    for a, b in [(ids.login, ids.verify), (ids.login, ids.hash), (ids.route, ids.login), (ids.test, ids.login)]:
        g.add_edge(a, b, relation="calls")
    g.add_edge(ids.svc, ids.refresh, relation="contains")
    g.add_edge(ids.svc, ids.login, relation="contains")
    return g, ids


# ----------------------------------------------------------------------------- doppioni
class FakeRetriever:
    def __init__(self, graph, node_ids):
        self.graph, self.node_ids = graph, node_ids

    def retrieve(self, question, top_k=5, expand_graph=False):
        out = []
        for rank, nid in enumerate(self.node_ids):
            d = self.graph.nodes[nid]
            out.append({"id": nid, "file": d["file_path"], "symbol": d["symbol_name"],
                        "score": 1.0 / (rank + 1)})
        return out[:top_k]


class FakeReranker:
    """login molto rilevante, hash molto poco, tutto il resto nella media."""
    def rerank(self, query, candidates, top_n=3):
        scored = []
        for c in candidates:
            c = dict(c)
            c["rerank_score"] = {"login": 5.0, "hash": -20.0}.get(c["symbol"], 1.0)
            scored.append(c)
        return sorted(scored, key=lambda c: -c["rerank_score"])[:top_n]


def make_agent(graph, retrieved=(), reranker=None, llm=None, **kwargs):
    agent = GraphQuestionAgent(SimpleNamespace(graph=graph), reranker=reranker, llm=llm,
                               use_reranker=False, **kwargs)
    agent.retriever = FakeRetriever(graph, list(retrieved))
    return agent


def roles(result):
    return {f["id"]: f["role"] for f in result["focus"]}


# ----------------------------------------------------------------------------- seed ed espansione
def test_explicit_symbol_is_first_match(toy):
    g, ids = toy
    res = make_agent(g, [ids.verify], FakeReranker()).ask("cosa fa la funzione login?")
    assert res["focus"][0]["id"] == ids.login
    assert res["focus"][0]["role"] == "match" and res["focus"][0]["rank"] == 1


def test_expansion_adds_callees_callers_and_container(toy):
    g, ids = toy
    r = roles(make_agent(g, [], FakeReranker()).ask("cosa fa la funzione login?"))
    assert r[ids.verify] == "callee" and r[ids.hash] == "callee"
    assert r[ids.route] == "caller"
    assert r[ids.svc] == "container"


def test_test_files_and_anonymous_nodes_are_not_neighbours(toy):
    g, ids = toy
    focus_ids = [f["id"] for f in make_agent(g).ask("cosa fa la funzione login?")["focus"]]
    assert ids.test not in focus_ids
    assert ids.anon not in focus_ids


def test_tests_are_included_when_question_mentions_them(toy):
    g, ids = toy
    focus_ids = [f["id"] for f in make_agent(g).ask("come viene testato login?")["focus"]]
    assert ids.test in focus_ids


def test_file_mention_seeds_the_symbols_of_that_file(toy):
    g, ids = toy
    focus_ids = [f["id"] for f in make_agent(g).ask("cosa contiene crypto.ts?")["focus"]]
    assert ids.hash in focus_ids


def test_ambiguous_symbol_names_are_ignored():
    g = nx.DiGraph()
    for i in range(6):
        _add(g, f"f{i}.py", "run")
    agent = make_agent(g)
    assert agent._explicit_seeds("cosa fa run?", include_tests=False) == []


def test_semantic_seeds_are_used_without_explicit_symbol(toy):
    g, ids = toy
    res = make_agent(g, [ids.verify]).ask("verifica della firma")
    assert res["focus"][0]["id"] == ids.verify and res["focus"][0]["role"] == "match"


def test_rerank_margin_drops_weak_seed(toy):
    g, ids = toy
    res = make_agent(g, [ids.hash, ids.login], FakeReranker()).ask("gestione dell'accesso utente")
    matches = [f["id"] for f in res["focus"] if f["role"] == "match"]
    assert ids.login in matches and ids.hash not in matches


def test_max_nodes_cap_is_respected(toy):
    g, ids = toy
    res = make_agent(g, [], FakeReranker(), max_nodes=3).ask("cosa fa la funzione login?")
    assert len(res["focus"]) <= 3


def test_result_is_json_serialisable_and_has_stats(toy):
    g, ids = toy
    res = make_agent(g, [], FakeReranker()).ask("cosa fa la funzione login?")
    json.dumps(res)
    assert res["stats"]["nodes"] == len(res["focus"])
    assert res["stats"]["matches"] >= 1


# ----------------------------------------------------------------------------- casi particolari
def test_empty_question(toy):
    g, _ = toy
    res = make_agent(g).ask("   ")
    assert res["mode"] == "empty" and res["focus"] == []


def test_no_results_gives_message_not_error():
    g = nx.DiGraph()
    res = make_agent(g).ask("qualcosa che non esiste")
    assert res["focus"] == [] and "nulla" in res["answer"].lower()


def test_overview_question_without_symbol(toy):
    g, _ = toy
    res = make_agent(g).ask("quante funzioni ci sono?")
    assert res["mode"] == "overview" and res["focus"] == []
    assert str(g.number_of_nodes()) in res["answer"]


def test_overview_not_triggered_when_a_symbol_is_named(toy):
    g, ids = toy
    res = make_agent(g, [], FakeReranker()).ask("quante volte viene chiamata login?")
    assert res["mode"] != "overview"
    assert res["focus"][0]["id"] == ids.login


# ----------------------------------------------------------------------------- LLM
class LlmPicksVerifyThenLogin:
    """Trova nel prompt i numeri di verifyToken e login e li restituisce in quest'ordine."""
    def complete(self, system, user):
        blocks = [b for b in user.split("\n\n") if b.startswith("[")]
        number = lambda name: next(int(b[1:b.index("]")]) for b in blocks if f"] {name} (" in b)
        payload = {"answer": "`login` verifica il token.",
                   "relevant": [number("verifyToken"), number("login")]}
        return "```json\n" + json.dumps(payload) + "\n```"


class LlmFreeText:
    def complete(self, system, user):
        return "Risposta libera senza JSON"


class LlmBroken:
    def complete(self, system, user):
        raise RuntimeError("boom")


def test_llm_prunes_and_orders_nodes(toy):
    g, ids = toy
    res = make_agent(g, [], FakeReranker(), llm=LlmPicksVerifyThenLogin()).ask("cosa fa la funzione login?")
    assert res["mode"] == "llm"
    assert [f["id"] for f in res["focus"]] == [ids.verify, ids.login]
    assert [f["rank"] for f in res["focus"]] == [1, 2]
    assert "verifica" in res["answer"]


def test_llm_free_text_keeps_all_nodes(toy):
    g, _ = toy
    res = make_agent(g, [], FakeReranker(), llm=LlmFreeText()).ask("cosa fa la funzione login?")
    assert res["mode"] == "llm" and res["answer"].startswith("Risposta libera")
    assert len(res["focus"]) > 2


def test_llm_failure_falls_back_to_deterministic_answer(toy):
    g, _ = toy
    res = make_agent(g, [], FakeReranker(), llm=LlmBroken()).ask("cosa fa la funzione login?")
    assert res["mode"] == "retrieval"
    assert res["focus"] and "`login`" in res["answer"]


def test_llm_ignores_out_of_range_and_duplicate_indices(toy):
    g, ids = toy

    class Llm:
        def complete(self, system, user):
            return json.dumps({"answer": "ok", "relevant": [1, 1, 999, "x", 2]})

    res = make_agent(g, [], FakeReranker(), llm=Llm()).ask("cosa fa la funzione login?")
    assert len(res["focus"]) == 2


@pytest.mark.parametrize("raw,expected", [
    ('{"answer": "a", "relevant": [1]}', {"answer": "a", "relevant": [1]}),
    ('```json\n{"answer": "a"}\n```', {"answer": "a"}),
    ('testo prima {"answer": "a"} testo dopo', {"answer": "a"}),
    ("niente json", None),
    ("{rotto", None),
])
def test_parse_json_is_tolerant(raw, expected):
    assert GraphQuestionAgent._parse_json(raw) == expected
