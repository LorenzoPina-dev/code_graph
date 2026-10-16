"""
Test del visualizzatore 3D con pannello agente e del server locale (serve_graph.py).

- HTML: il pannello compare solo con un endpoint agente, il JSON dei dati non può chiudere
  il tag <script> (bug "</script>" nel codice sorgente), il JavaScript generato è valido.
- Server: GET / e POST /api/ask rispondono come atteso, anche in caso di errori.
Nessun modello viene scaricato: l'agente è sostituito da un doppione.
"""

import json
import re
import shutil
import subprocess
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import networkx as nx
import pytest

from core.visualizer import Graph3DVisualizer
from serve_graph import make_handler


# ----------------------------------------------------------------------------- HTML
@pytest.fixture()
def graph_with_script_tag():
    g = nx.DiGraph()
    g.add_node("a.ts::foo", file_path="a.ts", symbol_name="foo", symbol_type="function",
               start_line=1, end_line=5, docstring="d", dependencies=[], calls=["bar"],
               content="const html = '<script>alert(1)</script>';  // <!-- commento")
    g.add_node("a.ts::bar", file_path="a.ts", symbol_name="bar", symbol_type="method",
               start_line=6, end_line=9, docstring="", dependencies=[], calls=[], content="bar()")
    g.add_edge("a.ts::foo", "a.ts::bar", relation="calls")
    return g


def test_agent_panel_only_with_endpoint(graph_with_script_tag):
    viz = Graph3DVisualizer(graph_with_script_tag)
    with_agent = viz.render_html(agent_endpoint="/api/ask")
    without_agent = viz.render_html()
    assert 'id="agent-panel"' in with_agent and 'id="agent-input"' in with_agent
    assert 'id="agent-panel"' not in without_agent


def test_endpoint_is_embedded_as_json(graph_with_script_tag):
    viz = Graph3DVisualizer(graph_with_script_tag)
    assert 'const AGENT_ENDPOINT = "/api/ask";' in viz.render_html(agent_endpoint="/api/ask")
    assert "const AGENT_ENDPOINT = null;" in viz.render_html()


def test_source_code_cannot_close_the_script_tag(graph_with_script_tag):
    html = Graph3DVisualizer(graph_with_script_tag).render_html(agent_endpoint="/api/ask")
    # Due soli </script> reali: quello dello script CDN e quello inline. Il </script> presente
    # nel codice sorgente indicizzato deve comparire solo come <\/script> (escape JSON valido).
    assert len(re.findall(r"</script", html, re.I)) == 2
    assert "<\\/script>" in html
    assert "<!--" not in html.split("const gData", 1)[1]


def test_filters_and_rotation_controls_are_present(graph_with_script_tag):
    html = Graph3DVisualizer(graph_with_script_tag).render_html()
    for group in ("class", "function", "method", "module"):
        assert f'data-group="{group}"' in html
    assert 'id="btn-rotate"' in html and 'id="btn-reset"' in html
    assert 'id="graph-3d"' in html and "3d-graph" not in html.replace("graph-3d", "")


def test_export_html_stays_backward_compatible(graph_with_script_tag, tmp_path):
    out = Graph3DVisualizer(graph_with_script_tag).export_html(str(tmp_path / "g.html"))
    content = open(out, encoding="utf-8").read()
    assert "ForceGraph3D" in content and 'id="agent-panel"' not in content


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js non installato")
@pytest.mark.parametrize("endpoint", [None, "/api/ask"])
def test_generated_javascript_is_syntactically_valid(graph_with_script_tag, tmp_path, endpoint):
    html = Graph3DVisualizer(graph_with_script_tag).render_html(agent_endpoint=endpoint)
    inline = re.findall(r"<script>(.*?)</script>", html, re.S)
    assert inline, "nessuno script inline trovato"
    js_file = tmp_path / "viewer.js"
    js_file.write_text(inline[-1], encoding="utf-8")
    proc = subprocess.run(["node", "--check", str(js_file)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


# ----------------------------------------------------------------------------- server
class FakeAgent:
    def __init__(self):
        self.questions = []
        self.fail = False

    def ask(self, question):
        self.questions.append(question)
        if self.fail:
            raise RuntimeError("errore simulato")
        return {"question": question, "mode": "retrieval", "answer": "ciao àèì",
                "focus": [{"id": "a.ts::foo", "role": "match", "rank": 1, "reason": "x", "via": None}],
                "stats": {}}


@pytest.fixture()
def server():
    agent = FakeAgent()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler("<html>agent-panel</html>", agent))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", agent
    httpd.shutdown()
    httpd.server_close()


def _post(url, payload, raw=False):
    data = payload if raw else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8"))


def test_get_root_serves_the_viewer(server):
    base, _ = server
    with urllib.request.urlopen(base + "/") as resp:
        assert resp.status == 200
        assert "text/html" in resp.headers["Content-Type"]
        assert "agent-panel" in resp.read().decode("utf-8")


def test_health_and_unknown_paths(server):
    base, _ = server
    with urllib.request.urlopen(base + "/health") as resp:
        assert json.loads(resp.read())["status"] == "ok"
    with pytest.raises(urllib.error.HTTPError) as err:
        urllib.request.urlopen(base + "/nope")
    assert err.value.code == 404


def test_ask_returns_focus_as_json_with_utf8(server):
    base, agent = server
    status, body = _post(base + "/api/ask", {"question": "come funziona foo?"})
    assert status == 200
    assert body["focus"][0]["id"] == "a.ts::foo" and body["answer"] == "ciao àèì"
    assert agent.questions == ["come funziona foo?"]


@pytest.mark.parametrize("payload,raw", [({"question": "   "}, False), ({}, False), (b"non json", True)])
def test_ask_rejects_bad_requests(server, payload, raw):
    base, agent = server
    status, body = _post(base + "/api/ask", payload, raw=raw)
    assert status == 400 and "error" in body
    assert agent.questions == []


def test_agent_error_returns_500_and_server_keeps_working(server):
    base, agent = server
    agent.fail = True
    status, body = _post(base + "/api/ask", {"question": "boom?"})
    assert status == 500 and "errore simulato" in body["error"]
    agent.fail = False
    status, _ = _post(base + "/api/ask", {"question": "ok?"})
    assert status == 200


def test_post_to_unknown_path_is_404(server):
    base, _ = server
    status, _ = _post(base + "/api/other", {"question": "x"})
    assert status == 404
