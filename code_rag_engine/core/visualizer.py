"""
Modulo di esportazione interattiva 3D per il Code Property Graph (CPG).
Genera una Web App 3D autonoma (Three.js / 3d-force-graph) per esplorare:
- Nodi con color-coding semantico (Class, Function, Method, Module)
- Dimensione proporzionale alle connessioni (grado di centralità)
- Archi orientati e con etichetta (contains, calls, imports)
- Pannello laterale interattivo al click con ispezione del codice sorgente, docstring, chiamate e metadati
- Ricerca istantanea con auto-focus e filtro per tipo di simbolo
- Navigazione vicini: pulsanti per spostarsi ai nodi collegati (incoming/outgoing)
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
import networkx as nx

class Graph3DVisualizer:
    def __init__(self, graph: nx.DiGraph):
        self.graph = graph

    # Palette centralizzata: colore per tipo → indice numerico per risparmiare bytes nel JSON
    TYPE_COLORS = {
        "class":      "#ff79c6",
        "struct":     "#ffb86c",
        "interface":  "#8be9fd",
        "function":   "#50fa7b",
        "method":     "#8be9fd",
        "module":     "#bd93f9",
        "code_block": "#f1fa8c",
    }
    DEFAULT_COLOR = "#f8f8f2"

    def to_graph_data(self) -> Dict[str, Any]:
        """
        Converte il grafo NetworkX nel formato JSON richiesto dal visualizzatore 3D.
        Ottimizzazione: il codice sorgente completo è escluso dal JSON principale
        e restituito in un dizionario separato (code_store) per ridurre le dimensioni HTML.
        """
        nodes = []
        links = []
        code_store: Dict[str, str] = {}   # node_id → codice sorgente (lazy-load nel browser)

        degree_dict = dict(self.graph.degree())

        for node_id, data in self.graph.nodes(data=True):
            sym_type = data.get("symbol_type", "function")
            color = self.TYPE_COLORS.get(sym_type, self.DEFAULT_COLOR)
            deg = degree_dict.get(node_id, 1)
            val = max(2, min(15, deg * 2))

            # Tronca docstring lunga nel payload principale (max 300 chars)
            docstring = (data.get("docstring") or "")[:300]

            # Il codice sorgente va nel lookup separato, NON nel nodo principale
            code = data.get("content", "")
            if code:
                code_store[node_id] = code

            # Calls: massimo 20 voci per non gonfiare il JSON
            calls = (data.get("calls") or [])[:20]
            # Dependencies: massimo 15 voci
            deps = (data.get("dependencies") or [])[:15]

            nodes.append({
                "id": node_id,
                "name": data.get("symbol_name", node_id),
                "type": sym_type,
                "fp": data.get("file_path", ""),       # chiave abbreviata
                "sl": data.get("start_line", 0),        # start_line abbreviato
                "el": data.get("end_line", 0),          # end_line abbreviato
                "doc": docstring,
                "calls": calls,
                "deps": deps,
                "color": color,
                "val": val
            })

        link_colors = {
            "contains": "#bd93f9",
            "calls": "#ff5555",
            "imports": "#f1fa8c"
        }

        for source, target, edge_data in self.graph.edges(data=True):
            rel = edge_data.get("relation", "related")
            links.append({
                "source": source,
                "target": target,
                "relation": rel,
                "color": link_colors.get(rel, "#6272a4")
            })

        return {"nodes": nodes, "links": links, "code_store": code_store}

    @staticmethod
    def _safe_json(s: str) -> str:
        """Rende il JSON sicuro per l'inserimento dentro un tag <script> inline:
        impedisce che '</script>' o '<!--' presenti nel codice sorgente chiudano il blocco."""
        return (s.replace("</", "<\\/")
                 .replace("<!--", "<\\!--")
                 .replace("\u2028", "\\u2028")
                 .replace("\u2029", "\\u2029"))

    def export_html(self, output_path: str = "graph_3d_viewer.html", title: str = "Code Property Graph 3D Explorer") -> str:
        """Genera un file HTML autonomo che renderizza il grafo 3D interattivo."""
        graph_data = self.to_graph_data()
        code_store = graph_data.pop("code_store", {})
        graph_json = self._safe_json(json.dumps(graph_data, ensure_ascii=False, separators=(',', ':')))
        code_store_json = self._safe_json(json.dumps(code_store, ensure_ascii=False, separators=(',', ':')))

        html_template = f"""<!DOCTYPE html>
<html lang="it">
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <style>
    body {{
      margin: 0;
      overflow: hidden;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background: #0d1117;
      color: #c9d1d9;
    }}
    #graph-3d {{
      width: 100vw;
      height: 100vh;
    }}
    #ui-container {{
      position: absolute;
      top: 15px;
      left: 15px;
      z-index: 100;
      background: rgba(22, 27, 34, 0.85);
      backdrop-filter: blur(8px);
      padding: 16px 20px;
      border-radius: 10px;
      border: 1px solid #30363d;
      box-shadow: 0 8px 24px rgba(0,0,0,0.5);
      max-width: 320px;
    }}
    #ui-container h2 {{
      margin: 0 0 10px 0;
      font-size: 16px;
      color: #58a6ff;
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    .stat-badge {{
      display: inline-block;
      padding: 2px 8px;
      background: #21262d;
      border-radius: 12px;
      font-size: 11px;
      color: #8b949e;
      margin-right: 6px;
      border: 1px solid #30363d;
    }}
    .search-box {{
      width: 100%;
      box-sizing: border-box;
      padding: 8px 12px;
      background: #0d1117;
      border: 1px solid #30363d;
      border-radius: 6px;
      color: #f0f6fc;
      font-size: 13px;
      margin-top: 10px;
      outline: none;
    }}
    .search-box:focus {{
      border-color: #58a6ff;
      box-shadow: 0 0 0 2px rgba(88,166,255,0.2);
    }}
    .legend {{
      margin-top: 12px;
      font-size: 12px;
      display: flex;
      flex-direction: column;
      gap: 5px;
    }}
    .legend-item {{
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    .dot {{
      width: 10px;
      height: 10px;
      border-radius: 50%;
      display: inline-block;
    }}
    #details-panel {{
      position: absolute;
      top: 15px;
      right: 15px;
      width: 420px;
      max-height: calc(100vh - 30px);
      background: rgba(22, 27, 34, 0.95);
      backdrop-filter: blur(10px);
      border-radius: 10px;
      border: 1px solid #30363d;
      box-shadow: -4px 8px 24px rgba(0,0,0,0.6);
      z-index: 100;
      display: none;
      flex-direction: column;
      overflow: hidden;
    }}
    #details-header {{
      padding: 14px 18px;
      background: #161b22;
      border-bottom: 1px solid #30363d;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}
    #details-title {{
      font-size: 15px;
      font-weight: 600;
      color: #58a6ff;
      word-break: break-all;
    }}
    .close-btn {{
      cursor: pointer;
      font-size: 18px;
      color: #8b949e;
      background: none;
      border: none;
    }}
    .close-btn:hover {{
      color: #f0f6fc;
    }}
    #details-body {{
      padding: 16px 18px;
      overflow-y: auto;
      font-size: 13px;
      line-height: 1.5;
    }}
    .meta-group {{
      margin-bottom: 12px;
    }}
    .meta-label {{
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      color: #8b949e;
      margin-bottom: 4px;
    }}
    .meta-val {{
      color: #f0f6fc;
      font-family: monospace;
      font-size: 12px;
    }}
    pre.code-view {{
      background: #0d1117;
      padding: 12px;
      border-radius: 6px;
      border: 1px solid #30363d;
      color: #e6edf3;
      overflow-x: auto;
      font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
      font-size: 12px;
      max-height: 250px;
    }}
    .tag {{
      display: inline-block;
      padding: 2px 6px;
      margin: 2px;
      background: #21262d;
      border: 1px solid #30363d;
      border-radius: 4px;
      font-size: 11px;
      font-family: monospace;
      color: #7ee787;
    }}
    #instructions {{
      position: absolute;
      bottom: 15px;
      left: 15px;
      z-index: 100;
      background: rgba(22, 27, 34, 0.7);
      padding: 8px 14px;
      border-radius: 6px;
      font-size: 11px;
      color: #8b949e;
      pointer-events: none;
    }}

    /* ── Neighbor navigation ── */
    #nav-section {{
      margin-bottom: 14px;
      border: 1px solid #30363d;
      border-radius: 8px;
      overflow: hidden;
    }}
    #nav-tabs {{
      display: flex;
      background: #161b22;
      border-bottom: 1px solid #30363d;
    }}
    .nav-tab {{
      flex: 1;
      padding: 7px 0;
      font-size: 12px;
      font-weight: 600;
      cursor: pointer;
      border: none;
      background: transparent;
      color: #8b949e;
      transition: color 0.15s, background 0.15s;
      letter-spacing: 0.3px;
    }}
    .nav-tab:first-child {{ border-right: 1px solid #30363d; }}
    .nav-tab.active {{
      color: #58a6ff;
      background: #0d1117;
    }}
    .nav-tab:hover:not(.active) {{
      color: #c9d1d9;
      background: #21262d;
    }}
    #nav-list {{
      max-height: 140px;
      overflow-y: auto;
      padding: 6px 0;
    }}
    .nav-item {{
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 5px 10px;
      cursor: pointer;
      transition: background 0.12s;
      border-bottom: 1px solid #21262d;
    }}
    .nav-item:last-child {{ border-bottom: none; }}
    .nav-item:hover {{ background: #21262d; }}
    .nav-dot {{
      width: 8px;
      height: 8px;
      border-radius: 50%;
      flex-shrink: 0;
    }}
    .nav-name {{
      font-family: monospace;
      font-size: 12px;
      color: #c9d1d9;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      flex: 1;
    }}
    .nav-rel {{
      font-size: 10px;
      color: #6e7681;
      flex-shrink: 0;
    }}
    #nav-empty {{
      padding: 8px 10px;
      color: #6e7681;
      font-size: 11px;
      font-style: italic;
    }}
    /* breadcrumb back button */
    #nav-back {{
      display: none;
      align-items: center;
      gap: 6px;
      padding: 6px 10px;
      cursor: pointer;
      background: #161b22;
      border-bottom: 1px solid #30363d;
      font-size: 12px;
      color: #8b949e;
      transition: color 0.12s;
    }}
    #nav-back:hover {{ color: #58a6ff; }}
    #nav-back-label {{ font-family: monospace; max-width: 200px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
  </style>
  <script src="https://unpkg.com/3d-force-graph@1.73.3/dist/3d-force-graph.min.js"></script>
</head>
<body>
  <div id="graph-3d"></div>

  <div id="ui-container">
    <h2>
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="6" cy="6" r="3"/><circle cx="18" cy="18" r="3"/><circle cx="6" cy="18" r="3"/><line x1="6" y1="9" x2="6" y2="15"/><line x1="9" y1="6" x2="15" y2="18"/></svg>
      Code Property Graph 3D
    </h2>
    <div>
      <span class="stat-badge" id="node-count">{len(graph_data['nodes'])} Nodi</span>
      <span class="stat-badge" id="edge-count">{len(graph_data['links'])} Archi</span>
    </div>
    
    <input type="text" id="search-input" class="search-box" placeholder="Cerca simbolo o file...">

    <div class="legend">
      <div class="legend-item"><span class="dot" style="background: #ff79c6;"></span> Classi</div>
      <div class="legend-item"><span class="dot" style="background: #50fa7b;"></span> Funzioni</div>
      <div class="legend-item"><span class="dot" style="background: #8be9fd;"></span> Metodi</div>
      <div class="legend-item"><span class="dot" style="background: #bd93f9;"></span> Moduli</div>
      <div class="legend-item"><span class="dot" style="background: #ff5555;"></span> Archi Calls (chiamate)</div>
      <div class="legend-item"><span class="dot" style="background: #bd93f9;"></span> Archi Contains (class/def)</div>
    </div>
  </div>

  <div id="details-panel">
    <div id="details-header">
      <span id="details-title">Nome Simbolo</span>
      <button class="close-btn" onclick="closeDetails()">&times;</button>
    </div>
    <div id="details-body">

      <!-- Tipo & File -->
      <div class="meta-group">
        <div class="meta-label">Tipo &amp; File</div>
        <div class="meta-val"><span id="meta-type" class="tag"></span> <span id="meta-file"></span></div>
      </div>

      <!-- ── Navigazione Vicini ── -->
      <div id="nav-section">
        <div id="nav-back" onclick="navBack()">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="15 18 9 12 15 6"/></svg>
          <span id="nav-back-label"></span>
        </div>
        <div id="nav-tabs">
          <button class="nav-tab active" id="tab-out" onclick="switchTab('out')">
            &#8594; Outgoing
          </button>
          <button class="nav-tab" id="tab-in" onclick="switchTab('in')">
            Incoming &#8592;
          </button>
        </div>
        <div id="nav-list">
          <div id="nav-empty">Nessun vicino.</div>
        </div>
      </div>

      <!-- Docstring -->
      <div class="meta-group">
        <div class="meta-label">Docstring</div>
        <div id="meta-docstring" style="color: #8b949e; font-style: italic;"></div>
      </div>

      <!-- Chiamate AST -->
      <div class="meta-group">
        <div class="meta-label">Chiamate Rilevate (AST Calls)</div>
        <div id="meta-calls"></div>
      </div>

      <!-- Dipendenze -->
      <div class="meta-group">
        <div class="meta-label">Dipendenze Importate</div>
        <div id="meta-deps"></div>
      </div>

      <!-- Codice -->
      <div class="meta-group">
        <div class="meta-label">Codice Sorgente</div>
        <pre class="code-view" id="meta-code"></pre>
      </div>
    </div>
  </div>

  <div id="instructions">
    Trascina: Ruota 3D | Rotella: Zoom | Clic su nodo: Ispeziona &amp; Focus
  </div>

  <script>
    const gData = {graph_json};
    const codeStore = {code_store_json};

    // ── Index for fast lookup ──
    const nodeById = {{}};
    gData.nodes.forEach(n => {{ nodeById[n.id] = n; }});

    // Precompute adjacency: outgoing and incoming per node id
    const outLinks = {{}};  // nodeId -> [{{neighborId, relation}}]
    const inLinks  = {{}};  // nodeId -> [{{neighborId, relation}}]
    gData.nodes.forEach(n => {{ outLinks[n.id] = []; inLinks[n.id] = []; }});
    gData.links.forEach(l => {{
      const src = (typeof l.source === 'object') ? l.source.id : l.source;
      const tgt = (typeof l.target === 'object') ? l.target.id : l.target;
      if (outLinks[src]) outLinks[src].push({{ id: tgt, relation: l.relation }});
      if (inLinks[tgt])  inLinks[tgt].push({{ id: src, relation: l.relation }});
    }});

    // Navigation state
    let currentNodeId = null;
    let navTabMode = 'out';
    let navHistory = [];   // stack of {{id, name}} for back navigation

    const Graph = ForceGraph3D()(document.getElementById('graph-3d'))
      .graphData(gData)
      .nodeId('id')
      .nodeLabel(node => `${{node.name}} [${{node.type}}]\\n${{node.fp}}`)
      .nodeColor(node => node.color)
      .nodeVal(node => node.val)
      .nodeResolution(16)
      .linkSource('source')
      .linkTarget('target')
      .linkColor(link => link.color)
      .linkDirectionalParticles(2)
      .linkDirectionalParticleWidth(1.2)
      .linkDirectionalParticleSpeed(0.006)
      .linkWidth(1.2)
      .linkLabel(link => link.relation)
      .onNodeClick(node => {{
        navHistory = [];
        openDetails(node, false);
        focusNode(node);
      }});

    // Re-index links after graph init (ForceGraph resolves ids to objects)
    Graph.onEngineStop(() => {{
      gData.links.forEach(l => {{
        const src = (typeof l.source === 'object') ? l.source.id : l.source;
        const tgt = (typeof l.target === 'object') ? l.target.id : l.target;
        if (!outLinks[src] || outLinks[src].find(x => x.id === tgt)) return;
        outLinks[src].push({{ id: tgt, relation: l.relation }});
        if (inLinks[tgt]) inLinks[tgt].push({{ id: src, relation: l.relation }});
      }});
    }});

    function focusNode(node) {{
      if (!node || node.x === undefined) return;
      const distance = 80;
      const distRatio = 1 + distance / Math.hypot(node.x || 1, node.y || 1, node.z || 1);
      Graph.cameraPosition(
        {{ x: node.x * distRatio, y: node.y * distRatio, z: node.z * distRatio }},
        node,
        1200
      );
    }}

    function resolveNode(id) {{
      return nodeById[id] || gData.nodes.find(n => n.id === id);
    }}

    // ── Neighbor nav UI ──
    function switchTab(mode) {{
      navTabMode = mode;
      document.getElementById('tab-out').classList.toggle('active', mode === 'out');
      document.getElementById('tab-in').classList.toggle('active', mode === 'in');
      renderNavList(currentNodeId);
    }}

    function renderNavList(nodeId) {{
      const list = document.getElementById('nav-list');
      list.innerHTML = '';
      const neighbors = (navTabMode === 'out' ? outLinks[nodeId] : inLinks[nodeId]) || [];
      if (neighbors.length === 0) {{
        list.innerHTML = '<div id="nav-empty">Nessun vicino in questa direzione.</div>';
        return;
      }}
      neighbors.slice(0, 60).forEach(nb => {{
        const n = resolveNode(nb.id);
        if (!n) return;
        const item = document.createElement('div');
        item.className = 'nav-item';
        item.title = n.fp;
        item.innerHTML = `
          <span class="nav-dot" style="background:${{n.color}}"></span>
          <span class="nav-name">${{n.name}}</span>
          <span class="nav-rel">${{nb.relation || ''}}</span>
        `;
        item.addEventListener('click', () => {{
          const prevNode = resolveNode(currentNodeId);
          if (prevNode) navHistory.push({{ id: prevNode.id, name: prevNode.name }});
          openDetails(n, true);
          focusNode(n);
        }});
        list.appendChild(item);
      }});
      if (neighbors.length > 60) {{
        const more = document.createElement('div');
        more.id = 'nav-empty';
        more.textContent = `... e altri ${{neighbors.length - 60}} nodi`;
        list.appendChild(more);
      }}
    }}

    function navBack() {{
      if (navHistory.length === 0) return;
      const prev = navHistory.pop();
      const n = resolveNode(prev.id);
      if (n) {{
        openDetails(n, true);
        focusNode(n);
      }}
    }}

    // ── Details panel ──
    function openDetails(node, keepHistory) {{
      if (!keepHistory) navHistory = [];
      currentNodeId = node.id;

      document.getElementById('details-title').textContent = node.name;
      document.getElementById('meta-type').textContent = node.type.toUpperCase();
      document.getElementById('meta-type').style.color = node.color;
      document.getElementById('meta-file').textContent = `${{node.fp}} (L${{node.sl}}-L${{node.el}})`;
      document.getElementById('meta-docstring').textContent = node.doc || 'Nessuna docstring specificata.';

      // back button
      const backEl = document.getElementById('nav-back');
      if (navHistory.length > 0) {{
        const prev = navHistory[navHistory.length - 1];
        document.getElementById('nav-back-label').textContent = '← ' + prev.name;
        backEl.style.display = 'flex';
      }} else {{
        backEl.style.display = 'none';
      }}

      // neighbor list
      renderNavList(currentNodeId);

      // AST calls
      const callsContainer = document.getElementById('meta-calls');
      callsContainer.innerHTML = '';
      if (node.calls && node.calls.length > 0) {{
        node.calls.forEach(c => {{
          const span = document.createElement('span');
          span.className = 'tag';
          span.textContent = c;
          callsContainer.appendChild(span);
        }});
      }} else {{
        callsContainer.innerHTML = '<span style="color: #6e7681; font-size: 11px;">Nessuna chiamata esterna</span>';
      }}

      const depsContainer = document.getElementById('meta-deps');
      depsContainer.innerHTML = '';
      if (node.deps && node.deps.length > 0) {{
        node.deps.forEach(d => {{
          const span = document.createElement('span');
          span.className = 'tag';
          span.style.color = '#79c0ff';
          span.textContent = d;
          depsContainer.appendChild(span);
        }});
      }} else {{
        depsContainer.innerHTML = '<span style="color: #6e7681; font-size: 11px;">Nessuna dipendenza diretta</span>';
      }}

      // Codice caricato dal codeStore (lazy) — nessun impatto sul JSON principale
      document.getElementById('meta-code').textContent = codeStore[node.id] || '(codice non disponibile)';
      document.getElementById('details-panel').style.display = 'flex';
    }}

    function closeDetails() {{
      document.getElementById('details-panel').style.display = 'none';
      currentNodeId = null;
      navHistory = [];
    }}

    // ── Ricerca simboli e zoom ──
    document.getElementById('search-input').addEventListener('input', (e) => {{
      const query = e.target.value.toLowerCase().trim();
      if (!query) return;

      const matched = gData.nodes.find(n =>
        n.name.toLowerCase().includes(query) ||
        n.fp.toLowerCase().includes(query)
      );

      if (matched) {{
        navHistory = [];
        openDetails(matched, false);
        focusNode(matched);
      }}
    }});
  </script>
</body>
</html>"""


        path = Path(output_path).resolve()
        with open(path, "w", encoding="utf-8") as f:
            f.write(html_template)
        return str(path)

