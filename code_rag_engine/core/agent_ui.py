"""
Frammenti UI (CSS, HTML, JavaScript) del pannello "Chiedi al grafo" del visualizzatore 3D.

Sono stringhe semplici (NON f-string): le parentesi graffe non vanno raddoppiate.
Vengono inserite da Graph3DVisualizer.render_html(); il pannello compare solo se viene
indicato un agent_endpoint (cioè quando il viewer è servito da serve_graph.py).
"""

AGENT_CSS = r"""
    /* ── Pannello agente ── */
    #agent-panel {
      position: absolute;
      left: 50%;
      bottom: 15px;
      transform: translateX(-50%);
      width: min(640px, calc(100vw - 800px));
      min-width: 420px;
      z-index: 110;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }
    #agent-bar {
      display: flex;
      align-items: center;
      gap: 8px;
      background: rgba(22, 27, 34, 0.92);
      backdrop-filter: blur(8px);
      border: 1px solid #30363d;
      border-radius: 10px;
      padding: 8px 10px;
      box-shadow: 0 8px 24px rgba(0,0,0,0.5);
    }
    #agent-bar:focus-within {
      border-color: #58a6ff;
      box-shadow: 0 0 0 2px rgba(88,166,255,0.2), 0 8px 24px rgba(0,0,0,0.5);
    }
    #agent-icon { color: #ffd33d; font-size: 16px; }
    #agent-input {
      flex: 1;
      background: transparent;
      border: none;
      outline: none;
      color: #f0f6fc;
      font-size: 13px;
    }
    #agent-bar button {
      background: #238636;
      color: #fff;
      border: 1px solid rgba(240,246,252,0.1);
      border-radius: 6px;
      padding: 5px 12px;
      font-size: 12px;
      cursor: pointer;
    }
    #agent-bar button:hover { background: #2ea043; }
    #agent-bar button:disabled { opacity: 0.5; cursor: default; }
    #agent-bar button#agent-clear { background: #21262d; color: #c9d1d9; padding: 5px 9px; }
    #agent-bar button#agent-clear:hover { background: #30363d; }
    #agent-out {
      display: none;
      max-height: 42vh;
      overflow-y: auto;
      background: rgba(22, 27, 34, 0.95);
      backdrop-filter: blur(10px);
      border: 1px solid #30363d;
      border-radius: 10px;
      padding: 12px 14px;
      box-shadow: 0 8px 24px rgba(0,0,0,0.5);
      font-size: 13px;
      line-height: 1.5;
    }
    .agent-answer code, .agent-step .agent-name {
      font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
    }
    .agent-answer code {
      background: #21262d;
      border: 1px solid #30363d;
      border-radius: 4px;
      padding: 0 4px;
      font-size: 12px;
      color: #7ee787;
    }
    .agent-mode { margin-top: 6px; font-size: 11px; color: #6e7681; }
    .agent-status { color: #8b949e; font-style: italic; }
    .agent-status.err { color: #ff7b72; font-style: normal; }
    .agent-steps-title {
      margin: 12px 0 4px;
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      color: #8b949e;
    }
    .agent-step {
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 4px 6px;
      border-radius: 6px;
      cursor: pointer;
    }
    .agent-step:hover { background: #21262d; }
    .agent-rank {
      background: #ffd33d;
      color: #0d1117;
      border-radius: 50%;
      width: 18px;
      height: 18px;
      font-size: 11px;
      font-weight: 700;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      flex-shrink: 0;
    }
    .agent-dot { width: 9px; height: 9px; border-radius: 50%; flex-shrink: 0; }
    .agent-step .agent-name { font-size: 12px; color: #c9d1d9; white-space: nowrap; }
    .agent-role {
      font-size: 10px;
      color: #8b949e;
      border: 1px solid #30363d;
      border-radius: 10px;
      padding: 0 6px;
      flex-shrink: 0;
    }
    .agent-file {
      margin-left: auto;
      max-width: 42%;
      color: #6e7681;
      font-size: 11px;
      font-family: monospace;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    #instructions { max-width: 260px; }
    .agent-trace {
      margin-top: 10px;
      border-top: 1px solid #30363d;
      padding-top: 8px;
      color: #8b949e;
      font-size: 11px;
    }
    .agent-trace summary { cursor: pointer; color: #58a6ff; }
    .agent-trace-row { padding: 3px 0; }
    .agent-trace-edge { font-family: monospace; color: #c9d1d9; }
    .agent-context-item { cursor: pointer; border: 1px solid #30363d; border-radius: 5px; padding: 5px; margin-top: 4px; }
    .agent-context-item:hover { background: #21262d; }
    .agent-context-code { display: block; margin-top: 3px; color: #7ee787; white-space: pre-wrap; max-height: 90px; overflow: hidden; }
    .agent-turn { border-bottom: 1px solid #21262d; padding-bottom: 10px; margin-bottom: 10px; }
    .agent-history { color: #8b949e; font-size: 11px; margin-bottom: 8px; }
    .agent-history-question { color: #c9d1d9; margin-top: 4px; }
"""

AGENT_PANEL_HTML = r"""
  <div id="agent-panel">
    <div id="agent-out"></div>
    <div id="agent-bar">
      <span id="agent-icon">&#10022;</span>
      <input type="text" id="agent-input" autocomplete="off"
             placeholder="Chiedi al grafo… (es. come viene gestita la scadenza dei prodotti?)">
      <button id="agent-send">Chiedi</button>
      <button id="agent-clear" title="Rimuovi l'evidenziazione">&times;</button>
    </div>
  </div>
"""

AGENT_JS = r"""
    // ── Agente: evidenzia nel grafo la parte rilevante per la domanda ──
    const AGENT_ENDPOINT = __AGENT_ENDPOINT__;
    const ROLE_LABEL = {
      match: 'rilevante', callee: 'chiamato', caller: 'chiamante',
      next: 'poi', previous: 'prima',
      container: 'contenitore', member: 'membro', context: 'contesto agente'
    };

    function escapeHtml(s) {
      return String(s).replace(/[&<>"']/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));
    }
    function renderRich(text) {
      return escapeHtml(text)
        .replace(/`([^`]+)`/g, '<code>$1</code>')
        .replace(/\*\*([^*]+)\*\*/g, '<b>$1</b>')
        .replace(/\n/g, '<br>');
    }

    // Con un'evidenziazione dell'agente attiva, la X del pannello dettagli chiude solo il pannello
    function closePanel() {
      if (selectedId || !focusActive) { closeDetails(); return; }
      document.getElementById('details-panel').style.display = 'none';
      currentNodeId = null;
    }

    function ensureTypeVisible(node) {
      const g = groupOf(node.type);
      if (g && !enabledGroups[g]) {
        enabledGroups[g] = true;
        const cb = document.querySelector('input[data-group="' + g + '"]');
        if (cb) cb.checked = true;
        return true;
      }
      return false;
    }

    // Inquadra l'insieme di nodi evidenziati senza avvicinarsi troppo
    function fitNodes(ids) {
      const pts = ids.map(id => nodeById[id]).filter(n => n && n.x !== undefined);
      if (!pts.length) return;
      const c = {x: 0, y: 0, z: 0};
      pts.forEach(n => { c.x += n.x; c.y += n.y; c.z += n.z; });
      c.x /= pts.length; c.y /= pts.length; c.z /= pts.length;
      let radius = 0;
      pts.forEach(n => { radius = Math.max(radius, Math.hypot(n.x - c.x, n.y - c.y, n.z - c.z)); });
      const distance = Math.min(900, Math.max(200, radius * 2.4 + 100));
      const cam = Graph.cameraPosition();
      const dx = cam.x - c.x, dy = cam.y - c.y, dz = cam.z - c.z;
      const len = Math.hypot(dx, dy, dz) || 1;
      Graph.cameraPosition(
        {x: c.x + dx / len * distance, y: c.y + dy / len * distance, z: c.z + dz / len * distance},
        c, 1000
      );
    }

    // Sposta lo sguardo su un singolo nodo mantenendo l'evidenziazione
    function flyTo(node) {
      if (!node || node.x === undefined) return;
      const cam = Graph.cameraPosition();
      const dx = cam.x - node.x, dy = cam.y - node.y, dz = cam.z - node.z;
      const len = Math.hypot(dx, dy, dz) || 1;
      const d = 260;
      Graph.cameraPosition(
        {x: node.x + dx / len * d, y: node.y + dy / len * d, z: node.z + dz / len * d},
        node, 800
      );
    }

    // focusNodes: [{id, role, rank, reason}] -> accende questi nodi e gli archi tra loro
    function setFocus(focusNodes) {
      const valid = (focusNodes || []).filter(f => nodeById[f.id]);
      if (!valid.length) { closeDetails(); return; }

      selectedId = null;   // evita che applyFilters() chiuda qualcosa
      document.getElementById('details-panel').style.display = 'none';
      currentNodeId = null;
      navHistory = [];

      let changed = false;
      valid.forEach(f => { if (ensureTypeVisible(nodeById[f.id])) changed = true; });
      if (changed) applyFilters();

      focusActive = true;
      emphasisNodes.clear();
      highlightNodes.clear();
      highlightLinks.clear();
      valid.forEach(f => {
        highlightNodes.add(f.id);
        if (f.role === 'match') emphasisNodes.add(f.id);
      });
      gData.links.forEach(l => {
        if (highlightNodes.has(linkId(l.source)) && highlightNodes.has(linkId(l.target))) highlightLinks.add(l);
      });
      refreshGraphStyle();
      fitNodes(valid.map(f => f.id));
    }

    if (AGENT_ENDPOINT) {
      const agentInput = document.getElementById('agent-input');
      const agentSend = document.getElementById('agent-send');
      const agentClear = document.getElementById('agent-clear');
      const agentOut = document.getElementById('agent-out');
      const agentTranscript = [];

      function renderAnswer(data) {
        agentOut.innerHTML = '';
        agentOut.style.display = 'block';

        agentTranscript.push({question: data.question || '', answer: data.answer || ''});
        if (agentTranscript.length > 1) {
          const history = document.createElement('div');
          history.className = 'agent-history';
          history.textContent = 'Conversazione precedente';
          agentTranscript.slice(0, -1).forEach(turn => {
            const q = document.createElement('div');
            q.className = 'agent-history-question';
            q.textContent = 'Tu: ' + turn.question;
            history.appendChild(q);
          });
          agentOut.appendChild(history);
        }

        const turn = document.createElement('div');
        turn.className = 'agent-turn';
        const ans = document.createElement('div');
        ans.className = 'agent-answer';
        ans.innerHTML = renderRich(data.answer || '');
        turn.appendChild(ans);

        const mode = document.createElement('div');
        mode.className = 'agent-mode';
        mode.textContent = data.mode === 'llm'
          ? 'Risposta sintetizzata da LLM sui simboli trovati nel grafo'
          : 'Ricerca ibrida + grafo (nessun LLM configurato: imposta una chiave provider per una spiegazione discorsiva)';
        turn.appendChild(mode);
        agentOut.appendChild(turn);

        const trace = data.trace || {};
        const traceBox = document.createElement('details');
        traceBox.className = 'agent-trace';
        const summary = document.createElement('summary');
        summary.textContent = 'Come ho costruito la risposta';
        traceBox.appendChild(summary);
        (trace.steps || []).forEach(step => {
          const row = document.createElement('div');
          row.className = 'agent-trace-row';
          const icon = step.status === 'failed' ? '!' : '✓';
          row.textContent = icon + ' ' + step.tool + (step.count !== undefined ? ' (' + step.count + ')' : '') +
            (step.description ? ' — ' + step.description : '');
          if (step.status === 'failed') row.style.color = '#ff7b72';
          traceBox.appendChild(row);
        });
        const contextCount = (trace.context_nodes || []).length;
        const usedCount = (trace.used_nodes || []).length;
        const edgeCount = (trace.navigated_edges || []).length;
        const counts = document.createElement('div');
        counts.className = 'agent-trace-row';
        counts.textContent = 'Contesto passato: ' + contextCount +
          ' nodi · usati nella risposta: ' + usedCount + ' · archi navigati: ' + edgeCount;
        traceBox.appendChild(counts);
        (trace.navigated_edges || []).slice(0, 20).forEach(edge => {
          const row = document.createElement('div');
          row.className = 'agent-trace-row agent-trace-edge';
          row.textContent = edge.source_name + ' —[' + (edge.relations || []).join(', ') +
            ']→ ' + edge.target_name;
          traceBox.appendChild(row);
        });
        const promptTitle = document.createElement('div');
        promptTitle.className = 'agent-trace-row';
        promptTitle.textContent = 'Materiale passato al modello';
        traceBox.appendChild(promptTitle);
        (trace.prompt_context || []).forEach(item => {
          const row = document.createElement('div');
          row.className = 'agent-context-item';
          row.title = 'Apri questo nodo nel grafo';
          row.innerHTML = '<b>' + escapeHtml(item.name || item.id) + '</b> · ' +
            escapeHtml(item.role || 'context') + '<br>' +
            escapeHtml(item.file || '') + ':' + escapeHtml(item.start_line || 0) +
            '-' + escapeHtml(item.end_line || 0);
          if (item.content_excerpt) {
            const code = document.createElement('code');
            code.className = 'agent-context-code';
            code.textContent = item.content_excerpt;
            row.appendChild(code);
          }
          row.addEventListener('click', () => {
            const node = nodeById[item.id];
            if (node) { openDetails(node, false, true); flyTo(node); }
          });
          traceBox.appendChild(row);
        });
        if (trace.flow_context) {
          const flow = document.createElement('pre');
          flow.className = 'agent-trace-row';
          flow.textContent = trace.flow_context;
          traceBox.appendChild(flow);
        }
        turn.appendChild(traceBox);

        const focus = (data.focus || []).filter(f => nodeById[f.id]);
        if (!focus.length) return;

        const title = document.createElement('div');
        title.className = 'agent-steps-title';
        title.textContent = 'Simboli evidenziati (' + focus.length + ') — clicca per ispezionare';
        agentOut.appendChild(title);

        focus.forEach(f => {
          const n = nodeById[f.id];
          const row = document.createElement('div');
          row.className = 'agent-step';
          row.title = f.reason || '';
          row.innerHTML =
            (f.rank ? '<span class="agent-rank">' + f.rank + '</span>' : '') +
            '<span class="agent-dot" style="background:' + n.color + '"></span>' +
            '<span class="agent-name">' + escapeHtml(n.name) + '</span>' +
            '<span class="agent-role">' + (ROLE_LABEL[f.role] || f.role) + '</span>' +
            '<span class="agent-file">' + escapeHtml(n.fp) + ':' + n.sl + '</span>';
          row.addEventListener('click', () => {
            openDetails(n, false, true);   // apre il codice senza perdere l'evidenziazione
            flyTo(n);
          });
          agentOut.appendChild(row);
        });
      }

      async function askAgent() {
        const q = agentInput.value.trim();
        if (!q) return;
        agentSend.disabled = true;
        agentOut.style.display = 'block';
        agentOut.innerHTML = '<div class="agent-status">Cerco nella parte di grafo che riguarda la domanda…</div>';
        try {
          const res = await fetch(AGENT_ENDPOINT, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({question: q})
          });
          const data = await res.json().catch(() => ({}));
          if (!res.ok) throw new Error(data.error || ('HTTP ' + res.status));
          renderAnswer(data);
          const selected = data.focus || [];
          const selectedIds = new Set(selected.map(f => f.id));
          const context = ((data.trace || {}).context_nodes || [])
            .filter(id => !selectedIds.has(id))
            .map(id => ({id: id, role: 'context'}));
          setFocus(selected.concat(context));
        } catch (err) {
          agentOut.innerHTML = '<div class="agent-status err">Errore: ' + escapeHtml(err.message) + '</div>';
        } finally {
          agentSend.disabled = false;
        }
      }

      agentSend.addEventListener('click', askAgent);
      agentInput.addEventListener('keydown', e => { if (e.key === 'Enter') askAgent(); });
      agentClear.addEventListener('click', () => {
        closeDetails();
        agentOut.style.display = 'none';
      });
    }
"""
