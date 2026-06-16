/* Shared DAG rendering for the builder (edit mode) and the Config&Run preview
 * (read-only). One rendering architecture, two modes — see BUILDER_UX.md.
 * No build step: plain script exposing window.DagView. */
(function () {
  'use strict';

  const DRAWFLOW_JS = 'https://cdn.jsdelivr.net/npm/drawflow@0.0.59/dist/drawflow.min.js';
  const DRAWFLOW_CSS = 'https://cdn.jsdelivr.net/npm/drawflow@0.0.59/dist/drawflow.min.css';

  /* Load Drawflow once (pages that already include it skip the fetch). */
  function ensureDrawflow(cb, onError) {
    if (window.Drawflow) { cb(); return; }
    let pending = 0;
    const arm = () => { pending += 1; };
    const done = () => { pending -= 1; if (pending === 0) cb(); };
    let l = document.querySelector(`link[href="${DRAWFLOW_CSS}"]`);
    if (!l) {
      // wait for the stylesheet too: connection paths are computed from the *styled*
      // port positions — drawing before the CSS lands anchors edges wrong
      l = document.createElement('link');
      l.rel = 'stylesheet'; l.href = DRAWFLOW_CSS;
      arm();
      l.addEventListener('load', done);
      l.addEventListener('error', done); // tolerate css failure; js alone still works
      document.head.appendChild(l);
    }
    let s = document.querySelector(`script[src="${DRAWFLOW_JS}"]`);
    if (!s) {
      s = document.createElement('script');
      s.src = DRAWFLOW_JS;
      document.head.appendChild(s);
    }
    arm();
    s.addEventListener('load', done);
    if (onError) s.addEventListener('error', onError);
    if (window.Drawflow) cb(); // raced: already loaded between checks
  }

  /* Input ports for a spec: prefer the server's collapsed `input_ports` (a OneOf chain —
   * query-text / query-vectors — is ONE port that accepts any alternative, not N dangling
   * ports); else one port per flat input/optional_input (palette drops / legacy specs). */
  function inputPorts(spec) {
    if (Array.isArray(spec.input_ports)) return spec.input_ports;
    return (spec.inputs || []).map(a => ({ label: a, names: [a], optional: false }))
      .concat((spec.optional_inputs || []).map(a =>
        ({ label: a, names: [a], optional: true })));
  }

  /* Node card: id, model line, labeled port columns (optional inputs italic + ?).
   * `columns` ([{name, type}]) renders a dataset node's declared schema, so the
   * diagram reads what data an experiment consumes. */
  function nodeHtml(id, spec, params, columns, label) {
    const ins = inputPorts(spec).map(p => p.optional
      ? `<span class="opt" title="${p.names.join(' | ')} (optional)">${p.label}?</span>`
      : `<span title="${p.names.join(' | ')}">${p.label}</span>`).join('');
    const outs = spec.outputs.map(a => `<span title="${a}">${a}</span>`).join('');
    const model = params && params.model
      ? params.model + (params.size ? ' · ' + params.size : '') : '';
    const cols = (columns && columns.length)
      ? `<div class="node-columns">` + columns.map(c =>
          `<span title="${c.name} → ${c.artifact || ''}">${c.name}:${c.type}</span>`
        ).join('') + `</div>`
      : '';
    const title = label || id;
    return `<div class="node-title">${title}</div>` +
           (title !== id ? `<div class="node-id">${id}</div>` : '') +
           `<div class="node-model">${model}</div>` + cols +
           `<div class="node-io"><div class="col incol">${ins}</div>` +
           `<div class="col out">${outs}</div></div>`;
  }

  /* Label + ALIGN each port dot: tooltip = the artifact(s) it carries, and the dot is moved
   * vertically to sit beside its name row (the in-flow names drive node width; the dot
   * straddles the node border at the name's height). Drawflow draws connections from the dot
   * positions, so edges follow the aligned dots. Re-run after any content/layout change. */
  function portTitles(dfId, spec) {
    const el = document.getElementById('node-' + dfId);
    if (!el) return;
    const inSpans = el.querySelectorAll('.node-io .incol > span');
    const outSpans = el.querySelectorAll('.node-io .out > span');
    const place = (dot, span, side) => {
      if (!dot || !span) return;
      dot.style.position = 'absolute';
      dot.style.margin = '0';
      dot.style.bottom = 'auto';
      dot.style.top = (span.offsetTop + span.offsetHeight / 2) + 'px';
      if (side === 'left') {
        dot.style.left = '0'; dot.style.right = 'auto';
        dot.style.transform = 'translate(-50%, -50%)';
      } else {
        dot.style.right = '0'; dot.style.left = 'auto';
        dot.style.transform = 'translate(50%, -50%)';
      }
    };
    inputPorts(spec).forEach((port, i) => {
      const dot = el.querySelector('.input_' + (i + 1));
      if (dot) dot.title = port.names.join(' | ') + (port.optional ? ' (optional)' : '');
      place(dot, inSpans[i], 'left');
    });
    spec.outputs.forEach((a, i) => {
      const dot = el.querySelector('.output_' + (i + 1));
      if (dot) dot.title = a;
      place(dot, outSpans[i], 'right');
    });
  }

  /* The contract used to render ONE node. A node may carry its own field-aware contract
   * (preview / seed / template — ports + label + model family resolved for its discriminator
   * fields), so an operator instance renders as itself (corpus_embedding ≠ text_embedding).
   * Palette drops carry no contract → fall back to the static catalogue tile (operator
   * default). Used everywhere drawing keys on a spec (layout, ports, edges, param form). */
  function effectiveSpec(n, catalogue) {
    const base = (catalogue && catalogue[n.type || n.stage]) ||
                 { inputs: [], outputs: [], optional_inputs: [], category: 'transform' };
    const pick = (k, d) => (n[k] !== undefined && n[k] !== null ? n[k]
                            : (base[k] !== undefined ? base[k] : d));
    return {
      type: n.type || n.stage,
      label: n.label || base.label,
      category: n.category || base.category,
      domain: n.domain || base.domain,
      inputs: pick('inputs', []),
      outputs: pick('outputs', []),
      optional_inputs: pick('optional_inputs', []),
      input_ports: Array.isArray(n.input_ports) ? n.input_ports
                   : (Array.isArray(base.input_ports) ? base.input_ports : undefined),
      family: pick('family', null),
      model_field: pick('model_field', null),
      node_params: pick('node_params', []),
    };
  }

  /* Estimated card width (px) for level spacing — content-driven sizing means the
   * real width is only known after render; this tracks the CSS (10px labels, 18-char
   * ellipsis cap, two port columns) closely enough for column placement. */
  function estimateWidth(spec, params) {
    const cap = s => Math.min((s || '').length, 18);
    const inMax = Math.max(0,
      ...inputPorts(spec).map(p => cap(p.label) + (p.optional ? 1 : 0)));
    const outMax = Math.max(0, ...spec.outputs.map(cap));
    const model = params && params.model
      ? cap(params.model + (params.size ? ' · ' + params.size : '')) : 0;
    return Math.max(170, (inMax + outMax) * 6 + 48, model * 7 + 24, cap(spec.type) * 8);
  }

  /* Level-based layout: x per topological level (spacing from the widest card in the
   * previous level), y per row. Rows within a level are ordered by the barycenter of
   * their producers' rows (one-pass crossing reduction — BUILDER_UX.md §3.4). */
  function layoutByLevels(levels, nodes, catalogue) {
    const byId = {};
    (nodes || []).forEach(n => { byId[n.id] = n; });
    const rowOf = {};
    const ordered = levels.map((lvl, li) => {
      if (li === 0 || !nodes) {
        lvl.forEach((id, ri) => { rowOf[id] = ri; });
        return lvl.slice();
      }
      const scored = lvl.map((id, idx) => {
        const prods = (byId[id] && byId[id].bindings || [])
          .map(b => rowOf[b[1]]).filter(r => r !== undefined);
        const score = prods.length
          ? prods.reduce((a, b) => a + b, 0) / prods.length : idx;
        return { id, idx, score };
      });
      scored.sort((a, b) => a.score - b.score || a.idx - b.idx);
      scored.forEach((s, ri) => { rowOf[s.id] = ri; });
      return scored.map(s => s.id);
    });
    const pos = {};
    let x = 30;
    ordered.forEach(lvl => {
      let widest = 150;
      lvl.forEach((id, ri) => {
        pos[id] = { x, y: 30 + ri * 135 };
        const n = byId[id];
        if (n) widest = Math.max(widest, estimateWidth(effectiveSpec(n, catalogue), n.params));
      });
      x += widest + 90;  // card + gutter
    });
    return pos;
  }

  /* Edge pairs to draw for a consumer: newest bound producer per artifact —
   * except merge nodes (fusion), where every producer feeds the result. */
  function edgePairs(node) {
    const type = node.type || node.stage;
    if (type === 'fusion') return (node.bindings || []);
    const newest = {};
    (node.bindings || []).forEach(([art, prod]) => { newest[art] = prod; }); // last wins
    return Object.entries(newest);
  }

  /* Center of a rendered port dot in precanvas coordinates. */
  function portCenter(dfId, selector) {
    const node = document.getElementById('node-' + dfId);
    const port = node && node.querySelector(selector);
    if (!node || !port) return null;
    return {
      x: parseFloat(node.style.left || 0) + port.offsetLeft + port.offsetWidth / 2,
      y: parseFloat(node.style.top || 0) + port.offsetTop + port.offsetHeight / 2,
    };
  }

  /* Artifact label at the edge midpoint (halo pill) + arrow glyph at the consumer
   * anchor (BUILDER_UX.md §3.4). Suppressed on short edges — the port labels already
   * tell the story there. Appended to the transformed precanvas so they pan/zoom along. */
  function decorateEdge(editor, art, fromDf, outIdx, toDf, inIdx, optional) {
    const conn = editor.precanvas && editor.precanvas.querySelector(
      `.connection.node_in_node-${toDf}.node_out_node-${fromDf}` +
      `.output_${outIdx}.input_${inIdx}`);
    if (optional) {
      // de-emphasize: ordering/GT side-channels must not drown the data spine
      if (conn) conn.classList.add('dag-opt-edge');
      return;
    }
    const a = portCenter(fromDf, '.output_' + outIdx);
    const b = portCenter(toDf, '.input_' + inIdx);
    if (!a || !b || !editor.precanvas) return;
    const dist = Math.hypot(b.x - a.x, b.y - a.y);
    const arrow = document.createElement('div');
    arrow.className = 'dag-edge-arrow';
    arrow.textContent = '▶';
    arrow.style.left = (b.x - 14) + 'px';
    arrow.style.top = b.y + 'px';
    editor.precanvas.appendChild(arrow);
    if (dist < 70) return;
    const label = document.createElement('div');
    label.className = 'dag-edge-label';
    label.textContent = art;
    label.style.left = ((a.x + b.x) / 2) + 'px';
    label.style.top = ((a.y + b.y) / 2) + 'px';
    editor.precanvas.appendChild(label);
  }

  /* Render a whole graph {nodes:[{id, type|stage, params?, bindings?}], levels}
   * onto a Drawflow editor. Returns {nodeId: drawflowId}. */
  function drawGraph(editor, graph, catalogue) {
    const pos = layoutByLevels(graph.levels || [], graph.nodes || [], catalogue);
    const dfIds = {};
    const edges = [];
    (graph.nodes || []).forEach(n => {
      const type = n.type || n.stage;
      // skip genuinely-unknown nodes (no catalogue tile AND no carried contract)
      if (!(catalogue && catalogue[type]) && !n.inputs && !n.outputs) return;
      const spec = effectiveSpec(n, catalogue);
      const p = pos[n.id] || { x: 30, y: 30 };
      const nIn = inputPorts(spec).length;
      // category (source/model/transform/metric/sink) → a `cat-<category>` class so the
      // stylesheet colors each node card by its declared class (app.css).
      const cls = type + ' cat-' + (spec.category || 'transform');
      // stash the resolved contract on the node so the builder's param form is field-aware
      // immediately (formFor reads node.data._form) without a re-fetch.
      dfIds[n.id] = editor.addNode(n.id, nIn, spec.outputs.length, p.x, p.y, cls,
                                   { type, params: n.params || {}, _form: spec },
                                   nodeHtml(n.id, spec, n.params || {}, n.columns,
                                            spec.label));
      portTitles(dfIds[n.id], spec);
    });
    (graph.nodes || []).forEach(n => {
      const type = n.type || n.stage;
      if (!dfIds[n.id]) return;
      const spec = effectiveSpec(n, catalogue);
      const ports = inputPorts(spec);
      edgePairs(n).forEach(([art, prod]) => {
        const pNode = (graph.nodes || []).find(x => x.id === prod);
        const pSpec = pNode && effectiveSpec(pNode, catalogue);
        if (!pSpec || !dfIds[prod]) return;
        const outIdx = pSpec.outputs.indexOf(art) + 1;
        // a collapsed port accepts any of its OneOf alternatives → match by membership
        const inIdx = ports.findIndex(p => p.names.indexOf(art) >= 0) + 1;
        if (outIdx > 0 && inIdx > 0) {
          const optional = ports[inIdx - 1].optional;  // wired via an optional port
          try {
            editor.addConnection(dfIds[prod], dfIds[n.id],
                                 'output_' + outIdx, 'input_' + inIdx);
            edges.push({ art, fromDf: dfIds[prod], outIdx,
                         toDf: dfIds[n.id], inIdx, optional });
          } catch (e) { /* tolerate duplicate/odd connections */ }
        }
      });
    });
    // Port alignment + edge decorations are position-dependent: recompute once the layout
    // has settled (fonts/CSS can shift content-sized cards after the synchronous draw)…
    const realign = () => (graph.nodes || []).forEach(n => {
      if (dfIds[n.id]) portTitles(dfIds[n.id], effectiveSpec(n, catalogue));
    });
    const refreshAll = () => { realign(); refreshEdges(editor, dfIds, edges); };
    editor._dagRefresh = () => refreshEdges(editor, dfIds, edges);  // drag: edges only
    if (window.requestAnimationFrame) window.requestAnimationFrame(refreshAll);
    setTimeout(refreshAll, 150);
    // …and again while a node is dragged, so edge labels/arrows track the moving node
    // (Drawflow moves the connection SVG live but not our custom decorations). Attach once.
    watchNodeDrag(editor);
    return dfIds;
  }

  /* Re-decorate edges while a node is dragged (and once on release). Drawflow sets
   * `editor.drag` true during a node drag; throttle to one redraw per frame. Idempotent —
   * attaches its listeners only once per editor. */
  function watchNodeDrag(editor) {
    if (editor._dagDragWatched) return;
    editor._dagDragWatched = true;
    const el = editor.container;
    if (!el) return;
    let queued = false;
    const live = () => {
      if (!editor.drag || !editor._dagRefresh || queued) return;
      queued = true;
      (window.requestAnimationFrame || setTimeout)(() => {
        queued = false;
        editor._dagRefresh();
      });
    };
    const settle = () => {
      if (editor._dagRefresh) (window.requestAnimationFrame || setTimeout)(editor._dagRefresh);
    };
    el.addEventListener('mousemove', live);
    el.addEventListener('mouseup', settle);
    el.addEventListener('touchmove', live);
    el.addEventListener('touchend', settle);
  }

  /* Recompute every connection path from the current node geometry, then redraw the
   * edge decorations (labels/arrows) from the settled positions. Idempotent. */
  function refreshEdges(editor, dfIds, edges) {
    Object.values(dfIds).forEach(id => {
      try { editor.updateConnectionNodes('node-' + id); } catch (e) { /* noop */ }
    });
    if (editor.precanvas) {
      editor.precanvas.querySelectorAll('.dag-edge-label, .dag-edge-arrow')
        .forEach(el => el.remove());
    }
    edges.forEach(e =>
      decorateEdge(editor, e.art, e.fromDf, e.outIdx, e.toDf, e.inIdx, e.optional));
  }

  /* Read-only inspect panel: stage, artifacts, configured model + params. */
  function renderInspect(panel, info) {
    let rows = `<dt>node</dt><dd>${info.label || info.id} ` +
               `<small>${info.id}${info.stage ? ' · ' + info.stage : ''}</small></dd>`;
    rows += `<dt>artifacts</dt><dd>${(info.inputs || []).join(', ') || '·'} → ` +
            `${(info.outputs || []).join(', ') || '·'}</dd>`;
    if (info.columns && info.columns.length) {
      rows += `<dt>columns</dt><dd>${info.columns
        .map(c => `${c.name}: ${c.type}`).join(', ')}</dd>`;
    }
    Object.entries(info.config || {}).forEach(([k, v]) => {
      rows += `<dt>${k}</dt><dd>${typeof v === 'object' ? JSON.stringify(v) : v}</dd>`;
    });
    panel.innerHTML = `<dl class="dag-detail">${rows}</dl>`;
  }

  /* ── Zoom & pan (BUILDER_UX.md §3.4 / B4c) ──────────────────────────── */

  function updateZoomClass(editor, canvasEl) {
    /* edge artifact labels hide below 75% zoom (CSS .dag-zoom-small) */
    canvasEl.classList.toggle('dag-zoom-small', (editor.zoom || 1) < 0.75);
  }

  /* Scale + translate so the whole DAG is visible (capped at 100%). */
  function zoomFit(editor, canvasEl) {
    const nodes = canvasEl.querySelectorAll('.drawflow-node');
    if (!nodes.length) return;
    let maxX = 0, maxY = 0;
    nodes.forEach(n => {
      maxX = Math.max(maxX, parseFloat(n.style.left || 0) + n.offsetWidth);
      maxY = Math.max(maxY, parseFloat(n.style.top || 0) + n.offsetHeight);
    });
    const scale = Math.min(
      canvasEl.clientWidth / (maxX + 50),
      canvasEl.clientHeight / (maxY + 50),
      1
    );
    editor.canvas_x = 0;
    editor.canvas_y = 0;
    editor.zoom = Math.max(scale, editor.zoom_min || 0.3);
    editor.zoom_refresh();
    updateZoomClass(editor, canvasEl);
  }

  /* Mouse-wheel scroll-in/out + corner controls (+ / − / fit). Pan stays Drawflow's
   * native canvas drag (available in edit and fixed modes alike). */
  function attachZoom(editor, canvasEl) {
    editor.zoom_min = 0.2;  // let zoom-fit shrink wide graphs below Drawflow's 0.5
    canvasEl.addEventListener('wheel', ev => {
      ev.preventDefault();
      if (ev.deltaY < 0) editor.zoom_in(); else editor.zoom_out();
      updateZoomClass(editor, canvasEl);
    }, { passive: false });
    const bar = document.createElement('div');
    bar.className = 'dag-zoom-controls';
    [
      ['+', () => { editor.zoom_in(); updateZoomClass(editor, canvasEl); }],
      ['−', () => { editor.zoom_out(); updateZoomClass(editor, canvasEl); }],
      ['⤢', () => zoomFit(editor, canvasEl)],
    ].forEach(([txt, fn]) => {
      const b = document.createElement('button');
      b.type = 'button'; b.textContent = txt; b.title = txt === '⤢' ? 'fit' : 'zoom';
      b.onclick = fn;
      bar.appendChild(b);
    });
    canvasEl.appendChild(bar);
  }

  window.DagView = {
    ensureDrawflow, nodeHtml, portTitles, inputPorts, layoutByLevels, drawGraph,
    renderInspect, attachZoom, zoomFit,
  };
})();
