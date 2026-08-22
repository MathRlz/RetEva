/* Shared DAG rendering for the builder (edit mode) and the Config&Run preview
 * (read-only). One rendering architecture, two modes — see evaluator-architecture.md §12.1.
 * No build step: plain script exposing window.DagView. */
(function () {
  'use strict';

  const DRAWFLOW_JS = 'https://cdn.jsdelivr.net/npm/drawflow@0.0.59/dist/drawflow.min.js';
  const DRAWFLOW_CSS = 'https://cdn.jsdelivr.net/npm/drawflow@0.0.59/dist/drawflow.min.css';

  // Plumbing artifacts the simplified view hides from a node's input ports — derived bundles /
  // ordering signals, not user-meaningful data (mirrors pipeline STRUCTURAL_ARTIFACTS). Hidden
  // only when nothing in the CURRENTLY-RENDERED graph produces them: so the judge reads
  // "answer + documents" in the simplified graph / builder, but the full DAG (where
  // build_query_traces + report ARE drawn) keeps those ports and their edges.
  const STRUCTURAL_ARTIFACTS = ['metrics', 'query_traces'];
  let producedInView = null;  // Set of artifacts the current drawGraph render produces

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
    const ports = Array.isArray(spec.input_ports) ? spec.input_ports
      : (spec.inputs || []).map(a => ({ label: a, names: [a], optional: false }))
        .concat((spec.optional_inputs || []).map(a =>
          ({ label: a, names: [a], optional: true })));
    if (!producedInView) return ports;  // no render context yet → show all
    // Drop a port that carries ONLY structural plumbing none of the rendered nodes produces
    // (e.g. the judge's metrics/query_traces in the simplified graph). Edge-safe: edges match
    // by artifact name, and the dropped ports have no producer to draw an edge from.
    return ports.filter(p => !p.names.every(
      n => STRUCTURAL_ARTIFACTS.indexOf(n) >= 0 && !producedInView.has(n)));
  }

  /* Node card: id, model line, labeled port columns (optional inputs italic + ?).
   * `columns` ([{name, type, dtype}]) renders a dataset node's declared schema, so the
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
          `<span title="${c.name} → ${c.artifact || ''} (${c.dtype || '?'})">${c.name}:${c.type}</span>`
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
    // A carried contract is authoritative: only fall back to the catalogue tile when the key is
    // truly ABSENT (undefined). An explicit `null` is meaningful — e.g. the model-free tts node
    // resolves `family: null`; coalescing that null to a base default once made a loaded
    // model-free node render a model picker.
    const pick = (k, d) => (n[k] !== undefined ? n[k]
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

  /* Measure each node's REAL rendered footprint (content-driven width + the variable height that
   * ports / ground-truth rows / dataset columns add) by rendering its card off-screen with the
   * SAME classes + filtered ports the canvas uses. Layout then spaces cards by actual size, not an
   * estimate — which is what stops a tall card from overlapping its neighbour. Returns {id:{w,h}}.
   * Must run AFTER producedInView is set (so measured ports == rendered ports). */
  function measureNodes(nodes, catalogue) {
    const host = document.createElement('div');
    host.className = 'drawflow';  // so `.drawflow .drawflow-node` CSS (max-content width) applies
    host.style.cssText = 'position:absolute; visibility:hidden; left:-99999px; top:0;';
    document.body.appendChild(host);
    const sizes = {};
    (nodes || []).forEach(n => {
      const spec = effectiveSpec(n, catalogue);
      const el = document.createElement('div');
      el.className = 'drawflow-node cat-' + (spec.category || 'transform');
      el.style.position = 'static';  // flow in the host so each reports its own content box
      el.innerHTML = '<div class="drawflow_content_node">' +
        nodeHtml(n.id, spec, n.params || {}, n.columns, spec.label) + '</div>';
      host.appendChild(el);
      sizes[n.id] = { w: el.offsetWidth || 170, h: el.offsetHeight || 120 };
    });
    document.body.removeChild(host);
    return sizes;
  }

  /* Level-based layout: x per topological level (advanced by the level's widest MEASURED card), y
   * by stacking each row's MEASURED height + a gap (so cards never overlap, whatever their size).
   * Rows within a level are ordered by the barycenter of their producers' rows (crossing
   * reduction). `sizes` = measureNodes() output; falls back to a default box if a node is absent. */
  function layoutByLevels(levels, nodes, sizes) {
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
    const sz = id => (sizes && sizes[id]) || { w: 170, h: 120 };
    const pos = {};
    let x = 30;
    ordered.forEach(lvl => {
      let widest = 170, y = 30;
      lvl.forEach(id => {
        pos[id] = { x, y };
        y += sz(id).h + 30;                  // next row below this card + a gap (no overlap)
        widest = Math.max(widest, sz(id).w);
      });
      x += widest + 90;                      // next level past the widest card + a gutter
    });
    return pos;
  }

  /* Edge triples [artifact, producer, isFallback] to draw for a consumer: EVERY binding is a
   * real data dependency and gets an edge (hiding one left the earlier producer's port looking
   * dangling — e.g. dataset_source.query_text feeding text_embedding as the fallback under
   * asr's hypothesis). At runtime the NEWEST bound producer per artifact wins, so the last
   * binding is the primary edge and earlier same-artifact bindings render as faded fallbacks. */
  function edgePairs(node) {
    const bindings = node.bindings || [];
    const newest = {};
    bindings.forEach(([art, prod]) => { newest[art] = prod; }); // last wins = runtime read
    return bindings.map(([art, prod]) => [art, prod, newest[art] !== prod]);
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
   * anchor. Suppressed on short edges — the port labels already
   * tell the story there. Appended to the transformed precanvas so they pan/zoom along. */
  function decorateEdge(editor, art, fromDf, outIdx, toDf, inIdx, optional) {
    const conn = editor.precanvas && editor.precanvas.querySelector(
      `.connection.node_in_node-${toDf}.node_out_node-${fromDf}` +
      `.output_${outIdx}.input_${inIdx}`);
    // a fallback edge (earlier producer of a multi-producer input) stays quiet: no label/arrow
    if (conn && conn.classList.contains('dag-fallback-edge')) return;
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
    // Which artifacts the nodes being rendered produce — drives inputPorts' structural-port
    // hiding (a metrics/query_traces port stays only when its producer is on screen).
    producedInView = new Set();
    (graph.nodes || []).forEach(n => (effectiveSpec(n, catalogue).outputs || [])
      .forEach(o => producedInView.add(o)));
    // Measure real card sizes FIRST (with producedInView set, so measured ports == rendered ports),
    // then place by measured footprint so cards never overlap.
    const sizes = measureNodes(graph.nodes || [], catalogue);
    const pos = layoutByLevels(graph.levels || [], graph.nodes || [], sizes);
    const dfIds = {};
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
      edgePairs(n).forEach(([art, prod, isFallback]) => {
        const pNode = (graph.nodes || []).find(x => x.id === prod);
        const pSpec = pNode && effectiveSpec(pNode, catalogue);
        if (!pSpec || !dfIds[prod]) return;
        const outIdx = pSpec.outputs.indexOf(art) + 1;
        // a collapsed port accepts any of its OneOf alternatives → match by membership
        const inIdx = ports.findIndex(p => p.names.indexOf(art) >= 0) + 1;
        if (outIdx > 0 && inIdx > 0) {
          try {
            editor.addConnection(dfIds[prod], dfIds[n.id],
                                 'output_' + outIdx, 'input_' + inIdx);
            if (isFallback) {
              // the runtime reads the newest producer; earlier bindings are real-but-fallback
              const conn = editor.precanvas && editor.precanvas.querySelector(
                `.connection.node_in_node-${dfIds[n.id]}.node_out_node-${dfIds[prod]}` +
                `.output_${outIdx}.input_${inIdx}`);
              if (conn) conn.classList.add('dag-fallback-edge');
            }
          } catch (e) { /* tolerate duplicate/odd connections */ }
        }
      });
    });
    // Port alignment + edge decorations are position-dependent: recompute once the layout
    // has settled (fonts/CSS can shift content-sized cards after the synchronous draw)…
    const realign = () => (graph.nodes || []).forEach(n => {
      if (dfIds[n.id]) portTitles(dfIds[n.id], effectiveSpec(n, catalogue));
    });
    const refreshAll = () => { realign(); refreshEdges(editor); };
    editor._dagRefresh = () => refreshEdges(editor);  // drag / connection add+remove
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

  /* The editor's live connection store for the current module (Drawflow keeps it here; the
   * builder reads the same path in exportSpec). Null before the editor is started. */
  function moduleData(editor) {
    const df = editor.drawflow && editor.drawflow.drawflow;
    const mod = df && (df[editor.module] || df.Home);
    return (mod && mod.data) || null;
  }

  /* The current connections as decoration descriptors, derived LIVE from the connection store —
   * so an interactively added edge gains a label and a deleted one loses it (no static list to
   * drift). `art` = the producer's artifact on that output port; `optional` = consumer port is
   * optional. A connection is `{node: toDf, output: 'input_N'}` under `outputs.output_M`. */
  function liveEdges(editor) {
    const data = moduleData(editor);
    const out = [];
    if (!data) return out;
    Object.keys(data).forEach(fromDf => {
      const node = data[fromDf];
      const spec = node.data && node.data._form;
      const outs = node.outputs || {};
      Object.keys(outs).forEach(outKey => {
        const outIdx = parseInt(outKey.split('_')[1], 10);
        const art = (spec && spec.outputs && spec.outputs[outIdx - 1]) || '';
        (outs[outKey].connections || []).forEach(c => {
          const toDf = String(c.node);
          const inIdx = parseInt(String(c.output).split('_')[1], 10);
          const toSpec = data[toDf] && data[toDf].data && data[toDf].data._form;
          const ports = toSpec ? inputPorts(toSpec) : [];
          const optional = !!(ports[inIdx - 1] && ports[inIdx - 1].optional);
          out.push({ art, fromDf, outIdx, toDf, inIdx, optional });
        });
      });
    });
    return out;
  }

  /* Recompute every connection path from the current node geometry, then redraw the edge
   * decorations (labels/arrows) from the LIVE connections. Idempotent. */
  function refreshEdges(editor) {
    const data = moduleData(editor);
    if (data) {
      Object.keys(data).forEach(id => {
        try { editor.updateConnectionNodes('node-' + id); } catch (e) { /* noop */ }
      });
    }
    if (editor.precanvas) {
      editor.precanvas.querySelectorAll('.dag-edge-label, .dag-edge-arrow')
        .forEach(el => el.remove());
    }
    liveEdges(editor).forEach(e =>
      decorateEdge(editor, e.art, e.fromDf, e.outIdx, e.toDf, e.inIdx, e.optional));
  }

  /* ── Zoom & pan ─────────────────────────────────────────────────────── */

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
    ensureDrawflow, nodeHtml, portTitles, inputPorts, drawGraph,
    attachZoom, zoomFit,
  };
})();
