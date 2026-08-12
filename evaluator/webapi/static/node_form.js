/* Shared per-node parameter form renderer.
 *
 * Used by the builder (Drawflow host) and the Config & Run page (plain-object host) so the two
 * stay in sync. The module never touches Drawflow — node access goes through a `host` adapter:
 *
 *   host = {
 *     panel,                 // target container element (cleared + filled on render)
 *     type, label,           // node type + display name (header)
 *     get params(),          // the node's current params object (read)
 *     get spec(),            // the /api/graph/node-form contract (node_params, family, model_field…)
 *     setParam(k, v),        // write a top-level param
 *     setExtraParam(k, v),   // write a nested params:{…} field (model Params schema)
 *     refetch(),             // re-resolve `spec` from the server (a discriminator changed) AND
 *                            //   re-render the panel — the model only signals; the host re-fetches
 *   }
 *
 * Model/dataset/option fetches are cached here (shared across both pages).
 */
(function () {
  let MODELS = null;
  let DATASETS = null;
  const PARAM_SCHEMAS = {};   // "family/type" -> /api/models/{family}/{type}/params
  const DATASET_INFO = {};    // dataset id -> /api/dataset/{id}/fields

  function modelsPromise() {  // lazy, cached: full per-family model lists from the registry
    if (MODELS) return Promise.resolve(MODELS);
    return fetch('/api/models').then(r => r.json()).then(d => (MODELS = d.models || d));
  }
  function schemaPromise(family, type) {  // per-model Params schema (sizes + fields + choices)
    const key = family + '/' + type;
    if (PARAM_SCHEMAS[key]) return Promise.resolve(PARAM_SCHEMAS[key]);
    return fetch(`/api/models/${family}/${type}/params`).then(r => r.json())
      .then(d => (PARAM_SCHEMAS[key] = d));
  }
  function datasetsPromise() {
    if (DATASETS) return Promise.resolve(DATASETS);
    return fetch('/api/datasets').then(r => r.json()).then(d => (DATASETS =
      d.datasets || (d.known_datasets || []).map(id => ({id, modality: 'other', domain: 'general'}))));
  }
  function datasetInfoPromise(id) {
    if (DATASET_INFO[id]) return Promise.resolve(DATASET_INFO[id]);
    return fetch(`/api/dataset/${id}/fields`).then(r => r.json()).then(d => (DATASET_INFO[id] = d));
  }

  function fieldRow(labelText, input, help) {
    const row = document.createElement('label');
    row.textContent = labelText + ' ';
    if (help) {
      row.title = help;                       // hover tooltip on the whole row
      const info = document.createElement('span');
      info.textContent = 'ⓘ'; info.className = 'help'; info.title = help;
      row.appendChild(info); row.append(' ');
    }
    row.appendChild(input);
    return row;
  }
  function selectEl(options, current, onChange) {
    const sel = document.createElement('select');
    options.forEach(o => {
      const opt = document.createElement('option');
      opt.value = o.value; opt.textContent = o.label;
      if (o.value === current) opt.selected = true;
      sel.appendChild(opt);
    });
    sel.onchange = () => onChange(sel.value);
    return sel;
  }
  function inputEl(type, current, placeholder, onChange) {
    const inp = document.createElement('input');
    inp.type = type; inp.value = current ?? ''; inp.placeholder = placeholder || '';
    inp.onchange = () => onChange(type === 'number' && inp.value !== ''
                                  ? Number(inp.value) : inp.value);
    return inp;
  }
  // dataset <select> grouped into optgroups by "Modality · domain" (Audio · medical, …).
  function datasetSelect(datasets, current, onChange) {
    const sel = document.createElement('select');
    const known = new Set(datasets.map(d => d.id));
    // Placeholder — the ONLY state with no dataset; a source opened from a config always names
    // one, so this never shows for an opened config (no stray '(from run config)').
    const blank = document.createElement('option');
    blank.value = ''; blank.textContent = '— select a dataset —';
    blank.disabled = true; if (!current) blank.selected = true;
    sel.appendChild(blank);
    // A value the registry doesn't list (e.g. a multi-source map key) still shows selected — a
    // set dataset never silently falls back to the placeholder.
    if (current && !known.has(current)) {
      const opt = document.createElement('option');
      opt.value = current; opt.textContent = current; opt.selected = true;
      sel.appendChild(opt);
    }
    const groups = {};
    datasets.forEach(d => {
      const mod = (d.modality || 'other');
      const key = mod.charAt(0).toUpperCase() + mod.slice(1) + ' · ' + (d.domain || 'general');
      (groups[key] ||= []).push(d);
    });
    Object.keys(groups).sort().forEach(key => {
      const og = document.createElement('optgroup'); og.label = key;
      groups[key].forEach(d => {
        const opt = document.createElement('option');
        opt.value = d.id; opt.textContent = d.id;
        opt.title = (d.description || '') +
          (d.compatible_pipeline_modes ? '\nmodes: ' + d.compatible_pipeline_modes.join(', ') : '');
        if (d.id === current) opt.selected = true;
        og.appendChild(opt);
      });
      sel.appendChild(og);
    });
    sel.onchange = () => onChange(sel.value);
    return sel;
  }

  // The model's own Params fields (size handled separately) → the `params:{…}` dict.
  function renderSchemaFields(panel, host, schema) {
    const extra = (host.params || {}).params || {};
    Object.entries(schema.params_schema || {}).forEach(([name, meta]) => {
      if (name === 'size') return;  // rendered as the size select
      const cur = extra[name] ?? meta.default ?? '';
      let widget;
      if (meta.choices && meta.choices.length) {
        widget = selectEl(meta.choices.map(c => ({value: c, label: c})), String(cur),
                          v => host.setExtraParam(name, v));
      } else if (typeof meta.default === 'boolean') {
        widget = document.createElement('input'); widget.type = 'checkbox';
        widget.checked = Boolean(cur);
        widget.onchange = () => host.setExtraParam(name, widget.checked);
      } else if (typeof meta.default === 'number') {
        widget = inputEl('number', cur, String(meta.default), v => host.setExtraParam(name, v));
      } else {
        widget = inputEl('text', cur, String(meta.default ?? ''), v => host.setExtraParam(name, v));
      }
      panel.appendChild(fieldRow(name, widget, meta.help));
    });
  }

  function renderModelSection(panel, host) {
    const spec = host.spec;
    const current = host.params || {};
    const group = document.createElement('div'); group.className = 'group';
    panel.appendChild(group);
    modelsPromise().then(models => {
      const fam = models[spec.family] || [];
      // The empty option NAMES the inherited model (the flat config default the run reads) —
      // a bare "(default)" tells the user nothing about what will actually run.
      const dm = spec.default_model;
      const dmName = dm && (fam.find(m => m.type === dm) || {}).name;
      const inheritLabel = dm ? `${dm}${dmName ? ' — ' + dmName : ''} (default)` : '(default)';
      const opts = [{value: '', label: inheritLabel}]
        .concat(fam.map(m => ({value: m.type, label: `${m.type} — ${m.name}`})));
      group.appendChild(fieldRow('model', selectEl(opts, current.model || '', v => {
        host.setParam('model', v);
        host.setParam('size', '');   // size choices belong to the model → reset
        host.setParam('params', null);
        renderPanel(host);           // re-render with the new model's schema
      })));
      if (!current.model) return;
      schemaPromise(spec.family, current.model).then(schema => {
        const sizes = Object.keys(schema.sizes || {});
        if (sizes.length > 1 || (sizes.length === 1 && sizes[0] !== 'default')) {
          const sizeOpts = sizes.map(s => ({value: s, label: s}));
          group.appendChild(fieldRow('size',
            selectEl(sizeOpts, current.size || schema.default_size || sizes[0],
                     v => host.setParam('size', v))));
        }
        renderSchemaFields(group, host, schema);
        // Advanced model-config the handler reads: an explicit checkpoint name, and (embedders)
        // the embedding space.
        group.appendChild(fieldRow('name', inputEl('text', current.name || '',
          schema.default_name || 'checkpoint (advanced)', v => host.setParam('name', v)),
          'Explicit model checkpoint — overrides the size pick.'));
        // LoRA/PEFT adapter — supported by the asr + text_embedding config folds
        // (asr_adapter_path / text_emb_adapter_path).
        if (spec.family === 'asr' || spec.family === 'text_embedding') {
          group.appendChild(fieldRow('adapter', inputEl('text', current.adapter || '',
            'LoRA adapter path (advanced)', v => host.setParam('adapter', v)),
            'Path to a LoRA/PEFT adapter directory applied on top of the base checkpoint.'));
        }
        if ((spec.family || '').includes('embedding')) {
          group.appendChild(fieldRow('embedding_space', inputEl('text', current.embedding_space || '',
            'space id (advanced)', v => host.setParam('embedding_space', v)),
            'Tag the embedding space so query/corpus compatibility is checked.'));
        }
        const hint = document.createElement('div'); hint.className = 'hint';
        hint.textContent = 'default: ' + (schema.default_name || '—');
        group.appendChild(hint);
      });
    });
  }

  // Dataset picker: select a REGISTERED dataset; its descriptor populates the node's columns
  // (params.fields) and surfaces the dataset's required settings.
  function renderDatasetSection(panel, host) {
    const current = host.params || {};
    const group = document.createElement('div'); group.className = 'group';
    panel.appendChild(group);
    datasetsPromise().then(datasets => {
      group.appendChild(fieldRow('dataset', datasetSelect(datasets, current.dataset || '', v => {
        host.setParam('dataset', v);
        host.setParam('fields', null);  // columns belong to the dataset → reset
        if (v) {
          datasetInfoPromise(v).then(info => {
            const fields = {};
            (info.fields || []).forEach(c => { fields[c.name] = c.artifact; });
            host.setParam('fields', fields);
            // derived (non-column) outputs the source also publishes (self-retrieval corpus) —
            // so the picked source matches the config-preview's ports
            host.setParam('extra_outputs', (info.derived_outputs || []).length
                          ? info.derived_outputs : null);
            host.refetch();  // re-resolve output ports to this dataset's real fields + re-render
          });
        } else { renderPanel(host); }
      })));
      if (!current.dataset) return;
      datasetInfoPromise(current.dataset).then(info => {
        if ((info.splits || []).length) {
          const opts = info.splits.map(sp => ({value: sp, label: sp}));
          group.appendChild(fieldRow('split', selectEl(
            opts, current.split || info.default_split || info.splits[0],
            v => host.setParam('split', v))));
        }
        (info.required_settings || []).forEach(rs => {
          const widget = inputEl('text', current[rs.key] ?? '', rs.field,
                                 v => host.setParam(rs.key, v));
          widget.classList.add('required-setting');
          if (!(current[rs.key] ?? '')) widget.classList.add('invalid');
          widget.addEventListener('input',
            () => widget.classList.toggle('invalid', !widget.value));
          const row = fieldRow(rs.key + ' *', widget);
          row.title = `required by dataset '${current.dataset}' (${rs.field})`;
          group.appendChild(row);
        });
        const hint = document.createElement('div'); hint.className = 'hint';
        hint.textContent = info.description || '';
        group.appendChild(hint);
      });
    });
  }

  /* Render the full param form for one node into host.panel (cleared first). */
  function renderPanel(host) {
    const panel = host.panel;
    const spec = host.spec;
    const current = host.params || {};
    panel.innerHTML = '';
    if (host.label || host.type) {
      const h = document.createElement('h4');
      h.innerHTML = `${host.label || host.type} <small>(${host.type || ''})</small>`;
      panel.appendChild(h);
    }

    // show_if: a field may be conditional on another param's value. Controllers re-render the
    // panel on change so dependents appear/disappear immediately.
    const params = spec.node_params || [];
    const controllers = new Set();
    params.forEach(p => Object.keys(p.show_if || {}).forEach(k => controllers.add(k)));
    const effective = key => {
      if (current[key] !== undefined && current[key] !== '') return String(current[key]);
      const decl = params.find(q => q.key === key);
      return decl && decl.default !== undefined ? String(decl.default) : '';
    };
    const isVisible = p => !p.show_if ||
      Object.entries(p.show_if).every(([dep, vals]) => vals.map(String).includes(effective(dep)));

    params.forEach(p => {
      if (p.kind === 'model' || p.kind === 'size' || p.kind === 'dict') return;  // model section
      if (p.kind === 'dataset') return;  // dataset section
      const cur = current[p.key] ?? '';
      const ph = p.default !== undefined ? String(p.default) : '';
      const onSet = v => {
        host.setParam(p.key, v);
        if (p.rerenders) host.refetch();                  // discriminator → field-aware re-resolve
        else if (controllers.has(p.key)) renderPanel(host);  // show_if → local re-render
      };
      let widget;
      if (p.kind === 'select') {
        // name the declared default in the empty option instead of a bare "(default)"
        const inheritLabel = p.default !== undefined && p.default !== null && p.default !== ''
          ? `${p.default} (default)` : '(default)';
        widget = selectEl([{value: '', label: inheritLabel}]
                            .concat((p.choices || []).map(c => ({value: c, label: c}))),
                          String(cur), onSet);
      } else if (p.kind === 'bool') {
        widget = document.createElement('input');
        widget.type = 'checkbox';
        widget.checked = cur === '' ? Boolean(p.default) : Boolean(cur);
        widget.onchange = () => {
          host.setParam(p.key, widget.checked);
          if (p.rerenders) host.refetch();
          else if (controllers.has(p.key)) renderPanel(host);
        };
      } else if (p.kind === 'number') {
        widget = inputEl('number', cur, ph, onSet);
      } else if (p.kind === 'device') {
        widget = inputEl('text', cur, 'cuda:0', onSet);
      } else if (p.kind === 'json') {
        widget = inputEl('text',
          cur === '' ? '' : (typeof cur === 'object' ? JSON.stringify(cur) : cur),
          '{} / []', v => {
            try {
              host.setParam(p.key, v === '' ? '' : JSON.parse(v));
              widget.classList.remove('invalid');
            } catch (e) { widget.classList.add('invalid'); }
          });
      } else {
        widget = inputEl('text', cur, ph, onSet);
      }
      const row = fieldRow(p.key, widget, p.help);
      if (!isVisible(p)) row.style.display = 'none';
      panel.appendChild(row);
    });

    if (params.some(p => p.kind === 'dataset')) renderDatasetSection(panel, host);
    if (spec.family) renderModelSection(panel, host);
    if (!spec.family && !params.length) {
      const em = document.createElement('em'); em.textContent = 'no declared params';
      panel.appendChild(em);
    }
  }

  window.NodeForm = {
    renderPanel,                // the per-node form renderer (builder + Config&Run)
    datasetInfo: DATASET_INFO,  // shared dataset-fields cache (builder's columnsFromParams reads it)
  };
})();
