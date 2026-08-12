/* Chart mounting + theming for the evaluator UI.
 *
 * One module owns every Plotly interaction, so templates carry data — never layout
 * or color. Loaded once from base.html (Plotly itself is vendored beside this file,
 * so charts work offline and a fragment opened standalone still renders).
 *
 * Colors are never literals here: they are read from the CSS custom properties in
 * app.css at plot time, which is what lets one palette definition serve the page
 * chrome and the charts, and lets a theme switch restyle both.
 */

// Mounted charts by host id, so a theme change can re-render them in place.
var _charts = new Map();
var _themeCache = null;

var CHART_CONFIG = { displaylogo: false, responsive: true, displayModeBar: 'hover' };

/* Resolved chart tokens for the CURRENT theme. Memoized because getComputedStyle is
 * not free and a page can mount several charts in one pass. */
function chartTheme() {
  if (_themeCache) return _themeCache;
  var css = getComputedStyle(document.documentElement);
  var get = function (name, fallback) {
    return (css.getPropertyValue(name) || '').trim() || fallback;
  };
  _themeCache = {
    ink: get('--chart-ink', '#e2e8f0'),
    grid: get('--chart-grid', '#1f2937'),
    zero: get('--chart-zero', '#374151'),
    muted: get('--chart-muted', '#94a3b8'),
    dim: get('--chart-dim', '#64748b'),
    pos: get('--chart-pos', '#3987e5'),
    neg: get('--chart-neg', '#e66767'),
    warn: get('--warn', '#fbbf24'),
    surface: get('--surface', '#111827'),
    cat: [1, 2, 3, 4].map(function (i) { return get('--chart-cat-' + i, '#3987e5'); }),
    ord: [1, 2, 3, 4, 5].map(function (i) { return get('--chart-ord-' + i, '#3987e5'); }),
    font: get('--chart-font', 'Inter, system-ui, sans-serif')
  };
  return _themeCache;
}

/* Theme-aware layout defaults, with the caller's overrides winning. Axis dicts are
 * merged one level deep so a caller can set `yaxis.title` without losing the grid color. */
function chartLayout(overrides) {
  var t = chartTheme();
  var base = {
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: { color: t.ink, family: t.font, size: 12 },
    margin: { l: 56, r: 16, t: 16, b: 56 },
    hoverlabel: { bgcolor: t.surface, bordercolor: t.grid,
                  font: { color: t.ink, family: t.font } },
    colorway: t.cat,
    xaxis: { gridcolor: t.grid, zerolinecolor: t.zero, linecolor: t.grid,
             tickcolor: t.grid, automargin: true },
    yaxis: { gridcolor: t.grid, zerolinecolor: t.zero, linecolor: t.grid,
             tickcolor: t.grid, automargin: true }
  };
  var out = Object.assign({}, base, overrides || {});
  ['xaxis', 'yaxis', 'font', 'margin', 'hoverlabel'].forEach(function (k) {
    if (overrides && overrides[k]) out[k] = Object.assign({}, base[k], overrides[k]);
  });
  return out;
}

/* htmx-safe Plotly mount: plot into a created child, never the swapped element.
 * htmx's settle phase restores a swapped element's original attributes ~20ms
 * after the swap, stripping the js-plotly-plot class Plotly set on it — its
 * absolute-positioning CSS dies and the stacked SVG layers flow over the page,
 * swallowing clicks (the "Results page unresponsive after delete" bug).
 */
function plotInto(id, traces, overrides, config) {
  var host = document.getElementById(id);
  if (!host || !window.Plotly) return;
  // Release the previous instance before dropping its node — replaceChildren alone
  // leaks Plotly's global resize listener on every htmx swap.
  var prev = host.firstElementChild;
  if (prev) Plotly.purge(prev);
  host.replaceChildren();
  var target = host.appendChild(document.createElement('div'));
  target.style.height = '100%';
  Plotly.newPlot(target, traces, chartLayout(overrides), config || CHART_CONFIG);
  _charts.set(id, { host: host, target: target, traces: traces,
                    overrides: overrides, config: config || CHART_CONFIG });
  _observe(host, target);
}

/* A chart mounted inside a collapsed <details> sizes to 0 and never recovers.
 * Resize once it gains a box. */
function _observe(host, target) {
  if (!window.ResizeObserver) return;
  var seen = host.clientHeight > 0 && host.clientWidth > 0;
  var ro = new ResizeObserver(function () {
    var live = host.clientHeight > 0 && host.clientWidth > 0;
    if (live && !seen) Plotly.Plots.resize(target);
    seen = live;
  });
  ro.observe(host);
}

/* Re-render every live chart against the new tokens. `react` (not `newPlot`) so the
 * reader's zoom/pan survives a theme switch. Charts whose host left the DOM (htmx
 * replaced the fragment) are dropped here rather than leaking. */
function _restyleCharts() {
  _themeCache = null;
  _charts.forEach(function (entry, id) {
    if (!document.contains(entry.host)) { _charts.delete(id); return; }
    Plotly.react(entry.target, entry.traces, chartLayout(entry.overrides), entry.config);
  });
}

/* Resolve one semantic color key ("cat1", "pos", "ord3") against the current theme.
 * Specs never carry hex literals — that is what lets a single server-built spec render
 * correctly in both themes. A key may also be an ARRAY of keys (per-point coloring). */
function _color(key) {
  var t = chartTheme();
  var map = {
    pos: t.pos, neg: t.neg, muted: t.muted, dim: t.dim, zero: t.zero,
    ink: t.ink, warn: t.warn, grid: t.grid
  };
  for (var i = 0; i < 4; i++) map['cat' + (i + 1)] = t.cat[i];
  for (var j = 0; j < 5; j++) map['ord' + (j + 1)] = t.ord[j];
  if (Array.isArray(key)) return key.map(function (k) { return map[k] || t.cat[0]; });
  return map[key] || t.cat[0];
}

/* Where a colorkey lands differs by context, so resolve it explicitly rather than
 * walking blindly: on a trace it paints the mark (and the line, for a line mode),
 * inside error bars it is the whisker color, and on a layout shape it is the rule. */
function _resolveTrace(trace) {
  var out = Object.assign({}, trace);
  if (out.colorkey) {
    var hex = _color(out.colorkey);
    out.marker = Object.assign({ color: hex }, out.marker || {});
    if (typeof out.mode === 'string' && out.mode.indexOf('lines') !== -1) {
      out.line = Object.assign({ color: hex }, out.line || {});
    }
    delete out.colorkey;
  }
  ['error_x', 'error_y'].forEach(function (k) {
    if (out[k] && out[k].colorkey) {
      out[k] = Object.assign({}, out[k], { color: _color(out[k].colorkey) });
      delete out[k].colorkey;
    }
  });
  return out;
}

function _resolveLayout(layout) {
  var out = Object.assign({}, layout || {});
  if (Array.isArray(out.shapes)) {
    out.shapes = out.shapes.map(function (s) {
      var shape = Object.assign({}, s);
      if (shape.colorkey) {
        shape.line = Object.assign({ color: _color(shape.colorkey) }, shape.line || {});
        delete shape.colorkey;
      }
      return shape;
    });
  }
  return out;
}

/* Mount a chart from a server-built spec (see webapi/chart_data.py). */
function plotFromSpec(id) {
  var el = document.getElementById(id + '-spec');
  if (!el) return;
  var spec;
  try {
    spec = JSON.parse(el.textContent);
  } catch (e) {
    // Non-finite numbers are the usual cause; chart_data._num should prevent them.
    console.error('chart spec parse failed for', id, e);
    return;
  }
  if (!spec || spec.empty || !spec.series || !spec.series.length) return;
  plotInto(id, spec.series.map(_resolveTrace), _resolveLayout(spec.layout));
}

window.addEventListener('evaluator:theme', _restyleCharts);
// Follow the OS when the user has not pinned a theme.
if (window.matchMedia) {
  window.matchMedia('(prefers-color-scheme: light)').addEventListener('change', function () {
    if (!document.documentElement.dataset.theme) _restyleCharts();
  });
}
