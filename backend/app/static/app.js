/* RAG Evaluation Harness — client.
   No framework: the page is three views over four endpoints, and a build step
   would cost more than it saves. */

const API = location.origin;

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

/** Escape before inserting retrieved text — corpus content is not trusted. */
const esc = (s) =>
  String(s ?? '').replace(/[&<>"']/g, (c) =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]
  );

const fmt = (n, d = 3) => (n === null || n === undefined ? '—' : Number(n).toFixed(d));

async function api(path, options = {}) {
  const res = await fetch(API + path, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const body = await res.json();
      if (body.detail) detail = typeof body.detail === 'string' ? body.detail : detail;
    } catch { /* non-JSON error body */ }
    throw new Error(detail);
  }
  return res.json();
}

/* ------------------------------- tabs --------------------------------- */

const PANELS = { 'tab-search': 'panel-search', 'tab-ask': 'panel-ask', 'tab-exp': 'panel-exp' };

$$('.tabs button').forEach((tab) => {
  tab.addEventListener('click', () => {
    $$('.tabs button').forEach((t) => {
      const on = t === tab;
      t.setAttribute('aria-selected', String(on));
      $('#' + PANELS[t.id]).hidden = !on;
    });
    if (tab.id === 'tab-exp' && !$('#exp-out').dataset.loaded) loadExperiments();
  });
});

/* ------------------------------ status -------------------------------- */

async function loadStatus() {
  const el = $('#status');
  try {
    const h = await api('/health');
    const docs = h.corpus?.documents ?? 0;
    const chunks = h.chunks?.count ?? 0;
    const gen = h.generator === 'unavailable' ? 'extractive' : h.generator.split(':')[0];
    el.innerHTML =
      `<span class="dot dot-ok"></span><span>${docs} docs · ${chunks.toLocaleString()} chunks · ${esc(gen)}</span>`;
  } catch (err) {
    el.innerHTML = `<span class="dot dot-bad"></span><span>${esc(err.message)}</span>`;
  }
}

/* ------------------------------ search -------------------------------- */

function renderHit(hit) {
  const d = hit.detail || {};
  const chips = [
    d.rrf !== undefined ? `<span class="chip rrf">rrf ${fmt(d.rrf, 4)}</span>` : '',
    d.dense !== undefined ? `<span class="chip dense">dense ${fmt(d.dense)}</span>` : '',
    d.bm25 !== undefined ? `<span class="chip bm25">bm25 ${fmt(d.bm25, 2)}</span>` : '',
  ].join('');

  const symbol = hit.meta?.symbol
    ? `<span class="sym">${esc(hit.meta.symbol)}${hit.meta.lineno ? `:${hit.meta.lineno}` : ''}</span>`
    : hit.meta?.heading_path
      ? `<span class="sym">${esc(hit.meta.heading_path)}</span>`
      : '';

  return `<article class="card">
    <div class="hit-head">
      <span class="rank">${hit.rank}</span>
      <span class="doc">${esc(hit.doc_id)}</span>
      ${symbol}
      <span class="scores">${chips}</span>
    </div>
    <pre class="snippet">${esc(hit.text)}</pre>
  </article>`;
}

async function runSearch(query, mode) {
  return api('/search', {
    method: 'POST',
    body: JSON.stringify({ query, k: 5, mode }),
  });
}

$('#search-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const query = $('#q').value.trim();
  if (!query) return;

  const out = $('#search-out');
  const button = $('#search-form button');
  const compareAll = $('#compare-all').checked;
  const modes = compareAll
    ? ['hybrid', 'dense', 'bm25']
    : [$$('input[name=mode]').find((r) => r.checked).value];

  button.disabled = true;
  out.innerHTML = '<div class="skeleton"></div><div class="skeleton"></div>';

  try {
    const results = await Promise.all(modes.map((m) => runSearch(query, m)));
    out.innerHTML = results
      .map((res, i) => {
        const heading = compareAll ? `<h2 class="section-label">${esc(modes[i])}</h2>` : '';
        const body = res.hits.length
          ? res.hits.map(renderHit).join('')
          : '<div class="card muted">Nothing retrieved for that query.</div>';
        return heading + body;
      })
      .join('');
  } catch (err) {
    out.innerHTML = `<div class="card error">${esc(err.message)}</div>`;
  } finally {
    button.disabled = false;
  }
});

/* -------------------------------- ask --------------------------------- */

$('#ask-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const query = $('#aq').value.trim();
  if (!query) return;

  const out = $('#ask-out');
  const button = $('#ask-form button');
  button.disabled = true;
  out.innerHTML = '<div class="skeleton"></div>';

  try {
    const a = await api('/ask', {
      method: 'POST',
      body: JSON.stringify({ query, k: 5, mode: 'hybrid' }),
    });

    // SVG, not emoji: emoji render inconsistently across platforms and are
    // announced as their unicode name by screen readers.
    const ICON_OK =
      '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M20 6L9 17l-5-5"/></svg>';
    const ICON_WARN =
      '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" aria-hidden="true"><path d="M12 9v4M12 17h.01M10.3 3.9L1.8 18a2 2 0 001.7 3h17a2 2 0 001.7-3L13.7 3.9a2 2 0 00-3.4 0z"/></svg>';

    const ungrounded = a.claims.filter((c) => !c.grounded).length;
    const verdict = a.fully_grounded
      ? `<span class="verdict ok">${ICON_OK} every claim is supported by its citation</span>`
      : `<span class="verdict bad">${ICON_WARN} ${ungrounded} of ${a.claims.length} claims unsupported</span>`;

    const claims = a.claims
      .map(
        (c) => `<div class="claim ${c.grounded ? '' : 'ungrounded'}">
          ${esc(c.text)}
          <span class="overlap">overlap ${fmt(c.overlap, 2)}${c.citations.length ? ` · cites [${c.citations.join('] [')}]` : ''}</span>
          ${c.reason ? `<span class="why">${esc(c.reason)}</span>` : ''}
        </div>`
      )
      .join('');

    const sources = a.sources
      .map(
        (s) => `<article class="card">
          <div class="hit-head">
            <span class="rank">${s.n}</span>
            <span class="doc">${esc(s.doc_id)}</span>
            <span class="scores"><span class="chip">score ${fmt(s.score, 4)}</span></span>
          </div>
          <pre class="snippet">${esc(s.preview)}</pre>
        </article>`
      )
      .join('');

    out.innerHTML = `
      <div class="card">
        ${verdict}
        <p style="margin:0 0 16px">${esc(a.text)}</p>
        <h2 class="section-label" style="margin-top:0">Claim verification</h2>
        ${claims}
        <p class="legend">
          Generated by <strong>${esc(a.model)}</strong>. Grounding is token overlap
          against the cited chunk — a weak check, but a measurement rather than
          asking the model to grade itself.
        </p>
      </div>
      <h2 class="section-label">Sources</h2>
      ${sources}`;
  } catch (err) {
    out.innerHTML = `<div class="card error">${esc(err.message)}</div>`;
  } finally {
    button.disabled = false;
  }
});

/* ----------------------------- experiments ---------------------------- */

const COLUMNS = [
  { key: 'chunking', label: 'chunking', get: (r) => r.config.chunking, num: false },
  { key: 'size', label: 'size', get: (r) => r.config.chunk_size, num: true },
  { key: 'retrieval', label: 'retrieval', get: (r) => r.config.retrieval, num: false },
  { key: 'chunks', label: 'chunks', get: (r) => r.chunk_stats.count, num: true },
  { key: 'r1', label: 'R@1', get: (r) => r.recall_at_1, num: true },
  { key: 'r5', label: 'R@5', get: (r) => r.recall_at_5, num: true },
  { key: 'mrr', label: 'MRR', get: (r) => r.mrr, num: true },
  { key: 'ndcg', label: 'nDCG@5', get: (r) => r.ndcg_at_5, num: true },
  { key: 'ms', label: 'ms/query', get: (r) => r.query_ms_mean, num: true },
];

let rows = [];
let sortKey = 'r1';
let sortDesc = true;

function renderTable() {
  const col = COLUMNS.find((c) => c.key === sortKey);
  const sorted = [...rows].sort((a, b) => {
    const av = col.get(a);
    const bv = col.get(b);
    const cmp = col.num ? av - bv : String(av).localeCompare(String(bv));
    return sortDesc ? -cmp : cmp;
  });
  const bestR1 = Math.max(...rows.map((r) => r.recall_at_1));

  $('#exp-out').innerHTML = `
    <div class="table-scroll"><table>
      <thead><tr>${COLUMNS.map(
        (c) =>
          `<th data-key="${c.key}" ${c.key === sortKey ? `aria-sort="${sortDesc ? 'descending' : 'ascending'}"` : ''}>${c.label}</th>`
      ).join('')}</tr></thead>
      <tbody>${sorted
        .map(
          (r) =>
            `<tr class="${r.recall_at_1 === bestR1 ? 'best' : ''}">${COLUMNS.map((c) => {
              const v = c.get(r);
              const text = c.num && typeof v === 'number' && v < 10 && !Number.isInteger(v)
                ? fmt(v)
                : typeof v === 'number'
                  ? v.toLocaleString()
                  : esc(v);
              return `<td class="${c.num ? 'num' : ''}">${text}</td>`;
            }).join('')}</tr>`
        )
        .join('')}</tbody>
    </table></div>
    <p class="legend">
      ${rows.length} configurations over ${rows[0]?.questions ?? 0} labelled questions.
      Highlighted rows share the best <strong>R@1</strong>. Sorted by
      <strong>${esc(col.label)}</strong> — click any header to re-sort.
      R@1 discriminates far better than R@5 here, because with a top-5 window
      almost every configuration finds the right document eventually; what
      separates them is whether it ranks <em>first</em>.
    </p>`;

  $$('#exp-out thead th').forEach((th) =>
    th.addEventListener('click', () => {
      const key = th.dataset.key;
      if (key === sortKey) sortDesc = !sortDesc;
      else {
        sortKey = key;
        sortDesc = true;
      }
      renderTable();
    })
  );
}

async function loadExperiments() {
  const out = $('#exp-out');
  out.innerHTML = '<div class="skeleton"></div>';
  try {
    const data = await api('/experiments');
    rows = data.results || [];
    out.dataset.loaded = '1';
    if (!rows.length) {
      out.innerHTML = '<div class="card muted">No stored results yet.</div>';
      return;
    }
    renderTable();
  } catch (err) {
    out.innerHTML = `<div class="card error">${esc(err.message)}<br><span class="muted">Run <code>python scripts/run_experiments.py</code> to generate the grid.</span></div>`;
  }
}

loadStatus();
