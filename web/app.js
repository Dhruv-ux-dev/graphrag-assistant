// GraphRAG Knowledge Assistant — web UI.
// Talks to the FastAPI backend (api.py). When no backend answers, it loads
// demo-data.json: a snapshot of a real run, so the static site still works.

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const TYPE_COLORS = {
  PERSON: 'var(--person)', ORGANIZATION: 'var(--org)', LOCATION: 'var(--loc)', TECHNOLOGY: 'var(--tech)',
  PRODUCT: 'var(--product)', EVENT: 'var(--event)', CONCEPT: 'var(--concept)',
};
const typeColor = (t) => TYPE_COLORS[t] || TYPE_COLORS.CONCEPT;
const norm = (s) => String(s).toLowerCase().replace(/[^0-9a-z]+/g, ' ').trim();

const state = { mode: null, demo: null, overview: null, depth: 2, highlight: null, docFilter: '', selectedDoc: null };

// ---------------- data layer ----------------
// Where the API lives: same origin by default, or a hosted server set in config.js
const API_BASE = (window.GRAPHRAG_API || '').replace(/\/?$/, '/');
const REMOTE_API = !!window.GRAPHRAG_API;

async function api(path, opts = {}) {
  const res = await fetch((REMOTE_API ? API_BASE : '') + 'api/' + path, { headers: { 'Content-Type': 'application/json' }, ...opts });
  if (!res.ok) {
    let detail = res.statusText;
    try { const j = await res.json(); detail = typeof j.detail === 'string' ? j.detail : JSON.stringify(j.detail); } catch {}
    throw new Error(detail);
  }
  return res.json();
}

async function goLive(status) {
  state.mode = 'live';
  setStatus(status);
  $('demo-banner').hidden = true;
  for (const id of ['open-upload', 'open-upload-2', 'clear-graph']) { $(id).disabled = false; $(id).title = ''; }
  await refresh();
  renderSamples();
}

async function loadDemo(waking) {
  state.mode = 'demo';
  state.demo = await (await fetch('demo-data.json')).json();
  state.overview = state.demo.overview;
  setStatus(null, waking);
  $('demo-banner').hidden = false;
  $('demo-model').textContent = state.demo.meta.model;
  $('banner-waking').hidden = !waking;
  $('banner-static').hidden = waking;
  for (const id of ['open-upload', 'open-upload-2', 'clear-graph']) {
    $(id).disabled = true;
    $(id).title = waking ? 'The live server is starting…' : 'Available when running locally';
  }
  renderAll();
  renderSamples();
}

async function boot() {
  if (!REMOTE_API) {
    try { return await goLive(await api('status')); } catch { return loadDemo(false); }
  }
  // Hosted: show the snapshot at once, then switch to live when the (possibly sleeping) server answers
  await loadDemo(true);
  const deadline = Date.now() + 150000;
  while (Date.now() < deadline) {
    try { return await goLive(await api('status')); } catch { await new Promise((r) => setTimeout(r, 5000)); }
  }
  $('banner-waking').hidden = true;
  $('banner-static').hidden = false;
  setStatus(null, false);
}

async function refresh() {
  state.overview = await api('overview');
  renderAll();
}

function setStatus(status, waking) {
  const set = (id, text, cls) => { $(id).textContent = text; $(id).className = cls; };
  if (!status) {
    set('st-llm', 'Snapshot', 'warn');
    set('st-store', 'In-memory', 'warn');
    set('st-api', waking ? 'Waking up…' : 'Static demo', 'warn');
    $('set-mode').textContent = 'Static demo (snapshot)';
    $('set-model').textContent = state.demo.meta.model;
    $('set-store').textContent = 'In-memory (snapshot)';
    return;
  }
  const llmName = status.provider === 'groq' ? 'Groq' : 'Ollama';
  $('st-llm').previousElementSibling.textContent = `LLM (${llmName})`;
  set('st-llm', status.ollama ? 'Running' : 'Offline', status.ollama ? 'ok' : 'off');
  set('st-store', status.store === 'neo4j' ? 'Neo4j' : 'In-memory', 'ok');
  set('st-api', 'Healthy', 'ok');
  $('set-mode').textContent = 'Live';
  $('set-model').textContent = status.model;
  $('set-store').textContent = status.store === 'neo4j' ? 'Neo4j' : 'In-memory graph';
}

// ---------------- rendering ----------------
function renderAll() {
  renderStats();
  renderDocs();
  renderGraph();
  renderTables();
}

function renderStats() {
  const { stats, graph } = state.overview;
  $('s-docs').textContent = stats.documents;
  $('s-ents').textContent = stats.entities;
  $('s-rels').textContent = stats.relationships;
  $('s-cite').textContent = stats.citationValidity == null ? '–' : stats.citationValidity + '%';
  const types = new Set(graph.nodes.map((n) => n.type));
  const relTypes = new Set(graph.edges.map((e) => e.type));
  $('s-ents-sub').textContent = types.size ? `${types.size} types` : '';
  $('s-rels-sub').textContent = relTypes.size ? `${relTypes.size} kinds` : '';
  $('s-cite-sub').textContent = stats.citationValidity == null ? 'no answers yet' : 'cited items found in context';
}

function renderDocs() {
  const docs = state.overview.documents.filter((d) => !state.docFilter || norm(d.name + ' ' + d.text).includes(norm(state.docFilter)));
  const list = $('doc-list');
  list.innerHTML = docs.map((d) => `
    <li data-doc="${esc(d.name)}" class="${state.selectedDoc === d.name ? 'on' : ''}">
      <span class="doc-ico">TXT</span>
      <div class="doc-main"><b>${esc(d.name)}</b>
        <small>${d.entities} entities · ${d.relationships} relationships${d.words ? ` · ${d.words} words` : ''}</small></div>
    </li>
    ${state.selectedDoc === d.name && d.text ? `<div class="doc-detail">${esc(d.text)}</div>` : ''}`).join('')
    || `<li class="muted">${state.overview.documents.length ? 'No documents match.' : 'No documents yet.'}</li>`;
  const n = state.overview.documents.length;
  $('indexed').hidden = n === 0;
  $('indexed-text').textContent = `${n} document${n === 1 ? '' : 's'} processed into ${state.overview.stats.entities} entities and ${state.overview.stats.relationships} relationships.`;
}

$('doc-list').addEventListener('click', (e) => {
  const li = e.target.closest('li[data-doc]');
  if (!li) return;
  const name = li.dataset.doc;
  state.selectedDoc = state.selectedDoc === name ? null : name;
  const ids = state.selectedDoc ? state.overview.graph.nodes.filter((n) => n.documents.includes(name)).map((n) => n.id) : null;
  state.highlight = ids ? { ids: new Set(ids), cited: new Set() } : null;
  renderDocs();
  applyHighlight();
});
$('doc-search').addEventListener('input', (e) => { state.docFilter = e.target.value; renderDocs(); });

function renderTables() {
  const { nodes, edges } = state.overview.graph;
  const byId = Object.fromEntries(nodes.map((n) => [n.id, n]));
  const pill = (t) => `<span class="type-pill" style="background:${typeColor(t)}">${esc(t)}</span>`;
  $('ent-body').innerHTML = [...nodes].sort((a, b) => a.name.localeCompare(b.name)).map((n) =>
    `<tr><td><b>${esc(n.name)}</b></td><td>${pill(n.type)}</td><td>${esc(n.summary)}</td><td>${n.documents.length}</td></tr>`).join('');
  $('rel-body').innerHTML = edges.map((e) =>
    `<tr><td>${esc(byId[e.source]?.name)}</td><td><code>${esc(e.type)}</code></td><td>${esc(byId[e.target]?.name)}</td><td>${esc(e.context)}</td></tr>`).join('');
}

// ---------------- graph (d3 force layout) ----------------
let sim, zoom, gRoot, nodeSel, linkSel, labelSel, labelBgSel;

function renderGraph() {
  const svg = d3.select('#graph-svg');
  svg.selectAll('*').remove();
  const { nodes: rawNodes, edges } = state.overview.graph;
  $('graph-empty').hidden = rawNodes.length > 0;
  const types = [...new Set(rawNodes.map((n) => n.type))];
  $('g-legend').innerHTML = types.map((t) => `<span><i style="background:${typeColor(t)}"></i>${t.toLowerCase()}</span>`).join('')
    + `<span>· ${rawNodes.length} nodes · ${edges.length} relations</span>`;
  if (!rawNodes.length) return;

  const box = $('graph-view').getBoundingClientRect();
  const W = box.width || 600, H = box.height || 440;
  const degree = {};
  edges.forEach((e) => { degree[e.source] = (degree[e.source] || 0) + 1; degree[e.target] = (degree[e.target] || 0) + 1; });
  const nodes = rawNodes.map((n) => ({ ...n, r: 9 + Math.min(14, (degree[n.id] || 0) * 2.2) }));
  const links = edges.map((e) => ({ ...e }));

  const defs = svg.append('defs');
  defs.append('marker').attr('id', 'arrow').attr('viewBox', '0 -4 8 8').attr('refX', 8).attr('markerWidth', 7).attr('markerHeight', 7).attr('orient', 'auto')
    .append('path').attr('d', 'M0,-4L8,0L0,4').attr('fill', 'rgba(165,180,252,.6)');

  gRoot = svg.append('g');
  zoom = d3.zoom().scaleExtent([0.3, 3]).on('zoom', (ev) => gRoot.attr('transform', ev.transform));
  svg.call(zoom).on('dblclick.zoom', null);

  linkSel = gRoot.append('g').selectAll('line').data(links).join('line').attr('class', 'link').attr('marker-end', 'url(#arrow)');
  const lg = gRoot.append('g');
  labelBgSel = lg.selectAll('rect').data(links).join('rect').attr('class', 'link-label-bg').attr('rx', 6);
  labelSel = lg.selectAll('text').data(links).join('text').attr('class', 'link-label').attr('text-anchor', 'middle').attr('dy', 3).text((d) => d.type);

  nodeSel = gRoot.append('g').selectAll('g').data(nodes).join('g').attr('class', 'node')
    .call(d3.drag()
      .on('start', (ev, d) => { if (!ev.active) sim.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
      .on('drag', (ev, d) => { d.fx = ev.x; d.fy = ev.y; })
      .on('end', (ev, d) => { if (!ev.active) sim.alphaTarget(0); d.fx = null; d.fy = null; }))
    .on('click', (ev, d) => { ev.stopPropagation(); showNode(d); });
  nodeSel.append('circle').attr('r', (d) => d.r).attr('fill', (d) => typeColor(d.type)).attr('fill-opacity', 0.28).attr('stroke', (d) => typeColor(d.type));
  nodeSel.append('text').attr('text-anchor', 'middle').attr('dy', (d) => d.r + 14).text((d) => d.name.length > 22 ? d.name.slice(0, 21) + '…' : d.name);
  svg.on('click', () => { $('node-card').hidden = true; });

  sim = d3.forceSimulation(nodes)
    .force('link', d3.forceLink(links).id((d) => d.id).distance(110).strength(0.5))
    .force('charge', d3.forceManyBody().strength(-420))
    .force('center', d3.forceCenter(W / 2, H / 2))
    .force('collide', d3.forceCollide().radius((d) => d.r + 26))
    .force('x', d3.forceX(W / 2).strength(0.04)).force('y', d3.forceY(H / 2).strength(0.06))
    .on('tick', ticked)
    .on('end', fit);
  applyHighlight();
}

function ticked() {
  linkSel.each(function (d) {
    const dx = d.target.x - d.source.x, dy = d.target.y - d.source.y, len = Math.hypot(dx, dy) || 1;
    const ux = dx / len, uy = dy / len;
    d3.select(this).attr('x1', d.source.x + ux * d.source.r).attr('y1', d.source.y + uy * d.source.r)
      .attr('x2', d.target.x - ux * (d.target.r + 3)).attr('y2', d.target.y - uy * (d.target.r + 3));
  });
  labelSel.attr('x', (d) => (d.source.x + d.target.x) / 2).attr('y', (d) => (d.source.y + d.target.y) / 2);
  labelBgSel.each(function (d, i) {
    const t = labelSel.nodes()[i]; const w = t.getComputedTextLength() + 12;
    d3.select(this).attr('x', (d.source.x + d.target.x) / 2 - w / 2).attr('y', (d.source.y + d.target.y) / 2 - 8).attr('width', w).attr('height', 15);
  });
  nodeSel.attr('transform', (d) => `translate(${d.x},${d.y})`);
}

function fit() {
  if (!nodeSel) return;
  const svg = d3.select('#graph-svg');
  const box = $('graph-view').getBoundingClientRect();
  const xs = nodeSel.data().map((d) => d.x), ys = nodeSel.data().map((d) => d.y);
  const minX = Math.min(...xs) - 60, maxX = Math.max(...xs) + 60, minY = Math.min(...ys) - 50, maxY = Math.max(...ys) + 60;
  const k = Math.min(1.4, 0.95 * Math.min(box.width / (maxX - minX), box.height / (maxY - minY)));
  svg.transition().duration(500).call(zoom.transform, d3.zoomIdentity.translate(box.width / 2, box.height / 2).scale(k).translate(-(minX + maxX) / 2, -(minY + maxY) / 2));
}

function applyHighlight() {
  if (!nodeSel) return;
  const h = state.highlight;
  nodeSel.classed('dim', (d) => h && !h.ids.has(d.id)).classed('cited', (d) => h && h.cited.has(d.id));
  const linkIn = (d) => h && h.ids.has(d.source.id ?? d.source) && h.ids.has(d.target.id ?? d.target);
  linkSel.classed('dim', (d) => h && !linkIn(d)).classed('hl', (d) => h && linkIn(d));
  labelSel.classed('dim', (d) => h && !linkIn(d));
  labelBgSel.classed('dim', (d) => h && !linkIn(d));
}

function showNode(d) {
  const { edges, nodes } = state.overview.graph;
  const byId = Object.fromEntries(nodes.map((n) => [n.id, n.name]));
  const rels = edges.filter((e) => (e.source.id ?? e.source) === d.id || (e.target.id ?? e.target) === d.id)
    .map((e) => `${esc(byId[e.source.id ?? e.source])} <code>${esc(e.type)}</code> ${esc(byId[e.target.id ?? e.target])}`);
  const card = $('node-card');
  card.innerHTML = `<b>${esc(d.name)}</b><span class="tag" style="background:${typeColor(d.type)}">${esc(d.type)}</span>
    <p>${esc(d.summary || 'No summary.')}</p>
    ${rels.length ? `<p class="muted">${rels.slice(0, 5).join('<br>')}</p>` : ''}
    <p class="muted">Source: ${esc(d.documents.join(', '))}</p>`;
  card.hidden = false;
}

$('z-in').onclick = () => d3.select('#graph-svg').transition().call(zoom.scaleBy, 1.3);
$('z-out').onclick = () => d3.select('#graph-svg').transition().call(zoom.scaleBy, 0.77);
$('z-fit').onclick = () => { state.highlight = null; state.selectedDoc = null; renderDocs(); applyHighlight(); fit(); };

$('graph-tabs').addEventListener('click', (e) => {
  const b = e.target.closest('button'); if (!b) return;
  [...$('graph-tabs').children].forEach((x) => x.classList.toggle('on', x === b));
  $('graph-view').hidden = b.dataset.t !== 'graph';
  $('entities-view').hidden = b.dataset.t !== 'entities';
  $('relationships-view').hidden = b.dataset.t !== 'relationships';
  if (b.dataset.t === 'graph') fit();
});

// ---------------- chat ----------------
function renderSamples() {
  $('samples').innerHTML = state.overview.samples.map((q) => `<button type="button">${esc(q)}</button>`).join('');
}
$('samples').addEventListener('click', (e) => { const b = e.target.closest('button'); if (b) ask(b.textContent); });
$('ask-form').addEventListener('submit', (e) => { e.preventDefault(); const q = $('ask-input').value.trim(); if (q) ask(q); });
$('new-chat').onclick = () => {
  $('chat').querySelectorAll('.msg').forEach((m) => m.remove());
  $('welcome').hidden = false;
  state.highlight = null; applyHighlight();
};

function addMsg(cls, html) {
  $('welcome').hidden = true;
  const div = document.createElement('div');
  div.className = 'msg ' + cls;
  div.innerHTML = html;
  $('chat').appendChild(div);
  $('chat').scrollTop = $('chat').scrollHeight;
  return div;
}

async function ask(question) {
  $('ask-input').value = '';
  addMsg('user', esc(question));
  const pending = addMsg('bot', '<p class="thinking">Searching the graph…</p>');
  try {
    let res;
    if (state.mode === 'live') {
      res = await api('ask', { method: 'POST', body: JSON.stringify({ question, depth: state.depth }) });
      refresh();
    } else {
      res = state.demo.answers[norm(question)];
      if (!res) {
        pending.innerHTML = `<p>The static demo can only replay the sample questions below, because it has no LLM behind it. Run the project locally to ask anything.</p>
          <div class="chips">${state.overview.samples.map((q) => `<button type="button">${esc(q)}</button>`).join('')}</div>`;
        pending.querySelector('.chips').addEventListener('click', (e) => { const b = e.target.closest('button'); if (b) ask(b.textContent); });
        return;
      }
    }
    renderAnswer(pending, res);
  } catch (err) {
    pending.innerHTML = `<p>⚠ ${esc(err.message)}</p>`;
  }
}

function renderAnswer(el, res) {
  const valid = new Set(res.sources.map((s) => s.n));
  const body = esc(res.answer).replace(/\[(\d+)\]/g, (m, n) =>
    valid.has(+n) ? `<button class="cite" data-n="${n}" title="Source ${n}">${n}</button>`
      : `<span class="cite bad" title="Citation [${n}] is not in the retrieved context">${n}</span>`);
  const sources = res.sources.map((s) => `
    <div class="src" data-n="${s.n}"><span class="n">${s.n}</span><div>
      <b>${esc(s.entity)}</b> <small>${esc(s.document)}</small>
      <q>${esc(s.evidence.length > 240 ? s.evidence.slice(0, 240) + '…' : s.evidence)}</q>
      <div class="path">${esc(s.path.join(' → '))}</div></div></div>`).join('');
  el.innerHTML = `<p>${body}</p>
    ${res.sources.length ? `<div class="sources"><h5><span>Sources (validated)</span><span>${res.sources.length} cited${res.invalidCitations.length ? ` · ${res.invalidCitations.length} rejected` : ''}</span></h5>${sources}</div>` : ''}
    <details class="trace"><summary>How this answer was found</summary><ol>${res.trace.map((t) => `<li>${esc(t)}</li>`).join('')}</ol></details>`;
  el.addEventListener('click', (e) => {
    const c = e.target.closest('button.cite'); if (!c) return;
    const src = el.querySelector(`.src[data-n="${c.dataset.n}"]`);
    src.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
    src.classList.add('flash'); setTimeout(() => src.classList.remove('flash'), 1200);
  });
  state.highlight = { ids: new Set(res.contextIds), cited: new Set(res.citedIds) };
  applyHighlight();
  $('chat').scrollTop = $('chat').scrollHeight;
}

// ---------------- upload ----------------
const dlg = $('upload');
function openUpload() {
  const tpl = $('tpl');
  tpl.innerHTML = '<option value="">— Write your own —</option>' + Object.keys(state.overview.templates || {}).map((k) => `<option>${esc(k)}</option>`).join('');
  $('up-err').textContent = '';
  dlg.showModal();
}
$('open-upload').onclick = openUpload;
$('open-upload-2').onclick = openUpload;
$('up-cancel').onclick = () => dlg.close();
$('tpl').onchange = (e) => {
  const t = state.overview.templates[e.target.value];
  if (t) { $('up-name').value = e.target.value; $('up-text').value = t; }
};
$('up-file').onchange = async (e) => {
  const f = e.target.files[0]; if (!f) return;
  $('up-text').value = (await f.text()).slice(0, 20000);
  if (!$('up-name').value) $('up-name').value = f.name.replace(/\.[^.]+$/, '');
};
$('upload-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const btn = $('up-go');
  btn.disabled = true; btn.textContent = 'Extracting…'; $('up-err').textContent = '';
  try {
    const r = await api('documents', { method: 'POST', body: JSON.stringify({ name: $('up-name').value, text: $('up-text').value }) });
    dlg.close();
    toast(`Added “${r.name}”: ${r.entities} entities, ${r.relationships} relationships`);
    $('up-name').value = ''; $('up-text').value = '';
    await refresh();
  } catch (err) {
    $('up-err').textContent = err.message;
  } finally {
    btn.disabled = false; btn.textContent = 'Extract & ingest';
  }
});

$('clear-graph').onclick = async () => {
  if (!confirm('Reset the graph? Documents you added will be removed.')) return;
  await api('graph', { method: 'DELETE' });
  toast('Graph reset');
  await refresh();
};

function toast(text) {
  const t = $('toast'); t.textContent = text; t.hidden = false;
  clearTimeout(toast.timer); toast.timer = setTimeout(() => (t.hidden = true), 3500);
}

// ---------------- navigation & search ----------------
$('nav').addEventListener('click', (e) => {
  const b = e.target.closest('button'); if (!b) return;
  [...$('nav').children].forEach((x) => x.classList.toggle('on', x === b));
  const v = b.dataset.view;
  const show = { dashboard: ['documents', 'graph', 'chat'], chat: ['chat'], documents: ['documents'], graph: ['graph'], settings: ['settings'] }[v];
  for (const p of ['documents', 'graph', 'chat', 'settings']) $('p-' + p).hidden = !show.includes(p);
  $('grid').classList.toggle('solo', show.length === 1);
  if (show.includes('graph')) setTimeout(() => { renderGraph(); }, 0);
});

$('depth').oninput = (e) => { state.depth = +e.target.value; $('depth-out').textContent = e.target.value; };

$('global-search').addEventListener('input', (e) => {
  const q = norm(e.target.value);
  if (!q) { state.highlight = null; applyHighlight(); return; }
  const ids = state.overview.graph.nodes.filter((n) => norm(n.name + ' ' + n.summary).includes(q)).map((n) => n.id);
  state.highlight = { ids: new Set(ids), cited: new Set(ids) };
  applyHighlight();
});
document.addEventListener('keydown', (e) => {
  if (e.key === '/' && document.activeElement.tagName !== 'INPUT' && document.activeElement.tagName !== 'TEXTAREA') {
    e.preventDefault(); $('global-search').focus();
  }
});
let resizeTimer;
window.addEventListener('resize', () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(() => state.overview && renderGraph(), 200); });

boot();
