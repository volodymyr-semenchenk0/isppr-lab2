from flask import Flask, render_template_string, request, jsonify
import numpy as np
from clustering import (
    isotonic_normalize, calc_distances_isotonic,
    isomorphic_normalize, calc_distances_isomorphic,
    calc_critical_radius, form_clusters,
    build_dendrite, calc_critical_distance, cut_dendrite
)

app = Flask(__name__)

# ---------- Дані ----------

EXAMPLE = {
    "labels":   ["Р1", "Р2", "Р3", "Р4", "Р5", "Р6"],
    "features": ["Х1", "Х2", "Х3", "Х4"],
    "matrix": [
        [30, 0.6,  2,   5],
        [33, 0.6,  2.5, 5],
        [50, 1.0,  2,  15],
        [45, 0.8,  2,  10],
        [20, 0.2,  1,   5],
        [25, 0.6,  1,  20],
    ]
}

VARIANT3 = {
    "labels":   ["Р1", "Р2", "Р3", "Р4", "Р5", "Р6", "Р7"],
    "features": ["Х1", "Х2", "Х3", "Х4", "Х5"],
    "matrix": [
        [8,  134, 2.25, 14,  8.5],
        [6,  141, 3.55, 14,  7.2],
        [6,  107, 1.75, 22,  7.8],
        [8,  128, 2.22, 13, 10.5],
        [9,  113, 1.45, 13, 13.1],
        [9,  141, 1.85, 22,  9.4],
        [8,  115, 2.15, 25, 11.3],
    ]
}


# ---------- Аналіз ----------

def analyse(data):
    M = np.array(data["matrix"], dtype=float)
    labels = data["labels"]
    res = {}

    # Ізотонічна
    norm_iso, col_sums, w = isotonic_normalize(M)
    D_iso = calc_distances_isotonic(w)
    r_iso, mins_iso = calc_critical_radius(D_iso)
    clusters_iso = form_clusters(D_iso, r_iso, labels)
    edges_iso = build_dendrite(D_iso, labels)
    dcrit_iso = calc_critical_distance(edges_iso)
    final_iso = cut_dendrite(edges_iso, dcrit_iso)

    res["isotonic"] = {
        "col_sums":       [round(x, 4) for x in col_sums],
        "norm":           [[round(v, 4) for v in row] for row in norm_iso],
        "w":              [round(x, 4) for x in w],
        "D":              [[round(v, 4) for v in row] for row in D_iso],
        "mins":           [round(x, 4) for x in mins_iso],
        "r":              round(r_iso, 4),
        "clusters_balls": clusters_iso,
        "edges":          [(u, v, round(d, 4)) for u, v, d in edges_iso],
        "dcrit":          round(dcrit_iso, 4),
        "clusters":       final_iso,
    }

    # Ізоморфна
    norm_ism = isomorphic_normalize(M)
    D_ism = calc_distances_isomorphic(norm_ism)
    r_ism, mins_ism = calc_critical_radius(D_ism)
    clusters_ism = form_clusters(D_ism, r_ism, labels)
    edges_ism = build_dendrite(D_ism, labels)
    dcrit_ism = calc_critical_distance(edges_ism)
    final_ism = cut_dendrite(edges_ism, dcrit_ism)

    res["isomorphic"] = {
        "norm":           [[round(v, 4) for v in row] for row in norm_ism],
        "D":              [[round(v, 4) for v in row] for row in D_ism],
        "mins":           [round(x, 4) for x in mins_ism],
        "r":              round(r_ism, 4),
        "clusters_balls": clusters_ism,
        "edges":          [(u, v, round(d, 4)) for u, v, d in edges_ism],
        "dcrit":          round(dcrit_ism, 4),
        "clusters":       final_ism,
    }

    res["labels"]   = labels
    res["features"] = data["features"]
    res["matrix"]   = data["matrix"]
    return res


# ---------- HTML ----------

PAGE = r"""
<!DOCTYPE html>
<html lang="uk">
<head>
<meta charset="UTF-8">
<title>Кластерний аналіз — Лаб. №2</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: Arial, sans-serif; font-size: 14px; background: #f0f2f5; color: #222; }

  header { background: #1e3a5f; color: #fff; padding: 14px 24px; }
  header h1 { font-size: 1.1rem; }
  header p  { font-size: 0.8rem; opacity: .7; margin-top: 3px; }

  .wrap { max-width: 1100px; margin: 24px auto; padding: 0 16px; }

  .tabs { display: flex; gap: 6px; margin-bottom: 16px; flex-wrap: wrap; }
  .tab  { padding: 8px 18px; border: 2px solid #b0bec5; border-radius: 6px;
          background: #fff; cursor: pointer; font-weight: 600; color: #1e3a5f; }
  .tab.active { background: #1e3a5f; color: #fff; border-color: #1e3a5f; }

  .panel { display: none; }
  .panel.active { display: block; }

  .card { background: #fff; border-radius: 8px;
          box-shadow: 0 1px 6px rgba(0,0,0,.1); margin-bottom: 16px; }
  .card-head { background: #1e3a5f; color: #fff; padding: 10px 16px;
               border-radius: 8px 8px 0 0; font-weight: 700; }
  .card-body { padding: 16px; }

  .tbl-wrap { overflow-x: auto; }
  table  { border-collapse: collapse; width: 100%; font-size: 13px; }
  th     { background: #1e3a5f; color: #fff; padding: 6px 10px; }
  td     { border: 1px solid #dde; padding: 5px 9px; text-align: center; }
  tr:nth-child(even) td { background: #f5f7fa; }
  td.lbl { font-weight: 700; background: #e8edf5 !important; }
  td.min { background: #fff8c5 !important; font-weight: 700; }
  td.cut { color: #c0392b; text-decoration: line-through; }
  td.ok  { color: #27ae60; }

  .cluster-row { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 8px; }
  .cluster { background: #e8edf5; border: 2px solid #1e3a5f;
             border-radius: 8px; padding: 8px 14px; }
  .cluster b { color: #1e3a5f; display: block; margin-bottom: 4px; }
  .tag { display: inline-block; color: #fff; border-radius: 4px;
         padding: 2px 8px; margin: 2px; font-size: 12px; font-weight: 600; }

  .btn { background: #1e3a5f; color: #fff; border: none; padding: 10px 24px;
         border-radius: 6px; cursor: pointer; font-size: 14px; font-weight: 700; }
  .btn:hover { background: #16304f; }

  label { font-weight: 600; display: block; margin-bottom: 4px; }
  input[type=text], textarea {
    border: 1px solid #b0bec5; border-radius: 6px;
    padding: 7px 10px; font-size: 13px; width: 100%; }
  textarea { min-height: 120px; font-family: monospace; resize: vertical; }

  .step-title { font-weight: 700; color: #1e3a5f; margin: 16px 0 6px; }
  .hint { color: #555; font-size: 12px; margin-bottom: 8px; }

  /* Dendrite SVG canvas */
  .dendrite-wrap { background: #f8faff; border: 1px solid #dde;
                   border-radius: 8px; overflow: hidden; margin-top: 4px; }
  .dendrite-wrap svg { display: block; width: 100%; }
</style>
</head>
<body>

<header>
  <h1>Лабораторна робота №2 — Кластерний аналіз</h1>
  <p>Ізотонічна та ізоморфна розбивка | Метод куль | Дендрит | Варіант №3</p>
</header>

<div class="wrap">

  <div class="tabs">
    <div class="tab active" onclick="switchTab('ex', this)">Приклад (6 об'єктів)</div>
    <div class="tab"        onclick="switchTab('v3', this)">Варіант №3 (7 об'єктів)</div>
    <div class="tab"        onclick="switchTab('cu', this)">Своя матриця</div>
  </div>

  <!-- Приклад -->
  <div id="panel-ex" class="panel active">
    <div class="card">
      <div class="card-head">Вхідна матриця — приклад</div>
      <div class="card-body">
        <div class="tbl-wrap">
          <table>
            <tr><th></th><th>Х1</th><th>Х2</th><th>Х3</th><th>Х4</th></tr>
            <tr><td class="lbl">Р1</td><td>30</td><td>0.6</td><td>2</td><td>5</td></tr>
            <tr><td class="lbl">Р2</td><td>33</td><td>0.6</td><td>2.5</td><td>5</td></tr>
            <tr><td class="lbl">Р3</td><td>50</td><td>1.0</td><td>2</td><td>15</td></tr>
            <tr><td class="lbl">Р4</td><td>45</td><td>0.8</td><td>2</td><td>10</td></tr>
            <tr><td class="lbl">Р5</td><td>20</td><td>0.2</td><td>1</td><td>5</td></tr>
            <tr><td class="lbl">Р6</td><td>25</td><td>0.6</td><td>1</td><td>20</td></tr>
          </table>
        </div>
        <br>
        <button class="btn" onclick="run('example')">▶ Запустити аналіз</button>
      </div>
    </div>
    <div id="out-ex"></div>
  </div>

  <!-- Варіант 3 -->
  <div id="panel-v3" class="panel">
    <div class="card">
      <div class="card-head">Вхідна матриця — Варіант №3</div>
      <div class="card-body">
        <div class="tbl-wrap">
          <table>
            <tr><th></th><th>Х1</th><th>Х2</th><th>Х3</th><th>Х4</th><th>Х5</th></tr>
            <tr><td class="lbl">Р1</td><td>8</td><td>134</td><td>2.25</td><td>14</td><td>8.5</td></tr>
            <tr><td class="lbl">Р2</td><td>6</td><td>141</td><td>3.55</td><td>14</td><td>7.2</td></tr>
            <tr><td class="lbl">Р3</td><td>6</td><td>107</td><td>1.75</td><td>22</td><td>7.8</td></tr>
            <tr><td class="lbl">Р4</td><td>8</td><td>128</td><td>2.22</td><td>13</td><td>10.5</td></tr>
            <tr><td class="lbl">Р5</td><td>9</td><td>113</td><td>1.45</td><td>13</td><td>13.1</td></tr>
            <tr><td class="lbl">Р6</td><td>9</td><td>141</td><td>1.85</td><td>22</td><td>9.4</td></tr>
            <tr><td class="lbl">Р7</td><td>8</td><td>115</td><td>2.15</td><td>25</td><td>11.3</td></tr>
          </table>
        </div>
        <br>
        <button class="btn" onclick="run('variant3')">▶ Запустити аналіз</button>
      </div>
    </div>
    <div id="out-v3"></div>
  </div>

  <!-- Своя матриця -->
  <div id="panel-cu" class="panel">
    <div class="card">
      <div class="card-head">Своя матриця</div>
      <div class="card-body">
        <label>Мітки об'єктів (через кому)</label>
        <input type="text" id="cu-labels" value="О1, О2, О3, О4, О5">
        <br><br>
        <label>Матриця (кожен об'єкт — новий рядок, значення через пробіл)</label>
        <br><br>
        <textarea id="cu-matrix">10 20 5 3
15 18 7 2
40  5 12 8
42  6 11 9
11 22 4 3</textarea>
        <br><br>
        <button class="btn" onclick="run('custom')">▶ Запустити аналіз</button>
      </div>
    </div>
    <div id="out-cu"></div>
  </div>

</div>

<script>
// ── палітра кольорів для кластерів ──
const PALETTE = [
  '#2980b9', '#27ae60', '#e67e22', '#8e44ad',
  '#c0392b', '#16a085', '#d35400', '#2c3e50'
];

function switchTab(name, el) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  el.classList.add('active');
  document.getElementById('panel-' + name).classList.add('active');
}

async function run(mode) {
  const outId = { example: 'out-ex', variant3: 'out-v3', custom: 'out-cu' }[mode];
  const out = document.getElementById(outId);
  out.innerHTML = '<p style="padding:12px;color:#666">Обраховується...</p>';

  let body = { mode };
  if (mode === 'custom') {
    body.labels = document.getElementById('cu-labels').value.split(',').map(s => s.trim());
    body.matrix = document.getElementById('cu-matrix').value.trim()
      .split('\n').map(row => row.trim().split(/\s+/).map(Number));
  }

  const resp = await fetch('/analyse', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body)
  });
  const data = await resp.json();

  if (data.error) {
    out.innerHTML = '<p style="color:red;padding:12px">' + data.error + '</p>';
    return;
  }

  out.innerHTML = renderResults(data);
  out.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ── головний рендер ──
function renderResults(d) {
  return renderMethod('Ізотонічна розбивка', d, d.isotonic, true)
       + renderMethod('Ізоморфна розбивка',  d, d.isomorphic, false);
}

function renderMethod(title, d, m, isIso) {
  const normCols = isIso ? [...d.features, 'wᵢ'] : d.features;
  const normRows = isIso ? m.norm.map((row, i) => [...row, m.w[i]]) : m.norm;
  const hint1 = isIso
    ? 'Суми по стовпцях: ' + d.features.map((f,i) => f+'='+m.col_sums[i]).join('; ')
    : 'Кожен рядок ділиться на свою евклідову норму √(Σx²)';

  return `<div class="card">
    <div class="card-head">${title}</div>
    <div class="card-body">

      <div class="step-title">Крок 1 — Нормована матриця</div>
      <p class="hint">${hint1}</p>
      ${makeTbl(d.labels, normCols, normRows)}

      <div class="step-title">Крок 2 — Матриця відстаней</div>
      <p class="hint">Жовтим виділено мінімум кожного рядка</p>
      ${makeDistTbl(d.labels, m.D, m.mins)}

      <div class="step-title">Крок 3 — Метод куль</div>
      <p class="hint">Критичний радіус r = <b>${m.r}</b></p>
      ${makeClusters(m.clusters_balls, m.clusters)}

      <div class="step-title">Крок 4 — Дендрит</div>
      <p class="hint">D_crit = <b>${m.dcrit}</b> &nbsp;|&nbsp;
        <span style="color:#c0392b">червоні ребра розриваються</span>,
        <span style="color:#27ae60">зелені зберігаються</span></p>
      ${makeDendrite(d.labels, m.edges, m.dcrit, m.clusters)}

      <div class="step-title">Крок 5 — Фінальні кластери</div>
      ${makeClusters(m.clusters, m.clusters)}

    </div>
  </div>`;
}

// ── таблиця нормованої матриці ──
function makeTbl(rowLabels, colLabels, data) {
  let h = '<div class="tbl-wrap"><table><tr><th></th>';
  colLabels.forEach(c => h += `<th>${c}</th>`);
  h += '</tr>';
  data.forEach((row, i) => {
    h += `<tr><td class="lbl">${rowLabels[i]}</td>`;
    row.forEach(v => h += `<td>${typeof v === 'number' ? v.toFixed(4) : v}</td>`);
    h += '</tr>';
  });
  return h + '</table></div>';
}

// ── таблиця відстаней ──
function makeDistTbl(labels, D, mins) {
  let h = '<div class="tbl-wrap"><table><tr><th></th>';
  labels.forEach(l => h += `<th>${l}</th>`);
  h += '</tr>';
  D.forEach((row, i) => {
    h += `<tr><td class="lbl">${labels[i]}</td>`;
    row.forEach((v, j) => {
      const isMin = i !== j && v === mins[i];
      h += `<td${isMin ? ' class="min"' : ''}>${i === j ? '0' : v.toFixed(4)}</td>`;
    });
    h += '</tr>';
  });
  return h + '</table></div>';
}

// ── кластери з кольорами ──
function makeClusters(list, finalClusters) {
  // Будуємо map: об'єкт → індекс фінального кластера
  const colorMap = buildColorMap(finalClusters);
  let h = '<div class="cluster-row">';
  list.forEach((cl, i) => {
    h += `<div class="cluster"><b>Кластер ${i + 1}</b>`;
    cl.forEach(o => {
      const color = PALETTE[colorMap[o] % PALETTE.length];
      h += `<span class="tag" style="background:${color}">${o}</span>`;
    });
    h += '</div>';
  });
  return h + '</div>';
}

// ── SVG візуалізація дендриту ──
function makeDendrite(labels, edges, dcrit, finalClusters) {
  const colorMap = buildColorMap(finalClusters);

  // 1. Будуємо список суміжності
  const adj = {};
  labels.forEach(l => adj[l] = []);
  edges.forEach(([u, v, d]) => {
    adj[u].push({ nb: v, d });
    adj[v].push({ nb: u, d });
  });

  // 2. BFS від кореня — визначаємо батьків і дітей
  const root = edges[0][0];
  const children = {};
  const depth = { [root]: 0 };
  labels.forEach(l => children[l] = []);
  const visited = new Set([root]);
  const queue = [root];
  let qi = 0;
  while (qi < queue.length) {
    const node = queue[qi++];
    (adj[node] || []).forEach(({ nb }) => {
      if (!visited.has(nb)) {
        visited.add(nb);
        children[node].push(nb);
        depth[nb] = depth[node] + 1;
        queue.push(nb);
      }
    });
  }
  // Вузли що не потрапили в BFS (ізольовані) — додаємо
  labels.forEach(l => { if (!(l in depth)) depth[l] = 0; });

  // 3. Призначаємо X: листки зліва направо, батько — посередині між дітьми
  let leafCounter = 0;
  const xPos = {};
  function assignX(node) {
    const kids = children[node] || [];
    if (kids.length === 0) {
      xPos[node] = leafCounter++;
    } else {
      kids.forEach(k => assignX(k));
      xPos[node] = (xPos[kids[0]] + xPos[kids[kids.length - 1]]) / 2;
    }
  }
  assignX(root);

  // 4. Розміри полотна
  const leafCount = Math.max(leafCounter, 1);
  const maxDepth  = Math.max(...Object.values(depth), 1);
  const cellW = 90;
  const cellH = 70;
  const nodeR = 22;
  const padX  = 50;
  const padY  = 40;
  const W = padX * 2 + (leafCount - 1) * cellW;
  const H = padY * 2 + maxDepth * cellH + 50; // +50 для легенди

  // X,Y кожного вузла
  const nx = {}, ny = {};
  labels.forEach(l => {
    nx[l] = padX + xPos[l] * cellW;
    ny[l] = padY + (depth[l] || 0) * cellH;
  });

  // 5. Рендеримо ребра
  let svgEdges = '';
  edges.forEach(([u, v, d]) => {
    const cut   = d > dcrit;
    const color = cut ? '#e74c3c' : '#27ae60';
    const dash  = cut ? 'stroke-dasharray="7 4"' : '';
    const sw    = cut ? 2 : 2.5;
    const mx    = (nx[u] + nx[v]) / 2;
    const my    = (ny[u] + ny[v]) / 2;

    svgEdges += `<line x1="${nx[u]}" y1="${ny[u]}" x2="${nx[v]}" y2="${ny[v]}"
      stroke="${color}" stroke-width="${sw}" ${dash}/>`;
    // Підпис на середині ребра
    svgEdges += `<text x="${mx}" y="${my - 5}" text-anchor="middle"
      font-size="10" fill="${color}" font-family="Arial"
      style="paint-order:stroke;stroke:#f8faff;stroke-width:3">${d.toFixed(4)}</text>`;
  });

  // 6. Вузли
  let svgNodes = '';
  labels.forEach(l => {
    const color = PALETTE[colorMap[l] % PALETTE.length];
    svgNodes += `<circle cx="${nx[l]}" cy="${ny[l]}" r="${nodeR}"
      fill="${color}" stroke="#fff" stroke-width="2.5"/>`;
    svgNodes += `<text x="${nx[l]}" y="${ny[l] + 5}" text-anchor="middle"
      font-size="12" font-weight="bold" fill="#fff" font-family="Arial">${l}</text>`;
  });

  // 7. Легенда
  const ly = H - 18;
  const legend = `
    <line x1="20" y1="${ly}" x2="52" y2="${ly}" stroke="#27ae60" stroke-width="2.5"/>
    <text x="57" y="${ly+4}" font-size="11" fill="#444" font-family="Arial">зберігається</text>
    <line x1="165" y1="${ly}" x2="197" y2="${ly}" stroke="#e74c3c"
      stroke-width="2" stroke-dasharray="7 4"/>
    <text x="202" y="${ly+4}" font-size="11" fill="#444" font-family="Arial">розрив (D_crit = ${dcrit})</text>
  `;

  return `<div class="dendrite-wrap">
    <svg viewBox="0 0 ${W} ${H}" style="height:${H}px">
      ${svgEdges}${svgNodes}${legend}
    </svg>
  </div>`;
}

// ── допоміжна: map об'єкт → індекс фінального кластера ──
function buildColorMap(clusters) {
  const map = {};
  (clusters || []).forEach((cl, i) => cl.forEach(o => map[o] = i));
  return map;
}
</script>
</body>
</html>
"""


@app.route('/')
def index():
    return render_template_string(PAGE)


@app.route('/analyse', methods=['POST'])
def analyse_route():
    try:
        body = request.get_json()
        mode = body.get('mode')

        if mode == 'example':
            data = EXAMPLE
        elif mode == 'variant3':
            data = VARIANT3
        elif mode == 'custom':
            labels = body['labels']
            matrix = body['matrix']
            data = {
                "labels":   labels,
                "features": [f"Х{i+1}" for i in range(len(matrix[0]))],
                "matrix":   matrix,
            }
        else:
            return jsonify({"error": "Невідомий режим"})

        return jsonify(analyse(data))

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == '__main__':
    app.run(debug=True, port=5000)
