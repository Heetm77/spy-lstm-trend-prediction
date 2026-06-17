from pathlib import Path
import pandas as pd
import json

REPORTS_V1 = Path("reports")
REPORTS_V2 = Path("reports/v2")
RAW_SPY = Path("data/raw/spy.csv")
OUTPUT = Path("reports/dashboard.html")

MODEL_NAMES = ["lstm_1_layer", "lstm_2_layer", "lstm_5_layer", "gru", "cnn_lstm"]
MODEL_LABELS = ["LSTM 1-Layer", "LSTM 2-Layer", "LSTM 5-Layer", "GRU", "CNN-LSTM"]
COLORS = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0"]


def load_metrics():
    v1 = pd.read_csv(REPORTS_V1 / "all_models_metrics.csv", index_col="model")
    v2 = pd.read_csv(REPORTS_V2 / "all_models_metrics.csv", index_col="model")
    return v1, v2


def load_history(reports_dir):
    histories = {}
    for name in MODEL_NAMES:
        path = reports_dir / f"{name}_history.csv"
        if path.exists():
            histories[name] = pd.read_csv(path)
    return histories


def load_backtest():
    return pd.read_csv(REPORTS_V1 / "backtest_timeseries.csv", parse_dates=["date"])


def build_equity_data(bt):
    bt = bt.dropna(subset=["benchmark_cumulative", "strategy_cumulative"])
    return {
        "dates": bt["date"].dt.strftime("%Y-%m-%d").tolist(),
        "benchmark": bt["benchmark_cumulative"].round(4).tolist(),
        "strategy": bt["strategy_cumulative"].round(4).tolist(),
    }


def build_roc_data(reports_dir):
    from sklearn.metrics import roc_curve, auc
    roc_data = {}
    for name in MODEL_NAMES:
        path = reports_dir / f"{name}_predictions.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        fpr, tpr, _ = roc_curve(df["actual"], df["predicted_probability"])
        roc_auc = auc(fpr, tpr)
        roc_data[name] = {
            "fpr": [round(x, 4) for x in fpr.tolist()][::5],
            "tpr": [round(x, 4) for x in tpr.tolist()][::5],
            "auc": round(roc_auc, 4),
        }
    return roc_data


def build_history_data(histories):
    result = {}
    for name, hist in histories.items():
        result[name] = {
            "epochs": list(range(1, len(hist) + 1)),
            "train_loss": hist["loss"].round(4).tolist(),
            "val_loss": hist["val_loss"].round(4).tolist(),
            "train_acc": hist["accuracy"].round(4).tolist(),
            "val_acc": hist["val_accuracy"].round(4).tolist(),
        }
    return result


def generate_html(d):
    data_json = json.dumps(d)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SPY LSTM Trend Prediction — Research Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: #0f1117; color: #e0e0e0; }}
  header {{ background: #1a1d2e; padding: 32px 40px; border-bottom: 1px solid #2a2d3e; }}
  header h1 {{ font-size: 1.8rem; font-weight: 700; color: #fff; }}
  header p {{ color: #8b8fa8; margin-top: 6px; font-size: 0.95rem; }}
  .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; padding: 32px 40px; max-width: 1400px; margin: 0 auto; }}
  .full {{ grid-column: 1 / -1; }}
  .card {{ background: #1a1d2e; border-radius: 12px; padding: 24px; border: 1px solid #2a2d3e; }}
  .card h2 {{ font-size: 1rem; font-weight: 600; color: #8b8fa8; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 16px; }}
  canvas {{ width: 100% !important; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.88rem; }}
  th {{ text-align: left; padding: 10px 12px; color: #8b8fa8; font-weight: 600; border-bottom: 1px solid #2a2d3e; }}
  td {{ padding: 10px 12px; border-bottom: 1px solid #1e2130; }}
  tr:last-child td {{ border-bottom: none; }}
  .best {{ color: #4CAF50; font-weight: 700; }}
  .tabs {{ display: flex; gap: 8px; margin-bottom: 16px; }}
  .tab {{ padding: 6px 16px; border-radius: 6px; border: 1px solid #2a2d3e; background: transparent; color: #8b8fa8; cursor: pointer; font-size: 0.85rem; }}
  .tab.active {{ background: #2196F3; color: #fff; border-color: #2196F3; }}
  .stat-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; }}
  .stat {{ background: #13151f; border-radius: 8px; padding: 16px; text-align: center; }}
  .stat .val {{ font-size: 1.5rem; font-weight: 700; color: #fff; }}
  .stat .lbl {{ font-size: 0.78rem; color: #8b8fa8; margin-top: 4px; }}
</style>
</head>
<body>
<header>
  <h1>SPY LSTM Trend Prediction</h1>
  <p>Deep learning architectures for next-day S&P 500 direction forecasting · 2010–2024 · Independent Research</p>
</header>
<div class="grid">

  <div class="card full">
    <h2>Experiment Summary</h2>
    <div class="stat-grid">
      <div class="stat"><div class="val">5</div><div class="lbl">Model Architectures</div></div>
      <div class="stat"><div class="val">3,574</div><div class="lbl">Trading Days</div></div>
      <div class="stat"><div class="val">33</div><div class="lbl">Features (V2)</div></div>
      <div class="stat"><div class="val">14yr</div><div class="lbl">Backtest Period</div></div>
    </div>
  </div>

  <div class="card full">
    <h2>Backtest — Strategy vs SPY Buy &amp; Hold</h2>
    <canvas id="equityChart" height="80"></canvas>
  </div>

  <div class="card">
    <h2>ROC Curves</h2>
    <div class="tabs">
      <button class="tab active" onclick="switchRoc('v1',this)">V1 Features (24)</button>
      <button class="tab" onclick="switchRoc('v2',this)">V2 Features (33)</button>
    </div>
    <canvas id="rocChart" height="140"></canvas>
  </div>

  <div class="card">
    <h2>Training vs Validation Loss</h2>
    <div class="tabs">
      <button class="tab active" onclick="switchHist('v1',this)">V1</button>
      <button class="tab" onclick="switchHist('v2',this)">V2</button>
    </div>
    <canvas id="lossChart" height="140"></canvas>
  </div>

  <div class="card">
    <h2>Model Metrics — V1 Features</h2>
    <table id="tableV1"></table>
  </div>

  <div class="card">
    <h2>Model Metrics — V2 Features</h2>
    <table id="tableV2"></table>
  </div>

</div>
<script>
const D = {data_json};

function makeMetricsTable(tableId, metrics) {{
  const el = document.getElementById(tableId);
  const cols = ['accuracy','precision','recall','f1_score','roc_auc'];
  let html = '<tr><th>Model</th>' + cols.map(c => `<th>${{c}}</th>`).join('') + '</tr>';
  const bestAuc = Math.max(...Object.values(metrics).map(m => m.roc_auc));
  D.model_names.forEach((name, i) => {{
    const m = metrics[name] || {{}};
    html += `<tr><td style="color:${{D.colors[i]}};font-weight:600">${{D.model_labels[i]}}</td>`;
    cols.forEach(c => {{
      const v = m[c] !== undefined ? m[c].toFixed(4) : '—';
      const isBest = c === 'roc_auc' && m[c] === bestAuc;
      html += `<td class="${{isBest ? 'best' : ''}}">${{v}}</td>`;
    }});
    html += '</tr>';
  }});
  el.innerHTML = html;
}}

makeMetricsTable('tableV1', D.metrics_v1);
makeMetricsTable('tableV2', D.metrics_v2);

const eqCtx = document.getElementById('equityChart').getContext('2d');
new Chart(eqCtx, {{
  type: 'line',
  data: {{
    labels: D.equity.dates,
    datasets: [
      {{ label: 'SPY Buy & Hold', data: D.equity.benchmark, borderColor: '#4CAF50', borderWidth: 2, pointRadius: 0, fill: false }},
      {{ label: 'LSTM Strategy', data: D.equity.strategy, borderColor: '#2196F3', borderWidth: 2, pointRadius: 0, fill: false }},
    ]
  }},
  options: {{ responsive: true, plugins: {{ legend: {{ labels: {{ color: '#e0e0e0' }} }} }}, scales: {{ x: {{ ticks: {{ color: '#8b8fa8', maxTicksLimit: 8 }}, grid: {{ color: '#1e2130' }} }}, y: {{ ticks: {{ color: '#8b8fa8' }}, grid: {{ color: '#1e2130' }} }} }} }}
}});

const rocCtx = document.getElementById('rocChart').getContext('2d');
function buildRocChart(roc_data) {{
  if (window._rocChart) window._rocChart.destroy();
  const datasets = D.model_names.map((name, i) => ({{
    label: `${{D.model_labels[i]}} (AUC=${{(roc_data[name]||{{}}).auc||0}})`,
    data: (roc_data[name]||{{fpr:[],tpr:[]}}).fpr.map((x,j) => ({{x, y: roc_data[name].tpr[j]}})),
    borderColor: D.colors[i], borderWidth: 2, pointRadius: 0, fill: false, showLine: true,
  }}));
  datasets.push({{ label: 'Random', data: [{{x:0,y:0}},{{x:1,y:1}}], borderColor: '#555', borderWidth: 1, borderDash: [4,4], pointRadius: 0, fill: false, showLine: true }});
  window._rocChart = new Chart(rocCtx, {{
    type: 'scatter',
    data: {{ datasets }},
    options: {{ responsive: true, plugins: {{ legend: {{ labels: {{ color: '#e0e0e0', font: {{ size: 11 }} }} }} }}, scales: {{ x: {{ min:0, max:1, title: {{ display:true, text:'FPR', color:'#8b8fa8' }}, ticks: {{ color:'#8b8fa8' }}, grid: {{ color:'#1e2130' }} }}, y: {{ min:0, max:1, title: {{ display:true, text:'TPR', color:'#8b8fa8' }}, ticks: {{ color:'#8b8fa8' }}, grid: {{ color:'#1e2130' }} }} }} }}
  }});
}}
buildRocChart(D.roc_v1);
function switchRoc(v, btn) {{
  btn.closest('.card').querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  btn.classList.add('active');
  buildRocChart(v === 'v1' ? D.roc_v1 : D.roc_v2);
}}

const lossCtx = document.getElementById('lossChart').getContext('2d');
function buildLossChart(hist_data) {{
  if (window._lossChart) window._lossChart.destroy();
  const datasets = [];
  D.model_names.forEach((name, i) => {{
    const h = hist_data[name];
    if (!h) return;
    datasets.push({{ label: `${{D.model_labels[i]}} train`, data: h.train_loss, borderColor: D.colors[i], borderWidth: 1.5, pointRadius: 0, fill: false }});
    datasets.push({{ label: `${{D.model_labels[i]}} val`, data: h.val_loss, borderColor: D.colors[i], borderWidth: 1.5, borderDash: [4,4], pointRadius: 0, fill: false }});
  }});
  window._lossChart = new Chart(lossCtx, {{
    type: 'line',
    data: {{ datasets }},
    options: {{ responsive: true, plugins: {{ legend: {{ labels: {{ color: '#e0e0e0', font: {{ size: 10 }}, boxWidth: 20 }} }} }}, scales: {{ x: {{ ticks: {{ color:'#8b8fa8' }}, grid: {{ color:'#1e2130' }} }}, y: {{ ticks: {{ color:'#8b8fa8' }}, grid: {{ color:'#1e2130' }} }} }} }}
  }});
}}
buildLossChart(D.hist_v1);
function switchHist(v, btn) {{
  btn.closest('.card').querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  btn.classList.add('active');
  buildLossChart(v === 'v1' ? D.hist_v1 : D.hist_v2);
}}
</script>
</body>
</html>"""


def main():
    v1_metrics, v2_metrics = load_metrics()
    v1_histories = load_history(REPORTS_V1)
    v2_histories = load_history(REPORTS_V2)
    bt = load_backtest()

    equity = build_equity_data(bt)
    roc_v1 = build_roc_data(REPORTS_V1)
    roc_v2 = build_roc_data(REPORTS_V2)
    hist_v1 = build_history_data(v1_histories)
    hist_v2 = build_history_data(v2_histories)

    metrics_v1 = v1_metrics[["accuracy", "precision", "recall", "f1_score", "roc_auc"]].round(4).to_dict("index")
    metrics_v2 = v2_metrics[["accuracy", "precision", "recall", "f1_score", "roc_auc"]].round(4).to_dict("index")

    data = {
        "model_names": MODEL_NAMES,
        "model_labels": MODEL_LABELS,
        "colors": COLORS,
        "equity": equity,
        "roc_v1": roc_v1,
        "roc_v2": roc_v2,
        "hist_v1": hist_v1,
        "hist_v2": hist_v2,
        "metrics_v1": metrics_v1,
        "metrics_v2": metrics_v2,
    }

    html = generate_html(data)
    OUTPUT.write_text(html, encoding="utf-8")
    print(f"Dashboard saved to {OUTPUT}")


if __name__ == "__main__":
    main()