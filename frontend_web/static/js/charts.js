/**
 * Chart.js helpers for Passos Mágicos Web UI
 * Called via htmx:afterSwap event to re-initialize charts after HTMX swaps
 */

// Allow HTMX to re-execute scripts inside swapped content
if (typeof htmx !== "undefined") {
  htmx.config.allowScriptTags = true;
}

const PM_COLORS = {
  pedras: {
    Quartzo: "#9E9E9E",
    Ágata: "#2196F3",
    Ametista: "#9C27B0",
    Topázio: "#FF9800",
  },
  clusters: ["#EF5350", "#FFA726", "#66BB6A", "#42A5F5"],
  inde: "#1976D2",
};

// Chart registry to destroy before re-creating (avoids Canvas reuse errors)
const chartRegistry = {};

function destroyChart(id) {
  if (chartRegistry[id]) {
    chartRegistry[id].destroy();
    delete chartRegistry[id];
  }
}

function createDonutChart(canvasId, labels, values, colors) {
  destroyChart(canvasId);
  const ctx = document.getElementById(canvasId);
  if (!ctx) return;
  chartRegistry[canvasId] = new Chart(ctx, {
    type: "doughnut",
    data: {
      labels,
      datasets: [{ data: values, backgroundColor: colors, borderWidth: 2 }],
    },
    options: {
      responsive: true,
      cutout: "55%",
      plugins: {
        legend: { position: "bottom", labels: { padding: 16, font: { size: 13 } } },
      },
    },
  });
}

function createBarChart(canvasId, labels, values, colors) {
  destroyChart(canvasId);
  const ctx = document.getElementById(canvasId);
  if (!ctx) return;
  chartRegistry[canvasId] = new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [{ data: values, backgroundColor: colors, borderRadius: 6 }],
    },
    options: {
      responsive: true,
      plugins: { legend: { display: false } },
      scales: {
        y: { beginAtZero: true, ticks: { stepSize: 1 } },
        x: { ticks: { font: { size: 11 } } },
      },
    },
  });
}

function createLineChart(canvasId, labels, values, color) {
  destroyChart(canvasId);
  const ctx = document.getElementById(canvasId);
  if (!ctx) return;
  chartRegistry[canvasId] = new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          data: values,
          borderColor: color,
          backgroundColor: color + "22",
          borderWidth: 3,
          pointRadius: 6,
          pointBackgroundColor: color,
          fill: true,
          tension: 0.3,
        },
      ],
    },
    options: {
      responsive: true,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (ctx) => ` INDE: ${ctx.parsed.y.toFixed(2)}`,
          },
        },
      },
      scales: {
        y: { min: 0, max: 10, ticks: { stepSize: 1 } },
      },
    },
  });
}
