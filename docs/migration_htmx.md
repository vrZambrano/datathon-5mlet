# Migração Frontend: Streamlit → HTMX + Jinja2 + Tailwind CSS

## Contexto

Migração do frontend Streamlit (`frontend/main.py`) para uma stack web moderna servida diretamente pelo FastAPI existente. Os endpoints JSON da API ficam **inalterados**. O novo frontend é montado em `/ui`.

**Stack nova:** FastAPI + Jinja2 + HTMX 2.0 + Alpine.js + Tailwind CSS (CDN) + Chart.js

---

## Checklist de Implementação

### Fase 1 — Infraestrutura ✅
- [x] Adicionar `python-multipart` e `markdown` em `requirements.txt`
- [x] Criar `frontend_web/static/js/charts.js`
- [x] Copiar logo para `frontend_web/static/img/`
- [x] Criar `frontend_web/templates/base.html` (shell: nav, sidebar, CDNs)
- [x] Criar `app/routes/web.py` com router skeleton
- [x] Montar `StaticFiles` e incluir `web.router` em `app/main.py`
- [x] `GET /ui` → redirect para `/ui/dashboard`

### Fase 2 — Dashboard ✅
- [x] Criar `frontend_web/templates/pages/dashboard.html`
- [x] Partial `partials/kpi_cards.html` + rota `GET /ui/partials/dashboard-kpis`
- [x] Partial `partials/chart_pedras.html` + rota `GET /ui/partials/chart-pedras`
- [x] Partial `partials/chart_clusters.html` + rota `GET /ui/partials/chart-clusters`
- [x] Partial `partials/chart_inde.html` + rota `GET /ui/partials/chart-inde`
- [x] Partial `partials/student_table.html` + rota `GET /ui/partials/student-table?ano=&pedra=&cluster=`
- [ ] Validar paridade de KPIs com o Streamlit (requer dados)

### Fase 3 — Predição de Risco e Clustering ✅
- [x] Criar `pages/risk.html` + student selector cascade (HTMX chained selects)
- [x] Partial `partials/risk_result.html` + rota `POST /ui/risk/predict`
- [x] Partial `partials/student_options.html` + rota `GET /ui/partials/student-options`
- [x] Rota `GET /ui/partials/student-data?ra=&ano=` (Alpine.js pre-fill)
- [x] Criar `pages/clustering.html`
- [x] Partial `partials/cluster_result.html` + rota `POST /ui/cluster/predict`

### Fase 4 — Relatórios LLM (Multi-step) ✅
- [x] Criar `pages/reports.html` com formulário completo
- [x] Rota `POST /ui/report/predictions` → `partials/report_predictions.html` (ML rápido)
- [x] Rota `POST /ui/report/llm` → `partials/report_content.html` (LLM lento)
- [x] Rota `GET /ui/report/download` → `PlainTextResponse` para download

### Fase 5 — Monitoramento ✅
- [x] Criar `pages/monitoring.html` com navegação de tabs via HTMX
- [x] Tab Drift: `partials/drift_table.html` + rota `GET /ui/partials/drift-table`
- [x] Tab Qualidade: `partials/quality_table.html` + rota `GET /ui/partials/quality-table`
- [x] Tab Relatório Visual: `<iframe src="/health/drift/report">` (lazy loading)
- [x] Tab Análise LLM: `partials/drift_llm.html` + rota `GET /ui/partials/drift-llm`

### Fase 6 — Docker ✅
- [x] Atualizar `docker/Dockerfile.api`: `COPY frontend_web/ ./frontend_web/`
- [x] Atualizar `docker/docker-compose.yml`: remover serviço `streamlit`, adicionar volume `frontend_web`
- [ ] Testar build completo (requer Docker disponível)

---

## Decisões de Arquitetura

### Convenção de cores
| Pedra | Tailwind |
|-------|---------|
| Quartzo | `bg-gray-400` |
| Ágata | `bg-blue-500` |
| Ametista | `bg-purple-500` |
| Topázio | `bg-orange-400` |

| Risco | Tailwind |
|-------|---------|
| BAIXO | `bg-green-100 text-green-800 border-green-300` |
| MEDIO | `bg-yellow-100 text-yellow-800 border-yellow-300` |
| ALTO | `bg-red-100 text-red-800 border-red-300` |

### Fluxo multi-step dos Relatórios
```
Click "Gerar Relatório"
  → POST /ui/report/predictions  (<1s, risco + cluster)
  → renderiza report_predictions.html
    contém: <div hx-post="/ui/report/llm" hx-trigger="load" ...>
  → auto-dispara POST /ui/report/llm (~10-30s, LLM)
  → renderiza report_content.html
```

### Chart.js em partials HTMX
```html
<canvas id="chartPedras"></canvas>
<script>
  new Chart(document.getElementById('chartPedras'), {
    type: 'doughnut',
    data: { labels: {{ pedra_labels|tojson }}, datasets: [{ data: {{ pedra_values|tojson }},
      backgroundColor: ['#9E9E9E','#2196F3','#9C27B0','#FF9800'] }] }
  });
</script>
```
Requer `htmx.config.allowScriptTags = true` no `base.html`.

---

## Gotchas
| Problema | Mitigação |
|---------|-----------|
| Scripts Chart.js não executam após HTMX swap | `htmx.config.allowScriptTags = true` |
| Timeout LLM (30s+) | `hx-request='{"timeout":90000}'` + `asyncio.wait_for(85s)` no server |
| Evidently iframe frame-busting | `X-Frame-Options: SAMEORIGIN` no endpoint `/health/drift/report` |
| Tabela de alunos grande | Paginação `?limit=50&offset=0` no partial |
| POST form values como string | Cast explícito `float()` nos handlers de `web.py` |
