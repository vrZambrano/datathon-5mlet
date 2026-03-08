"""
Streamlit Frontend - Passos Mágicos Assistente Pedagógico.

Esta aplicação fornece uma interface visual para:
- Visualizar métricas gerais
- Fazer predições de risco
- Ver clusters de alunos
- Gerar relatórios com LLM
"""

import os
import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional

# Configuração da página
st.set_page_config(
    page_title="Passos Mágicos - Assistente Pedagógico",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuração da API
API_URL = os.environ.get("API_URL", "http://localhost:8000")


# ============================================================================
# FUNÇÕES AUXILIARES
# ============================================================================

def fetch_health() -> dict:
    """Verifica saúde da API."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.json()
    except:
        return {"status": "error", "models_loaded": False}


@st.cache_data(ttl=120)
def fetch_students():
    """Busca lista de alunos da API (cache 2 min)."""
    try:
        resp = requests.get(f"{API_URL}/health/students", timeout=20)
        if resp.status_code == 200:
            data = resp.json()
            students = data.get("students", [])
            if students:
                return pd.DataFrame(students), data.get("anos_disponiveis", [])
    except Exception:
        pass
    return pd.DataFrame(), []


def predict_risk(aluno_data: dict) -> Optional[dict]:
    """Faz predição de risco."""
    try:
        response = requests.post(f"{API_URL}/predict/risk", json=aluno_data, timeout=10)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        st.error(f"Erro na predição: {e}")
        return None


def predict_cluster(aluno_data: dict) -> Optional[dict]:
    """Faz predição de cluster."""
    try:
        response = requests.post(f"{API_URL}/predict/cluster", json=aluno_data, timeout=10)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        st.error(f"Erro na predição: {e}")
        return None


def generate_report(report_data: dict) -> Optional[str]:
    """Gera relatório com LLM."""
    try:
        response = requests.post(f"{API_URL}/enrich/report", json=report_data, timeout=30)
        if response.status_code == 200:
            return response.json()["relatorio"]
        return None
    except Exception as e:
        st.error(f"Erro ao gerar relatório: {e}")
        return None


def risk_color(level: str) -> str:
    """Retorna cor baseada no nível de risco."""
    colors = {"BAIXO": "🟢", "MEDIO": "🟡", "ALTO": "🔴"}
    return colors.get(level, "⚪")


def safe_float(val, default=0.0):
    """Converte valor para float seguro."""
    try:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return default
        return float(val)
    except (TypeError, ValueError):
        return default


def render_student_selector(page_key: str, show_ano=True, show_pedra_filter=False, show_cluster_filter=False):
    """
    Renderiza filtros de seleção de aluno (Ano + RA/Nome).
    Opcionalmente mostra filtros de Pedra e Cluster sincronizados.
    Retorna o registro do aluno selecionado ou None.
    """
    df_students, anos_disponiveis = fetch_students()

    if df_students.empty:
        st.warning("⚠️ Não foi possível carregar dados dos alunos da API.")
        return None

    # Determina número de colunas baseado nos filtros ativos
    if show_pedra_filter and show_cluster_filter:
        col_ano, col_pedra, col_cluster, col_busca = st.columns([1, 1, 1, 2])
    elif show_pedra_filter or show_cluster_filter:
        col_ano, col_extra, col_busca = st.columns([1, 1, 2])
        if show_pedra_filter:
            col_pedra = col_extra
        else:
            col_cluster = col_extra
    else:
        col_ano, col_busca = st.columns([1, 3])

    # Aplica filtro de ano
    with col_ano:
        if show_ano and anos_disponiveis:
            ano_sel = st.selectbox(
                "Ano",
                options=anos_disponiveis,
                index=len(anos_disponiveis) - 1,
                key=f"{page_key}_ano"
            )
            df_filtered = df_students[df_students["ano"] == ano_sel]
        else:
            df_filtered = df_students
            ano_sel = None

    # Filtro de Pedra (sincronizado)
    if show_pedra_filter:
        with col_pedra:
            pedras_disponiveis = ["Todas"]
            if "pedra" in df_filtered.columns:
                pedras_unicas = df_filtered["pedra"].dropna().unique().tolist()
                pedras_disponiveis += sorted(pedras_unicas)
            pedra_sel = st.selectbox(
                "Filtrar por Pedra",
                options=pedras_disponiveis,
                index=0,
                key=f"{page_key}_pedra_filter"
            )
            if pedra_sel != "Todas":
                df_filtered = df_filtered[df_filtered["pedra"] == pedra_sel]

    # Filtro de Cluster (sincronizado)
    if show_cluster_filter:
        with col_cluster:
            clusters_disponiveis = ["Todos"]
            if "cluster" in df_filtered.columns:
                clusters_unicos = df_filtered["cluster"].dropna().unique().tolist()
                clusters_disponiveis += sorted(clusters_unicos)
            cluster_sel = st.selectbox(
                "Filtrar por Cluster",
                options=clusters_disponiveis,
                index=0,
                key=f"{page_key}_cluster_filter"
            )
            if cluster_sel != "Todos":
                df_filtered = df_filtered[df_filtered["cluster"] == cluster_sel]

    with col_busca:
        # Cria lista de opções: "RA - Nome"
        if "nome" in df_filtered.columns:
            df_filtered = df_filtered.sort_values("nome")
            options = df_filtered.apply(
                lambda r: f"{r['ra']} — {r.get('nome', '?')}", axis=1
            ).tolist()
        else:
            df_filtered = df_filtered.sort_values("ra")
            options = df_filtered["ra"].tolist()

        selected = st.selectbox(
            "Buscar Aluno (RA — Nome)",
            options=[""] + options,
            index=0,
            key=f"{page_key}_aluno",
            placeholder="Selecione um aluno..."
        )

    if not selected:
        return None

    # Extrai RA da seleção
    ra = selected.split(" — ")[0].strip()
    match = df_filtered[df_filtered["ra"] == ra]
    if match.empty:
        return None

    return match.iloc[0].to_dict()


# ============================================================================
# SIDEBAR
# ============================================================================

from pathlib import Path as _SidebarPath
_ICON_PATH = _SidebarPath(__file__).parent / "components" / "Passos-magicos-icon-cor.png"

with st.sidebar:
    if _ICON_PATH.exists():
        st.image(str(_ICON_PATH), width=180)
    st.title("🎓 Passos Mágicos")
    st.markdown("---")

    # Status da API
    health = fetch_health()
    if health.get("status") == "healthy":
        st.success("✓ API Online")
        if health.get("models_loaded"):
            st.success("✓ Modelos Carregados")
        else:
            st.warning("✗ Modelos Não Carregados")
    else:
        st.error("✗ API Offline")

    st.markdown("---")

    # Navegação
    st.markdown("### Navegação")
    page = st.radio(
        "Selecione uma página:",
        ["🏠 Dashboard", "📊 Predição de Risco", "👥 Clusters", "📝 Relatórios LLM"],
        label_visibility="collapsed"
    )


# ============================================================================
# PÁGINAS
# ============================================================================

# DASHBOARD
if page == "🏠 Dashboard":
    st.title("📊 Dashboard - Visão Geral")

    # -- busca dados brutos (todos os alunos, todos os anos) --
    df_all, anos_disp = fetch_students()

    if df_all.empty:
        st.warning("⚠️ Não foi possível carregar dados da API. Verifique se a API está rodando.")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total de Alunos", "---")
        with col2:
            st.metric("INDE Médio", "---")
        with col3:
            st.metric("Risco Alto", "---")
        with col4:
            st.metric("Clusters", "4")
    else:
        # ---- filtro por ano ----
        year_opts = ["Todos"] + [str(a) for a in sorted(anos_disp)] if anos_disp else ["Todos"]
        selected_year = st.selectbox("Filtrar por Ano", year_opts, key="dash_year")

        if selected_year != "Todos":
            df_view = df_all[df_all["ano"].astype(str) == selected_year].copy()
        else:
            # sem filtro: usa último registro de cada aluno
            df_view = df_all.sort_values("ano").groupby("ra", as_index=False).last()

        total_alunos = len(df_view)
        inde_medio = round(float(df_view["inde"].dropna().mean()), 2) if "inde" in df_view.columns and not df_view["inde"].dropna().empty else None

        # -- cluster local (joblib) --
        import joblib, json
        from pathlib import Path as _Path

        # Path funciona local (models/) e Docker (/app/models/)
        _models_dir = _Path("/app/models") if _Path("/app/models").exists() else _Path("models")

        cluster_feats = ["inde", "ieg", "ida", "ips", "iaa"]
        cluster_dist: dict = {}
        cluster_names: dict = {}
        df_view["_cluster"] = None

        try:
            _km = joblib.load(_models_dir / "clustering_model.pkl")
            _sc = joblib.load(_models_dir / "scaler.pkl")
            _lp = _models_dir / "cluster_labels.json"
            if _lp.exists():
                with open(_lp) as _f:
                    cluster_names = json.load(_f).get("cluster_names", {})
            _mask = df_view[cluster_feats].dropna().index
            if len(_mask):
                _X = _sc.transform(df_view.loc[_mask, cluster_feats].values)
                _labels = _km.predict(_X)
                df_view.loc[_mask, "_cluster"] = [cluster_names.get(str(c), f"Cluster {c}") for c in _labels]
                for cid in sorted(set(_labels)):
                    name = cluster_names.get(str(cid), f"Cluster {cid}")
                    cluster_dist[name] = int((_labels == cid).sum())
        except Exception as e:
            st.warning(f"Erro ao carregar clusters: {e}")

        # -- pedra distribution --
        pedra_dist: dict = {}
        if "pedra" in df_view.columns:
            _pc = df_view["pedra"].dropna().value_counts()
            pedra_dist = {str(k): int(v) for k, v in _pc.items()}

        # ---- KPIs ----
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total de Alunos", f"{total_alunos:,}")
        with col2:
            st.metric("INDE Médio", f"{inde_medio:.2f}" if inde_medio else "---")
        with col3:
            st.metric("Pedras", len(pedra_dist))
        with col4:
            st.metric("Clusters", len(cluster_dist) or 4)

        st.markdown("---")

        # ---- gráficos lado a lado ----
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Distribuição de Pedras")
            if pedra_dist:
                pedra_colors = {
                    "Quartzo": "#9E9E9E", "Ágata": "#2196F3",
                    "Ametista": "#9C27B0", "Topázio": "#FF9800",
                }
                labels = list(pedra_dist.keys())
                values = list(pedra_dist.values())
                colors = [pedra_colors.get(k, "#607D8B") for k in labels]
                fig = go.Figure(data=[go.Pie(
                    labels=labels, values=values,
                    marker=dict(colors=colors),
                    textinfo="label+percent+value",
                    hole=0.35,
                )])
                fig.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=350)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("📊 Dados de pedras não disponíveis")

        with col2:
            st.subheader("Alunos por Cluster")
            if cluster_dist:
                fig = go.Figure(data=[go.Bar(
                    x=list(cluster_dist.keys()),
                    y=list(cluster_dist.values()),
                    marker_color=["#EF5350", "#FFA726", "#66BB6A", "#42A5F5"][:len(cluster_dist)],
                    text=list(cluster_dist.values()),
                    textposition="auto",
                )])
                fig.update_layout(
                    xaxis_title="Cluster", yaxis_title="Nº de Alunos",
                    margin=dict(t=20, b=20, l=20, r=20), height=350,
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("📊 Dados de clusters não disponíveis")

        # ---- Evolução INDE (sempre todos os anos, sem filtro) ----
        if "inde" in df_all.columns and "ano" in df_all.columns:
            inde_por_ano = {
                str(int(ano)): round(float(grp["inde"].dropna().mean()), 2)
                for ano, grp in df_all.groupby("ano")
                if not grp["inde"].dropna().empty
            }
            if inde_por_ano:
                st.markdown("---")
                st.subheader("Evolução do INDE Médio por Ano")
                anos = sorted(inde_por_ano.keys())
                valores = [inde_por_ano[a] for a in anos]
                fig = go.Figure(data=[go.Scatter(
                    x=anos, y=valores,
                    mode="lines+markers+text",
                    text=[str(v) for v in valores],
                    textposition="top center",
                    line=dict(color="#1976D2", width=3),
                    marker=dict(size=10),
                )])
                fig.update_layout(
                    xaxis_title="Ano", yaxis_title="INDE Médio",
                    margin=dict(t=20, b=20, l=20, r=20), height=300,
                )
                st.plotly_chart(fig, use_container_width=True)

        # ---- Tabela de alunos com filtros ----
        st.markdown("---")
        st.subheader("📋 Tabela de Alunos")

        fcol1, fcol2 = st.columns(2)
        with fcol1:
            pedra_filter = st.selectbox(
                "Filtrar por Pedra",
                ["Todas"] + sorted(pedra_dist.keys()),
                key="dash_pedra_filter",
            )
        with fcol2:
            cluster_filter = st.selectbox(
                "Filtrar por Cluster",
                ["Todos"] + sorted(cluster_dist.keys()),
                key="dash_cluster_filter",
            )

        df_table = df_view.copy()
        if pedra_filter != "Todas" and "pedra" in df_table.columns:
            df_table = df_table[df_table["pedra"] == pedra_filter]
        if cluster_filter != "Todos" and "_cluster" in df_table.columns:
            df_table = df_table[df_table["_cluster"] == cluster_filter]

        show_cols = [c for c in ["ra", "nome", "ano", "pedra", "inde", "ieg", "ida", "ips", "iaa", "_cluster"] if c in df_table.columns]
        rename_map = {"_cluster": "Cluster", "nome": "Nome", "ra": "RA", "pedra": "Pedra", "inde": "INDE", "ieg": "IEG", "ida": "IDA", "ips": "IPS", "iaa": "IAA"}
        df_display = df_table[show_cols].rename(columns=rename_map)
        st.dataframe(df_display.reset_index(drop=True), use_container_width=True, height=400)


# PREDIÇÃO DE RISCO
elif page == "📊 Predição de Risco":
    st.title("📊 Predição de Risco de Queda")

    st.markdown("Selecione um aluno da base ou insira dados manualmente:")

    # Seletor de aluno
    aluno = render_student_selector("risk")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Identificação")
        aluno_id = st.text_input("RA do Aluno", value=aluno["ra"] if aluno else "RA-1234")

        st.subheader("Indicadores Principais")
        inde = st.slider("INDE", 0.0, 10.0, safe_float(aluno.get("inde"), 7.0) if aluno else 7.0, 0.1)
        ieg = st.slider("IEG (Engajamento)", 0.0, 10.0, safe_float(aluno.get("ieg"), 6.0) if aluno else 6.0, 0.1)
        ida = st.slider("IDA (Desempenho)", 0.0, 10.0, safe_float(aluno.get("ida"), 7.0) if aluno else 7.0, 0.1)

    with col2:
        st.subheader("Indicadores Secundários")
        ips = st.slider("IPS (Psicossocial)", 0.0, 10.0, safe_float(aluno.get("ips"), 5.0) if aluno else 5.0, 0.1)
        iaa = st.slider("IAA (Autoavaliação)", 0.0, 10.0, safe_float(aluno.get("iaa"), 6.0) if aluno else 6.0, 0.1)
        ian = st.slider("IAN (Adequação ao Nível)", 0.0, 10.0, safe_float(aluno.get("ian"), 7.0) if aluno else 7.0, 0.1)

        st.subheader("Histórico")
        anos_no_programa = st.number_input("Anos no Programa", 1, 10, 2)

        if aluno:
            st.info(f"📋 Pedra: **{aluno.get('pedra', '—')}** | Fase: **{aluno.get('fase', '—')}** | Turma: **{aluno.get('turma', '—')}**")

    st.markdown("---")

    if st.button("🔍 Calcular Risco", type="primary", use_container_width=True):
        aluno_data = {
            "aluno_id": aluno_id,
            "ano": 2024,
            "inde": inde,
            "ieg": ieg,
            "ida": ida,
            "ips": ips,
            "iaa": iaa,
            "ian": ian,
            "anos_no_programa": anos_no_programa,
            "tendencia_inde": safe_float(aluno.get("tendencia_inde"), 0.0) if aluno else 0.0,
            "pedras_mudadas_total": safe_float(aluno.get("pedras_mudadas_total"), 0.0) if aluno else 0.0,
            "delta_inde": safe_float(aluno.get("delta_inde"), None) if aluno else None,
            "delta_ieg": safe_float(aluno.get("delta_ieg"), None) if aluno else None,
            "delta_ida": safe_float(aluno.get("delta_ida"), None) if aluno else None,
        }

        with st.spinner("Calculando risco..."):
            resultado = predict_risk(aluno_data)

        if resultado:
            risco_nivel = resultado["risco_classe"]
            risco_prob = resultado["risco_probabilidade"]

            col_r1, col_r2 = st.columns(2)
            with col_r1:
                st.success(f"Risco: {risk_color(risco_nivel)} **{risco_nivel}**")
            with col_r2:
                st.metric("Probabilidade", f"{risco_prob:.1%}")

            if resultado.get("features_importantes"):
                st.subheader("Features Mais Importantes")
                for feat, imp in resultado["features_importantes"].items():
                    st.write(f"- {feat}: {imp:.3f}")


# CLUSTERS
elif page == "👥 Clusters":
    st.title("👥 Análise de Clusters")

    st.markdown("Selecione um aluno da base ou insira dados manualmente:")

    # Seletor de aluno
    aluno = render_student_selector("cluster")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        aluno_id = st.text_input("RA do Aluno", value=aluno["ra"] if aluno else "RA-1234", key="cl_ra")

        st.subheader("Indicadores")
        inde = st.slider("INDE", 0.0, 10.0, safe_float(aluno.get("inde"), 7.0) if aluno else 7.0, 0.1, key="cluster_inde")
        ieg = st.slider("IEG", 0.0, 10.0, safe_float(aluno.get("ieg"), 6.0) if aluno else 6.0, 0.1, key="cluster_ieg")
        ida = st.slider("IDA", 0.0, 10.0, safe_float(aluno.get("ida"), 7.0) if aluno else 7.0, 0.1, key="cluster_ida")
        ips = st.slider("IPS", 0.0, 10.0, safe_float(aluno.get("ips"), 5.0) if aluno else 5.0, 0.1, key="cluster_ips")
        iaa = st.slider("IAA", 0.0, 10.0, safe_float(aluno.get("iaa"), 6.0) if aluno else 6.0, 0.1, key="cluster_iaa")

    with col2:
        st.subheader("Clusters Disponíveis")
        st.info("""
        - **Alto Desempenho**: Alunos com INDE alto e bom engajamento
        - **Engajados com Dificuldade**: Esforço alto, notas abaixo do esperado
        - **Em Risco**: Queda recente no engajamento
        - **Desmotivados Crônicos**: Baixo engajamento há muito tempo
        """)

        if aluno:
            st.markdown("---")
            st.markdown(f"**Aluno selecionado:** {aluno.get('nome', aluno.get('ra', '—'))}")
            st.markdown(f"Pedra: **{aluno.get('pedra', '—')}** | Fase: **{aluno.get('fase', '—')}**")

    st.markdown("---")

    if st.button("🎯 Identificar Perfil", type="primary", use_container_width=True):
        aluno_data = {
            "aluno_id": aluno_id,
            "ano": 2024,
            "inde": inde,
            "ieg": ieg,
            "ida": ida,
            "ips": ips,
            "iaa": iaa
        }

        with st.spinner("Analisando perfil..."):
            resultado = predict_cluster(aluno_data)

        if resultado:
            cluster_nome = resultado["cluster_nome"]
            st.success(f"Cluster: **{cluster_nome}**")
            if resultado.get("cluster_descricao"):
                st.info(resultado["cluster_descricao"])


# RELATÓRIOS LLM
elif page == "📝 Relatórios LLM":
    st.title("📝 Geração de Relatórios com IA")

    st.markdown("Selecione um aluno da base ou preencha manualmente para gerar relatório pedagógico:")

    # Seletor de aluno com filtros de Pedra e Cluster sincronizados
    aluno = render_student_selector("report", show_pedra_filter=True, show_cluster_filter=True)

    st.markdown("---")

    col1, col2 = st.columns(2)

    pedra_options = ["Quartzo", "Ágata", "Ametista", "Topázio"]

    with col1:
        st.subheader("Dados do Aluno")
        aluno_id = st.text_input("RA", value=aluno["ra"] if aluno else "RA-1234", key="report_ra")
        nome = st.text_input("Nome do Aluno", value=aluno.get("nome", "João Silva") if aluno else "João Silva", key="report_nome")
        idade = st.number_input("Idade", 5, 100, 15, key="report_idade")

        # Pedra - tenta pré-selecionar se aluno selecionado
        pedra_default = 0
        if aluno and aluno.get("pedra"):
            pedra_val = aluno["pedra"]
            if pedra_val in pedra_options:
                pedra_default = pedra_options.index(pedra_val)
        pedra = st.selectbox("Pedra Atual", pedra_options, index=pedra_default, key="report_pedra")

        inde = st.slider("INDE", 0.0, 10.0, safe_float(aluno.get("inde"), 7.0) if aluno else 7.0, 0.1, key="report_inde")
        anos_no_programa = st.number_input("Anos no Programa", 1, 10,
            int(safe_float(aluno.get("anos_no_programa"), 2)) if aluno else 2,
            key="report_anos")

    with col2:
        st.subheader("Indicadores")
        ieg = st.slider("IEG (Engajamento)", 0.0, 10.0, safe_float(aluno.get("ieg"), 6.0) if aluno else 6.0, 0.1, key="report_ieg")
        ida = st.slider("IDA (Desempenho)", 0.0, 10.0, safe_float(aluno.get("ida"), 7.0) if aluno else 7.0, 0.1, key="report_ida")
        ips = st.slider("IPS (Psicossocial)", 0.0, 10.0, safe_float(aluno.get("ips"), 5.0) if aluno else 5.0, 0.1, key="report_ips")
        iaa = st.slider("IAA (Autoavaliação)", 0.0, 10.0, safe_float(aluno.get("iaa"), 6.0) if aluno else 6.0, 0.1, key="report_iaa")
        ian = st.slider("IAN (Adequação)", 0.0, 10.0, safe_float(aluno.get("ian"), 7.0) if aluno else 7.0, 0.1, key="report_ian")
        ipv = st.slider("IPV (Ponto de Virada)", 0.0, 10.0, safe_float(aluno.get("ipv"), 5.0) if aluno else 5.0, 0.1, key="report_ipv")
        ipp = st.slider("IPP (Psicopedagógico)", 0.0, 10.0, safe_float(aluno.get("ipp"), 5.0) if aluno else 5.0, 0.1, key="report_ipp")

    st.markdown("---")
    feedback = st.text_area(
        "Feedbacks Anteriores (opcional)",
        "",
        height=80,
        key="report_feedback"
    )

    if st.button("🤖 Gerar Relatório", type="primary", use_container_width=True):

        # ---- 1. Predição de Risco (ML) ----
        risk_payload = {
            "aluno_id": aluno_id,
            "ano": 2024,
            "inde": inde,
            "ieg": ieg,
            "ida": ida,
            "ips": ips,
            "iaa": iaa,
            "ian": ian,
            "anos_no_programa": anos_no_programa,
            "tendencia_inde": safe_float(aluno.get("tendencia_inde"), 0.0) if aluno else 0.0,
            "pedras_mudadas_total": safe_float(aluno.get("pedras_mudadas_total"), 0.0) if aluno else 0.0,
        }

        with st.spinner("🔍 Executando modelo de risco..."):
            risco_result = predict_risk(risk_payload)

        risco_classe = risco_result["risco_classe"] if risco_result else "N/A"
        risco_prob = risco_result["risco_probabilidade"] if risco_result else 0

        # ---- 2. Predição de Cluster (ML) ----
        cluster_payload = {
            "aluno_id": aluno_id,
            "ano": 2024,
            "inde": inde,
            "ieg": ieg,
            "ida": ida,
            "ips": ips,
            "iaa": iaa,
        }

        with st.spinner("👥 Executando modelo de clusterização..."):
            cluster_result = predict_cluster(cluster_payload)

        cluster_nome = cluster_result["cluster_nome"] if cluster_result else "Não identificado"

        # ---- 3. Exibe predições no topo ----
        st.markdown("---")
        st.subheader("🤖 Predições dos Modelos de Machine Learning")

        pred_col1, pred_col2 = st.columns(2)
        with pred_col1:
            if risco_result:
                cor = {"ALTO": "🔴", "MEDIO": "🟡", "BAIXO": "🟢"}.get(risco_classe, "⚪")
                st.metric("Risco de Queda", f"{cor} {risco_classe}", f"{risco_prob:.1%}")
                if risco_result.get("features_importantes"):
                    st.caption("Features mais relevantes:")
                    for feat, imp in risco_result["features_importantes"].items():
                        st.caption(f"  • {feat}: {imp:.3f}")
            else:
                st.warning("Não foi possível calcular o risco")

        with pred_col2:
            if cluster_result:
                st.metric("Cluster / Perfil", cluster_nome)
                if cluster_result.get("cluster_descricao"):
                    st.caption(cluster_result["cluster_descricao"])
            else:
                st.warning("Não foi possível identificar o cluster")

        # ---- 4. Tendência do INDE (derivada do valor numérico) ----
        tend_val = safe_float(aluno.get("tendencia_inde"), 0.0) if aluno else 0.0
        if tend_val > 0.1:
            tendencia = "crescendo"
        elif tend_val < -0.1:
            tendencia = "decrescendo"
        else:
            tendencia = "estável"

        # ---- 5. Gera relatório com LLM usando as predições ----
        report_data = {
            "aluno_id": aluno_id,
            "nome": nome,
            "idade": idade,
            "pedra": pedra,
            "inde": inde,
            "ieg": ieg,
            "ida": ida,
            "ips": ips,
            "iaa": iaa,
            "ian": ian,
            "ipv": ipv,
            "ipp": ipp,
            "anos_no_programa": anos_no_programa,
            "tendencia_inde": tendencia,
            "cluster_nome": cluster_nome,
            "risco_percentual": round(risco_prob * 100, 1),
            "risco_classe": risco_classe,
            "feedback_texto": feedback
        }

        with st.spinner("📝 Gerando relatório com IA..."):
            relatorio = generate_report(report_data)

        if relatorio:
            st.success("Relatório gerado com sucesso!")
            st.markdown("---")

            # Remove cabeçalho duplicado da resposta LLM (se existir)
            # A LLM às vezes inclui título próprio que duplica o nosso
            linhas = relatorio.strip().split('\n')
            linhas_limpas = []
            skip_header = True
            for linha in linhas:
                linha_lower = linha.lower().strip()
                # Pula linhas de cabeçalho até encontrar conteúdo real
                if skip_header:
                    if linha_lower.startswith('# relatório') or linha_lower.startswith('## relatório'):
                        continue
                    if linha_lower.startswith('pedra:') and 'inde:' in linha_lower:
                        continue
                    if linha_lower.startswith('**pedra') or linha_lower.startswith('> **pedra'):
                        continue
                    if linha.strip() == '' or linha.strip() == '---':
                        continue
                    skip_header = False  # Encontrou conteúdo real
                linhas_limpas.append(linha)
            relatorio_limpo = '\n'.join(linhas_limpas)

            # Cabeçalho do relatório com dados e predições
            st.markdown(f"# Relatório Pedagógico — {nome}")
            st.markdown(
                f"Pedra: **{pedra}** | INDE: **{inde:.1f}** | "
                f"Idade: **{idade}** anos | Tempo no Programa: **{anos_no_programa}** anos"
            )

            # Badges de predição
            risco_badge = {"ALTO": "🔴 ALTO", "MEDIO": "🟡 MÉDIO", "BAIXO": "🟢 BAIXO"}.get(risco_classe, risco_classe)
            st.markdown(
                f"> **Predição de Risco:** {risco_badge} ({risco_prob:.1%}) &nbsp;|&nbsp; "
                f"**Cluster:** {cluster_nome}"
            )
            st.markdown("---")

            st.markdown(relatorio_limpo)

            # Texto completo para download (inclui cabeçalho)
            report_full = (
                f"Relatório Pedagógico — {nome}\n"
                f"Pedra: {pedra} | INDE: {inde:.1f} | Idade: {idade} anos | Tempo no Programa: {anos_no_programa} anos\n"
                f"Predição de Risco: {risco_classe} ({risco_prob:.1%}) | Cluster: {cluster_nome}\n"
                f"{'='*60}\n\n"
                f"{relatorio_limpo}"
            )
            st.download_button(
                "📥 Baixar Relatório",
                report_full,
                file_name=f"relatorio_{aluno_id}.txt",
                mime="text/plain"
            )
        else:
            st.error("Erro ao gerar relatório. Verifique se a API e LLM estão configurados.")


# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
    <small>Passos Mágicos - Assistente Pedagógico v1.0</small>
    </div>
    """,
    unsafe_allow_html=True
)
