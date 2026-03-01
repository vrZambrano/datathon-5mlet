"""
Streamlit Frontend - Passos Mágicos Assistente Pedagógico.

Esta aplicação fornece uma interface visual para:
- Visualizar métricas gerais
- Fazer predições de risco
- Ver clusters de alunos
- Gerar relatórios com LLM
"""

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
API_URL = st.secrets.get("API_URL", "http://localhost:8000")


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
    colors = {
        "BAIXO": "🟢",
        "MEDIO": "🟡",
        "ALTO": "🔴"
    }
    return colors.get(level, "⚪")


# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
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

    st.markdown("---")
    st.markdown("### Configuração")
    api_url_input = st.text_input("API URL", value=API_URL)
    if api_url_input != API_URL:
        st.session_state["API_URL"] = api_url_input
        st.rerun()


# ============================================================================
# PÁGINAS
# ============================================================================

# DASHBOARD
if page == "🏠 Dashboard":
    st.title("📊 Dashboard - Visão Geral")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total de Alunos", "---")

    with col2:
        st.metric("INDE Médio", "---")

    with col3:
        st.metric("Risco Alto", "---")

    with col4:
        st.metric("Clusters", "4")

    st.markdown("---")

    # Gráfico de distribuição de pedras (placeholder)
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Distribuição de Pedras")
        st.info("📊 Gráfico será exibido após carregamento dos dados")

    with col2:
        st.subheader("Alunos por Cluster")
        st.info("📊 Gráfico será exibido após carregamento dos dados")


# PREDIÇÃO DE RISCO
elif page == "📊 Predição de Risco":
    st.title("📊 Predição de Risco de Queda")

    st.markdown("Insira os dados do aluno para verificar o risco de queda de pedra:")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Identificação")
        aluno_id = st.text_input("RA do Aluno", "RA-1234")

        st.subheader("Indicadores Principais")
        inde = st.slider("INDE", 0.0, 10.0, 7.0, 0.1)
        ieg = st.slider("IEG (Engajamento)", 0.0, 10.0, 6.0, 0.1)
        ida = st.slider("IDA (Desempenho)", 0.0, 10.0, 7.0, 0.1)

    with col2:
        st.subheader("Indicadores Secundários")
        ips = st.slider("IPS (Psicossocial)", 0.0, 10.0, 5.0, 0.1)
        iaa = st.slider("IAA (Autoavaliação)", 0.0, 10.0, 6.0, 0.1)
        ian = st.slider("IAN (Adequação ao Nível)", 0.0, 10.0, 7.0, 0.1)

        st.subheader("Histórico")
        anos_no_programa = st.number_input("Anos no Programa", 1, 10, 2)

    st.markdown("---")

    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        if st.button("🔍 Calcular Risco", type="primary", use_container_width=True):
            # Prepara dados
            aluno_data = {
                "aluno_id": aluno_id,
                "ano": 2024,
                "inde": inde,
                "ieg": ieg,
                "ida": ida,
                "ips": ips,
                "iaa": iaa,
                "ian": ian,
                "anos_no_programa": anos_no_programa
            }

            # Faz predição
            with st.spinner("Calculando risco..."):
                resultado = predict_risk(aluno_data)

            if resultado:
                risco_nivel = resultado["risco_classe"]
                risco_prob = resultado["risco_probabilidade"]

                st.success(f"Risco: {risk_color(risco_nivel)} {risco_nivel}")
                st.metric("Probabilidade", f"{risco_prob:.1%}")

                if resultado.get("features_importantes"):
                    st.subheader("Features Mais Importantes")
                    for feat, imp in resultado["features_importantes"].items():
                        st.write(f"- {feat}: {imp:.3f}")


# CLUSTERS
elif page == "👥 Clusters":
    st.title("👥 Análise de Clusters")

    st.markdown("Descubra o perfil do aluno baseado em suas características:")

    col1, col2 = st.columns(2)

    with col1:
        aluno_id = st.text_input("RA do Aluno", "RA-1234")

        st.subheader("Indicadores")
        inde = st.slider("INDE", 0.0, 10.0, 7.0, 0.1, key="cluster_inde")
        ieg = st.slider("IEG", 0.0, 10.0, 6.0, 0.1, key="cluster_ieg")
        ida = st.slider("IDA", 0.0, 10.0, 7.0, 0.1, key="cluster_ida")
        ips = st.slider("IPS", 0.0, 10.0, 5.0, 0.1, key="cluster_ips")
        iaa = st.slider("IAA", 0.0, 10.0, 6.0, 0.1, key="cluster_iaa")

    with col2:
        st.subheader("Clusters Disponíveis")
        st.info("""
        - **Alto Desempenho**: Alunos com INDE alto e bom engajamento
        - **Engajados com Dificuldade**: Esforço alto, notas abaixo do esperado
        - **Em Risco**: Queda recente no engajamento
        - **Desmotivados Crônicos**: Baixo engajamento há muito tempo
        """)

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

            st.success(f"Cluster: {cluster_nome}")
            if resultado.get("cluster_descricao"):
                st.info(resultado["cluster_descricao"])


# RELATÓRIOS LLM
elif page == "📝 Relatórios LLM":
    st.title("📝 Geração de Relatórios com IA")

    st.markdown("Gere relatórios personalizados com recomendações pedagógicas usando IA.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Dados do Aluno")
        aluno_id = st.text_input("RA", "RA-1234", key="report_ra")
        nome = st.text_input("Nome do Aluno", "João Silva", key="report_nome")
        idade = st.number_input("Idade", 5, 100, 15, key="report_idade")

        pedra = st.selectbox("Pedra Atual", ["Quartzo", "Ágata", "Ametista", "Topázio"])
        inde = st.slider("INDE", 0.0, 10.0, 7.0, 0.1, key="report_inde")

        anos_no_programa = st.number_input("Anos no Programa", 1, 10, 2, key="report_anos")

    with col2:
        st.subheader("Informações Adicionais")
        tendencia = st.selectbox(
            "Tendência do INDE",
            ["Crescendo", "Estável", "Decrescendo"]
        )

        cluster = st.selectbox(
            "Cluster",
            ["Alto Desempenho", "Engajados com Dificuldade", "Em Risco", "Desmotivados Crônicos"]
        )

        risco = st.slider("Risco de Queda (%)", 0, 100, 30, key="report_risco")

        feedback = st.text_area(
            "Feedbacks Anteriores (opcional)",
            "Aluno dedicado mas com dificuldade em matemática.",
            height=100
        )

    st.markdown("---")

    if st.button("🤖 Gerar Relatório", type="primary", use_container_width=True):
        report_data = {
            "aluno_id": aluno_id,
            "nome": nome,
            "idade": idade,
            "pedra": pedra,
            "inde": inde,
            "anos_no_programa": anos_no_programa,
            "tendencia_inde": tendencia.lower(),
            "cluster_nome": cluster,
            "risco_percentual": risco,
            "feedback_texto": feedback
        }

        with st.spinner("Gerando relatório com IA..."):
            relatorio = generate_report(report_data)

        if relatorio:
            st.success("Relatório gerado com sucesso!")
            st.markdown("---")
            st.markdown(relatorio)

            st.download_button(
                "📥 Baixar Relatório",
                relatorio,
                file_name=f"relatorio_{aluno_id}.txt",
                mime="text/plain"
            )
        else:
            st.error("Erro ao gerar relatório. Verifique se a API está configurada.")


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
