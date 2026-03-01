"""
Rotas de health check e monitoramento.
"""

import time
from fastapi import APIRouter, HTTPException
from loguru import logger
from pathlib import Path
import json
import pandas as pd
import numpy as np
import joblib
from app.core.config import get_settings
from app.models.schemas import HealthResponse

router = APIRouter()

settings = get_settings()

_start_time = time.time()


@router.get("", response_model=HealthResponse)
async def health_check():
    """
    Health check da API.

    Verifica status dos serviços e modelos.
    """
    # Verifica modelos
    model_files = [
        settings.classifier_model,
        settings.clustering_model,
        settings.scaler
    ]
    models_loaded = all(Path(m).exists() for m in model_files)

    # Verifica LLM
    llm_configured = bool(settings.openrouter_api_key and len(settings.openrouter_api_key) > 10)

    # Status dos serviços
    services = {
        "api": "ok",
        "classifier": "loaded" if Path(settings.classifier_model).exists() else "not_found",
        "clustering": "loaded" if Path(settings.clustering_model).exists() else "not_found",
        "llm": "configured" if llm_configured else "not_configured"
    }

    return HealthResponse(
        status="healthy",
        models_loaded=models_loaded,
        llm_configured=llm_configured,
        services=services
    )


@router.get("/metrics")
async def get_metrics():
    """
    Retorna métricas da API.

    Inclui uptime, status dos modelos e contadores básicos.
    """
    import time
    from pathlib import Path

    # Uptime (baseado no tempo de importação do módulo)
    uptime_seconds = time.time() - _start_time
    hours, remainder = divmod(int(uptime_seconds), 3600)
    minutes, seconds = divmod(remainder, 60)

    # Status dos modelos
    model_files = {
        "classifier": Path(settings.classifier_model),
        "clustering": Path(settings.clustering_model),
        "scaler": Path(settings.scaler),
    }
    models_status = {}
    for name, path in model_files.items():
        if path.exists():
            stat = path.stat()
            models_status[name] = {
                "status": "loaded",
                "size_kb": round(stat.st_size / 1024, 1),
                "modified": stat.st_mtime,
            }
        else:
            models_status[name] = {"status": "not_found"}

    return {
        "status": "ok",
        "uptime": f"{hours}h {minutes}m {seconds}s",
        "uptime_seconds": round(uptime_seconds, 1),
        "models": models_status,
        "environment": settings.environment,
    }


def _load_processed_data() -> pd.DataFrame:
    """Carrega e processa os CSVs para estatísticas do dashboard."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.data.loader import load_all_years
    from src.data.preprocessing import harmonize_datasets, normalize_pedra_column
    from src.data.feature_engineering import create_all_temporal_features

    datasets = load_all_years()
    if not datasets:
        return pd.DataFrame()

    df = harmonize_datasets(
        df_2022=datasets.get(2022),
        df_2023=datasets.get(2023),
        df_2024=datasets.get(2024),
    )
    df = normalize_pedra_column(df)
    df = create_all_temporal_features(df)
    return df


@router.get("/stats")
async def get_stats():
    """
    Retorna estatísticas agregadas para o dashboard.

    Inclui: total de alunos, INDE médio, distribuição de pedras,
    distribuição por clusters.
    """
    try:
        df = _load_processed_data()

        if df.empty:
            return {"error": "Nenhum dado disponível"}

        # Dados mais recentes de cada aluno
        df_latest = df.sort_values("ano").groupby("RA").last().reset_index()

        total_alunos = len(df_latest)

        # INDE médio
        inde_medio = None
        if "INDE" in df_latest.columns:
            inde_medio = round(float(df_latest["INDE"].dropna().mean()), 2)

        # Distribuição de pedras
        pedra_dist = {}
        if "PEDRA" in df_latest.columns:
            counts = df_latest["PEDRA"].dropna().value_counts()
            pedra_dist = {str(k): int(v) for k, v in counts.items()}

        # Risco alto - usa o classificador se disponível
        risco_alto = 0
        try:
            classifier_path = Path(settings.classifier_model)
            if classifier_path.exists():
                model = joblib.load(classifier_path)
                feature_cols = [
                    "INDE", "IEG", "IDA", "IPS", "IAA",
                    "delta_INDE", "delta_IEG", "delta_IDA",
                    "anos_no_programa", "tendencia_INDE",
                    "pedras_mudadas_total"
                ]
                available = [c for c in feature_cols if c in df_latest.columns]
                df_pred = df_latest[available].dropna()
                if not df_pred.empty:
                    # Preenche colunas faltantes com 0
                    for c in feature_cols:
                        if c not in df_pred.columns:
                            df_pred[c] = 0.0
                    df_pred = df_pred[feature_cols]
                    probs = model.predict_proba(df_pred)[:, 1]
                    risco_alto = int((probs >= settings.risk_threshold).sum())
        except Exception as e:
            logger.warning(f"Erro ao calcular risco: {e}")

        # Distribuição de clusters
        cluster_dist = {}
        try:
            clustering_path = Path(settings.clustering_model)
            scaler_path = Path(settings.scaler)
            labels_path = Path("models/cluster_labels.json")

            if clustering_path.exists() and scaler_path.exists():
                km = joblib.load(clustering_path)
                scaler = joblib.load(scaler_path)

                cluster_feats = ["INDE", "IEG", "IDA", "IPS", "IAA"]
                available_cf = [c for c in cluster_feats if c in df_latest.columns]
                df_cl = df_latest[available_cf].dropna()

                if not df_cl.empty:
                    X_scaled = scaler.transform(df_cl[cluster_feats])
                    labels = km.predict(X_scaled)

                    # Carrega nomes dos clusters
                    cluster_names = {}
                    if labels_path.exists():
                        with open(labels_path) as f:
                            ldata = json.load(f)
                        cluster_names = ldata.get("cluster_names", {})

                    for cid in sorted(set(labels)):
                        name = cluster_names.get(str(cid), f"Cluster {cid}")
                        cluster_dist[name] = int((labels == cid).sum())
        except Exception as e:
            logger.warning(f"Erro ao calcular clusters: {e}")

        # Evolução INDE por ano
        inde_por_ano = {}
        if "INDE" in df.columns and "ano" in df.columns:
            inde_por_ano = {
                str(int(ano)): round(float(grp["INDE"].dropna().mean()), 2)
                for ano, grp in df.groupby("ano") if not grp["INDE"].dropna().empty
            }

        return {
            "total_alunos": total_alunos,
            "inde_medio": inde_medio,
            "risco_alto": risco_alto,
            "n_clusters": 4,
            "pedra_distribuicao": pedra_dist,
            "cluster_distribuicao": cluster_dist,
            "inde_por_ano": inde_por_ano,
            "anos_disponiveis": sorted(df["ano"].unique().tolist()) if "ano" in df.columns else [],
        }
    except Exception as e:
        logger.error(f"Erro ao gerar stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/students")
async def get_students(ano: int = None):
    """
    Retorna lista de alunos com seus indicadores para seleção no frontend.

    Parâmetros:
        ano: Filtrar por ano específico (opcional, padrão = ano mais recente)
    """
    try:
        df = _load_processed_data()
        if df.empty:
            return {"students": []}

        # Se ano não especificado, usa todos os anos
        if ano:
            df_year = df[df["ano"] == ano]
        else:
            df_year = df

        if df_year.empty:
            return {"students": [], "anos_disponiveis": []}

        # Propaga NOME de outros anos para alunos sem nome
        if "NOME" in df.columns:
            name_lookup = df.dropna(subset=["NOME"]).drop_duplicates(subset=["RA"], keep="first")[["RA", "NOME"]]
            if "NOME" in df_year.columns:
                df_year = df_year.drop(columns=["NOME"])
            df_year = df_year.merge(name_lookup, on="RA", how="left")

        # Colunas relevantes para retorno
        cols_map = {
            "RA": "ra", "NOME": "nome", "ano": "ano",
            "INDE": "inde", "IEG": "ieg", "IDA": "ida",
            "IPS": "ips", "IAA": "iaa", "IAN": "ian",
            "IPV": "ipv", "IPP": "ipp",
            "PEDRA": "pedra", "FASE": "fase", "TURMA": "turma",
            "INSTITUICAO DE ENSINO": "instituicao",
            "ANO INGRESSO": "ano_ingresso",
            "tendencia_INDE": "tendencia_inde",
            "pedras_mudadas_total": "pedras_mudadas_total",
            "anos_no_programa": "anos_no_programa",
            "delta_INDE": "delta_inde",
            "delta_IEG": "delta_ieg",
            "delta_IDA": "delta_ida",
        }

        available_cols = {k: v for k, v in cols_map.items() if k in df_year.columns}
        df_out = df_year[list(available_cols.keys())].rename(columns=available_cols)

        # Converte NaN para None para JSON (replace NaN em colunas numéricas)
        for col in df_out.select_dtypes(include=["float64", "float32"]).columns:
            df_out[col] = df_out[col].where(df_out[col].notna(), None)
        # Converte NaN em colunas object
        df_out = df_out.where(df_out.notna(), None)

        # Ordena por nome/RA
        sort_col = "nome" if "nome" in df_out.columns else "ra"
        df_out = df_out.sort_values([sort_col, "ano"])

        anos_disponiveis = sorted(df["ano"].unique().tolist()) if "ano" in df.columns else []

        # Converte para records com NaN -> None
        import math
        records = []
        for _, row in df_out.iterrows():
            rec = {}
            for col, val in row.items():
                if val is None:
                    rec[col] = None
                elif isinstance(val, (float, np.floating)):
                    if math.isnan(val) or math.isinf(val):
                        rec[col] = None
                    else:
                        rec[col] = round(float(val), 2)
                elif isinstance(val, (np.integer,)):
                    rec[col] = int(val)
                else:
                    rec[col] = val
            records.append(rec)

        return {
            "students": records,
            "total": len(records),
            "anos_disponiveis": anos_disponiveis,
        }
    except Exception as e:
        logger.error(f"Erro ao listar alunos: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/drift")
async def check_data_drift():
    """
    Verifica data drift entre anos do dataset.

    Compara distribuições dos dados de referência (2022) com os anos
    mais recentes (2023, 2024) usando Evidently AI.
    """
    try:
        df = _load_processed_data()
        if df.empty:
            return {"error": "Nenhum dado disponível"}

        from src.monitoring.drift import compare_year_drift

        feature_cols = ["INDE", "IEG", "IDA", "IPS", "IAA"]
        available = [c for c in feature_cols if c in df.columns]

        results = {}
        anos = sorted(df["ano"].unique().tolist())

        if len(anos) >= 2:
            ref_year = anos[0]
            for cur_year in anos[1:]:
                key = f"{int(ref_year)}_vs_{int(cur_year)}"
                try:
                    drift_result = compare_year_drift(
                        df, int(ref_year), int(cur_year),
                        feature_cols=available,
                    )
                    results[key] = drift_result
                except Exception as e:
                    logger.warning(f"Erro no drift {key}: {e}")
                    results[key] = {"error": str(e)}

        return {
            "drift_analysis": results,
            "features_monitored": available,
            "anos_analisados": [int(a) for a in anos],
        }
    except Exception as e:
        logger.error(f"Erro ao verificar drift: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/drift/report")
async def get_drift_report_html():
    """
    Gera e retorna relatório de drift em HTML (Evidently AI) com
    interpretação automática por LLM.

    Combina o dashboard visual do Evidently com uma análise textual
    gerada por IA explicando os resultados de drift.
    """
    from fastapi.responses import HTMLResponse

    try:
        df = _load_processed_data()
        if df.empty:
            return HTMLResponse("<h1>Nenhum dado disponível</h1>", status_code=404)

        from src.monitoring.drift import save_drift_report_html, check_drift

        feature_cols = ["INDE", "IEG", "IDA", "IPS", "IAA"]
        available = [c for c in feature_cols if c in df.columns]

        anos = sorted(df["ano"].unique().tolist())
        if len(anos) < 2:
            return HTMLResponse("<h1>Dados insuficientes para drift</h1>", status_code=404)

        ref_year = int(anos[0])
        cur_year = int(anos[-1])

        ref = df[df["ano"] == ref_year][available].dropna()
        cur = df[df["ano"] == cur_year][available].dropna()

        # 1) Gera relatório HTML do Evidently
        output_path = "reports/drift_report.html"
        save_drift_report_html(ref, cur, output_path=output_path, feature_cols=available)

        with open(output_path, "r", encoding="utf-8") as f:
            html_content = f.read()

        # 2) Obtém dados de drift estruturados para o LLM
        drift_result = check_drift(ref, cur, feature_cols=available)

        # 3) Estatísticas descritivas por feature
        stats_summary = {}
        for col in available:
            stats_summary[col] = {
                "ref_mean": round(float(ref[col].mean()), 2),
                "ref_std": round(float(ref[col].std()), 2),
                "cur_mean": round(float(cur[col].mean()), 2),
                "cur_std": round(float(cur[col].std()), 2),
                "ref_count": int(len(ref)),
                "cur_count": int(len(cur)),
            }

        # 4) Tenta gerar interpretação via LLM
        llm_html = _generate_drift_llm_analysis(
            drift_result, stats_summary, ref_year, cur_year
        )

        # 5) Injeta a análise LLM no topo do HTML do Evidently
        if llm_html:
            html_content = _inject_llm_section(html_content, llm_html)

        logger.info(f"Relatório drift HTML gerado: {ref_year} vs {cur_year}")
        return HTMLResponse(content=html_content)

    except Exception as e:
        logger.error(f"Erro ao gerar relatório drift HTML: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _generate_drift_llm_analysis(
    drift_result: dict,
    stats_summary: dict,
    ref_year: int,
    cur_year: int,
) -> str:
    """
    Gera análise interpretativa do drift via LLM (OpenRouter).

    Retorna HTML formatado ou string vazia se LLM não disponível.
    """
    try:
        api_key = settings.openrouter_api_key
        if not api_key or len(api_key) < 10:
            logger.warning("LLM não configurado — análise de drift apenas com Evidently")
            return ""

        from app.services.llm_service import LLMService
        llm = LLMService(api_key=api_key)

        # Monta contexto detalhado para o LLM
        feature_lines = []
        for feat, info in drift_result.get("feature_drift", {}).items():
            stat = stats_summary.get(feat, {})
            drift_flag = "SIM" if info.get("drift_detected") else "NÃO"
            feature_lines.append(
                f"  - {feat}: drift={drift_flag}, "
                f"p-value={info.get('drift_score', 'N/A')}, "
                f"teste={info.get('stattest_name', 'N/A')}, "
                f"média ref={stat.get('ref_mean', '?')} (±{stat.get('ref_std', '?')}), "
                f"média atual={stat.get('cur_mean', '?')} (±{stat.get('cur_std', '?')}), "
                f"n_ref={stat.get('ref_count', '?')}, n_atual={stat.get('cur_count', '?')}"
            )

        prompt = f"""Você é um especialista em MLOps e monitoramento de modelos de Machine Learning, no contexto da ONG Passos Mágicos que atua na educação de jovens em vulnerabilidade social.

CONTEXTO: Estamos monitorando data drift nos indicadores educacionais dos alunos entre {ref_year} (dados de referência) e {cur_year} (dados atuais).

INDICADORES MONITORADOS:
- INDE: Índice de Desenvolvimento Educacional (principal indicador composto)
- IEG: Indicador de Engajamento
- IDA: Indicador de Desempenho Acadêmico
- IPS: Indicador Psicossocial
- IAA: Indicador de Autoavaliação

RESULTADO DO DRIFT:
- Dataset drift detectado: {"SIM" if drift_result.get("dataset_drift") else "NÃO"}
- Features com drift: {drift_result.get("n_drifted_features", 0)} de {drift_result.get("n_total_features", 0)} ({drift_result.get("drift_share", 0):.1%})

DETALHAMENTO POR FEATURE:
{chr(10).join(feature_lines)}

Gere uma análise completa em PORTUGUÊS (formato Markdown) com as seguintes seções:

## 🔍 Resumo Executivo
[2-3 frases resumindo o resultado geral do drift e seu impacto potencial nos modelos]

## 📊 Análise por Indicador
[Para cada indicador com drift detectado, explique: o que mudou, possíveis causas educacionais, e impacto no modelo preditivo. Para os sem drift, mencione brevemente sua estabilidade]

## ⚠️ Impacto nos Modelos de ML
[Como o drift afeta o classificador de risco e o modelo de clustering. O modelo precisa ser retreinado? Com que urgência?]

## 💡 Recomendações
1. [Ação concreta 1 — ex: retreinar modelo, ajustar features, coletar mais dados]
2. [Ação concreta 2]
3. [Ação concreta 3]

## 📈 Próximos Passos
[Sugestões de monitoramento contínuo e thresholds]

Use linguagem técnica mas acessível. Relacione as mudanças estatísticas com o contexto educacional da ONG."""

        import asyncio

        # Executa a chamada LLM (sync wrapper para contexto async)
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    llm_response = pool.submit(
                        _call_llm_sync, llm, prompt, settings.openrouter_model
                    ).result(timeout=60)
            else:
                llm_response = asyncio.run(
                    llm.simple_completion(prompt, temperature=0.4)
                )
        except Exception:
            # Fallback: chamada síncrona direta
            llm_response = _call_llm_sync(llm, prompt, settings.openrouter_model)

        if llm_response:
            logger.info(f"Análise LLM do drift gerada: {len(llm_response)} chars")
            return llm_response

        return ""

    except Exception as e:
        logger.error(f"Erro ao gerar análise LLM do drift: {e}")
        return ""


def _call_llm_sync(llm, prompt: str, model: str) -> str:
    """Chamada síncrona ao LLM via httpx (evita bug pydantic/openai SDK)."""
    import httpx

    try:
        api_key = settings.openrouter_api_key
        response = httpx.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Você é um especialista em MLOps e monitoramento de modelos "
                            "da ONG Passos Mágicos. Gere análises claras e acionáveis "
                            "sobre data drift em indicadores educacionais."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.4,
                "max_tokens": 2000,
            },
            timeout=60.0,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        logger.error(f"Erro na chamada LLM sync: {e}")
        return ""


def _inject_llm_section(evidently_html: str, llm_markdown: str) -> str:
    """
    Injeta seção de análise LLM no HTML do Evidently.

    Converte o markdown do LLM em HTML e insere um painel estilizado
    antes do conteúdo do Evidently.
    """
    import re

    # Converte markdown básico para HTML
    llm_html = _markdown_to_html(llm_markdown)

    llm_panel = f"""
    <div id="llm-drift-analysis" style="
        max-width: 1200px;
        margin: 30px auto;
        padding: 30px 40px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        color: white;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
    ">
        <div style="display: flex; align-items: center; margin-bottom: 20px; border-bottom: 1px solid rgba(255,255,255,0.3); padding-bottom: 15px;">
            <span style="font-size: 2em; margin-right: 15px;">🤖</span>
            <div>
                <h1 style="margin: 0; font-size: 1.6em; font-weight: 700;">Análise Inteligente de Data Drift</h1>
                <p style="margin: 4px 0 0 0; font-size: 0.9em; opacity: 0.85;">
                    Interpretação gerada por IA ({settings.openrouter_model}) · ONG Passos Mágicos
                </p>
            </div>
        </div>
        <div style="
            background: rgba(255,255,255,0.12);
            border-radius: 12px;
            padding: 25px 30px;
            line-height: 1.7;
            font-size: 0.95em;
        ">
            {llm_html}
        </div>
        <div style="margin-top: 15px; text-align: right; font-size: 0.8em; opacity: 0.7;">
            ⚡ Análise automática — os dados estatísticos detalhados estão no relatório Evidently abaixo
        </div>
    </div>
    <hr style="max-width: 1200px; margin: 20px auto; border: none; border-top: 2px solid #eee;">
    """

    # Injeta após o <body> tag
    if "<body>" in evidently_html:
        evidently_html = evidently_html.replace("<body>", f"<body>{llm_panel}", 1)
    elif "<body " in evidently_html.lower():
        # body com atributos
        body_match = re.search(r"<body[^>]*>", evidently_html, re.IGNORECASE)
        if body_match:
            insert_pos = body_match.end()
            evidently_html = (
                evidently_html[:insert_pos] + llm_panel + evidently_html[insert_pos:]
            )
    else:
        # Fallback: coloca no início
        evidently_html = llm_panel + evidently_html

    return evidently_html


def _markdown_to_html(md: str) -> str:
    """Converte markdown básico para HTML estilizado."""
    import re

    html = md

    # Headers
    html = re.sub(
        r"^## (.+)$",
        r'<h2 style="margin-top: 25px; margin-bottom: 10px; font-size: 1.25em; '
        r'border-bottom: 1px solid rgba(255,255,255,0.2); padding-bottom: 8px;">\1</h2>',
        html,
        flags=re.MULTILINE,
    )
    html = re.sub(
        r"^### (.+)$",
        r'<h3 style="margin-top: 18px; margin-bottom: 8px; font-size: 1.1em;">\1</h3>',
        html,
        flags=re.MULTILINE,
    )

    # Bold
    html = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", html)

    # Italic
    html = re.sub(r"\*(.+?)\*", r"<em>\1</em>", html)

    # Inline code
    html = re.sub(
        r"`(.+?)`",
        r'<code style="background: rgba(255,255,255,0.2); padding: 2px 6px; '
        r'border-radius: 4px; font-size: 0.9em;">\1</code>',
        html,
    )

    # Numbered lists
    html = re.sub(
        r"^(\d+)\. (.+)$",
        r'<div style="margin: 6px 0 6px 20px;"><strong>\1.</strong> \2</div>',
        html,
        flags=re.MULTILINE,
    )

    # Bullet lists
    html = re.sub(
        r"^[-•] (.+)$",
        r'<div style="margin: 4px 0 4px 20px;">• \1</div>',
        html,
        flags=re.MULTILINE,
    )

    # Paragraphs (double newlines)
    html = re.sub(r"\n\n", r"</p><p style='margin: 10px 0;'>", html)
    html = f"<p style='margin: 10px 0;'>{html}</p>"

    return html
