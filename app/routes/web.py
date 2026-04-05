"""
Rotas Web UI — HTMX + Jinja2 + Tailwind CSS.

Serve as páginas HTML do frontend moderno diretamente do FastAPI.
Os endpoints JSON da API ficam inalterados; este router chama os
serviços internos diretamente (sem round-trip HTTP).
"""

import asyncio
import math
from pathlib import Path
from typing import Optional

import joblib
import markdown as md_lib
import pandas as pd
from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, PlainTextResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from loguru import logger

from app.core.config import get_settings
from src.utils.constants import FEATURE_MEDIANS

router = APIRouter()
settings = get_settings()

templates = Jinja2Templates(directory="frontend_web/templates")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe(val, default=0.0):
    """Converte valor para float seguro."""
    try:
        if val is None:
            return default
        f = float(val)
        return default if (math.isnan(f) or math.isinf(f)) else f
    except Exception:
        return default


def _risk_class(prob: float) -> str:
    if prob >= 0.7:
        return "ALTO"
    elif prob >= 0.4:
        return "MEDIO"
    return "BAIXO"


def _tendencia_str(val: float) -> str:
    if val > 0.1:
        return "crescendo"
    elif val < -0.1:
        return "decrescendo"
    return "estável"


def _load_data():
    """Carrega dados processados via pipeline src/."""
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


def _get_students(ano: Optional[int] = None):
    """Retorna lista de alunos como list[dict]."""
    df = _load_data()
    if df.empty:
        return [], []

    anos_disp = sorted(df["ano"].unique().tolist())

    if ano:
        df_year = df[df["ano"] == ano].copy()
    else:
        df_year = df.copy()

    if df_year.empty:
        return [], anos_disp

    # Propaga NOME
    if "NOME" in df.columns:
        name_lookup = (
            df.dropna(subset=["NOME"])
            .drop_duplicates(subset=["RA"], keep="first")[["RA", "NOME"]]
        )
        df_year = df_year.drop(columns=["NOME"], errors="ignore")
        df_year = df_year.merge(name_lookup, on="RA", how="left")

    # Cluster via modelo
    try:
        km = joblib.load(Path(settings.clustering_model))
        scaler = joblib.load(Path(settings.scaler))
        import json
        labels_path = Path("models/cluster_labels.json")
        cluster_names = {}
        if labels_path.exists():
            with open(labels_path) as f:
                cluster_names = json.load(f).get("cluster_names", {})
        feats = ["INDE", "IEG", "IDA", "IPS", "IAA"]
        mask = df_year[feats].notna().all(axis=1)
        df_year["cluster"] = None
        if mask.any():
            X_s = scaler.transform(df_year.loc[mask, feats])
            preds = km.predict(X_s)
            df_year.loc[mask, "cluster"] = [
                cluster_names.get(str(c), f"Cluster {c}") for c in preds
            ]
    except Exception:
        df_year["cluster"] = None

    cols_map = {
        "RA": "ra", "NOME": "nome", "ano": "ano",
        "INDE": "inde", "IEG": "ieg", "IDA": "ida",
        "IPS": "ips", "IAA": "iaa", "IAN": "ian",
        "IPV": "ipv", "IPP": "ipp",
        "PEDRA": "pedra", "FASE": "fase", "TURMA": "turma",
        "tendencia_INDE": "tendencia_inde",
        "pedras_mudadas_total": "pedras_mudadas_total",
        "anos_no_programa": "anos_no_programa",
        "delta_INDE": "delta_inde",
        "delta_IEG": "delta_ieg",
        "delta_IDA": "delta_ida",
        "cluster": "cluster",
    }
    available = {k: v for k, v in cols_map.items() if k in df_year.columns}
    df_out = df_year[list(available.keys())].rename(columns=available)

    import numpy as np
    records = []
    for _, row in df_out.iterrows():
        rec = {}
        for col, val in row.items():
            if val is None:
                rec[col] = None
            elif isinstance(val, (float, np.floating)):
                rec[col] = None if (math.isnan(val) or math.isinf(val)) else round(float(val), 2)
            elif isinstance(val, (np.integer,)):
                rec[col] = int(val)
            else:
                rec[col] = val
        records.append(rec)

    sort_col = "nome" if "nome" in df_out.columns else "ra"
    records.sort(key=lambda r: (str(r.get(sort_col) or ""), r.get("ano") or 0))
    return records, anos_disp


def _get_stats():
    """Retorna stats para o dashboard."""
    import json, numpy as np
    df = _load_data()
    if df.empty:
        return {}

    df_latest = df.sort_values("ano").groupby("RA").last().reset_index()
    total_alunos = len(df_latest)
    inde_medio = round(float(df_latest["INDE"].dropna().mean()), 2) if "INDE" in df_latest.columns else None

    pedra_dist = {}
    if "PEDRA" in df_latest.columns:
        counts = df_latest["PEDRA"].dropna().value_counts()
        pedra_dist = {str(k): int(v) for k, v in counts.items()}

    cluster_dist = {}
    try:
        km = joblib.load(Path(settings.clustering_model))
        scaler = joblib.load(Path(settings.scaler))
        cluster_names = {}
        labels_path = Path("models/cluster_labels.json")
        if labels_path.exists():
            with open(labels_path) as f:
                cluster_names = json.load(f).get("cluster_names", {})
        feats = ["INDE", "IEG", "IDA", "IPS", "IAA"]
        avail = [c for c in feats if c in df_latest.columns]
        df_cl = df_latest[avail].dropna()
        if not df_cl.empty:
            X_s = scaler.transform(df_cl[feats])
            labels = km.predict(X_s)
            for cid in sorted(set(labels)):
                name = cluster_names.get(str(cid), f"Cluster {cid}")
                cluster_dist[name] = int((labels == cid).sum())
    except Exception:
        pass

    inde_por_ano = {}
    if "INDE" in df.columns and "ano" in df.columns:
        for ano, grp in df.groupby("ano"):
            vals = grp["INDE"].dropna()
            if not vals.empty:
                inde_por_ano[str(int(ano))] = round(float(vals.mean()), 2)

    risco_alto = 0
    try:
        model = joblib.load(Path(settings.classifier_model))
        feature_cols = ["INDE","IEG","IDA","IPS","IAA","delta_INDE","delta_IEG","delta_IDA",
                        "anos_no_programa","tendencia_INDE","pedras_mudadas_total"]
        avail = [c for c in feature_cols if c in df_latest.columns]
        df_pred = df_latest[avail].dropna()
        if not df_pred.empty:
            for c in feature_cols:
                if c not in df_pred.columns:
                    df_pred[c] = 0.0
            probs = model.predict_proba(df_pred[feature_cols])[:, 1]
            risco_alto = int((probs >= settings.risk_threshold).sum())
    except Exception:
        pass

    return {
        "total_alunos": total_alunos,
        "inde_medio": inde_medio,
        "risco_alto": risco_alto,
        "n_clusters": len(cluster_dist),
        "pedra_dist": pedra_dist,
        "cluster_dist": cluster_dist,
        "inde_por_ano": inde_por_ano,
    }


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

@router.get("/", response_class=HTMLResponse)
async def ui_root():
    return RedirectResponse(url="/ui/dashboard")


@router.get("/dashboard", response_class=HTMLResponse)
async def ui_dashboard(request: Request):
    return templates.TemplateResponse(
        "pages/dashboard.html",
        {"request": request, "active_page": "dashboard"},
    )


@router.get("/risk", response_class=HTMLResponse)
async def ui_risk(request: Request):
    return templates.TemplateResponse(
        "pages/risk.html",
        {"request": request, "active_page": "risk"},
    )


@router.get("/clustering", response_class=HTMLResponse)
async def ui_clustering(request: Request):
    return templates.TemplateResponse(
        "pages/clustering.html",
        {"request": request, "active_page": "clustering"},
    )


@router.get("/reports", response_class=HTMLResponse)
async def ui_reports(request: Request):
    return templates.TemplateResponse(
        "pages/reports.html",
        {"request": request, "active_page": "reports"},
    )


@router.get("/monitoring", response_class=HTMLResponse)
async def ui_monitoring(request: Request):
    return templates.TemplateResponse(
        "pages/monitoring.html",
        {"request": request, "active_page": "monitoring"},
    )


# ---------------------------------------------------------------------------
# Dashboard partials
# ---------------------------------------------------------------------------

@router.get("/partials/dashboard-kpis", response_class=HTMLResponse)
async def partial_dashboard_kpis(request: Request):
    stats = _get_stats()
    return templates.TemplateResponse(
        "partials/kpi_cards.html",
        {"request": request, "stats": stats},
    )


@router.get("/partials/chart-pedras", response_class=HTMLResponse)
async def partial_chart_pedras(request: Request):
    stats = _get_stats()
    pedra_order = ["Quartzo", "Ágata", "Ametista", "Topázio"]
    dist = stats.get("pedra_dist", {})
    labels = [p for p in pedra_order if p in dist]
    values = [dist[p] for p in labels]
    colors = {"Quartzo": "#9E9E9E", "Ágata": "#2196F3", "Ametista": "#9C27B0", "Topázio": "#FF9800"}
    bg_colors = [colors.get(p, "#ccc") for p in labels]
    return templates.TemplateResponse(
        "partials/chart_pedras.html",
        {"request": request, "labels": labels, "values": values, "colors": bg_colors},
    )


@router.get("/partials/chart-clusters", response_class=HTMLResponse)
async def partial_chart_clusters(request: Request):
    stats = _get_stats()
    dist = stats.get("cluster_dist", {})
    labels = list(dist.keys())
    values = list(dist.values())
    colors = ["#EF5350", "#FFA726", "#66BB6A", "#42A5F5"]
    return templates.TemplateResponse(
        "partials/chart_clusters.html",
        {"request": request, "labels": labels, "values": values, "colors": colors[:len(labels)]},
    )


@router.get("/partials/chart-inde", response_class=HTMLResponse)
async def partial_chart_inde(request: Request):
    stats = _get_stats()
    inde_por_ano = stats.get("inde_por_ano", {})
    labels = sorted(inde_por_ano.keys())
    values = [inde_por_ano[k] for k in labels]
    return templates.TemplateResponse(
        "partials/chart_inde.html",
        {"request": request, "labels": labels, "values": values},
    )


@router.get("/partials/student-table", response_class=HTMLResponse)
async def partial_student_table(
    request: Request,
    ano: Optional[int] = None,
    pedra: Optional[str] = None,
    cluster: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
):
    students, anos_disp = _get_students(ano)
    if pedra and pedra != "Todos":
        students = [s for s in students if s.get("pedra") == pedra]
    if cluster and cluster != "Todos":
        students = [s for s in students if s.get("cluster") == cluster]
    total = len(students)
    page_students = students[offset: offset + limit]
    return templates.TemplateResponse(
        "partials/student_table.html",
        {
            "request": request,
            "students": page_students,
            "total": total,
            "offset": offset,
            "limit": limit,
            "ano": ano,
            "pedra": pedra,
            "cluster": cluster,
        },
    )


# ---------------------------------------------------------------------------
# Student selector partials
# ---------------------------------------------------------------------------

@router.get("/partials/student-options", response_class=HTMLResponse)
async def partial_student_options(
    request: Request,
    ano: Optional[int] = None,
    pedra: Optional[str] = None,
    cluster: Optional[str] = None,
):
    students, _ = _get_students(ano)
    if pedra and pedra != "Todos":
        students = [s for s in students if s.get("pedra") == pedra]
    if cluster and cluster != "Todos":
        students = [s for s in students if s.get("cluster") == cluster]
    return templates.TemplateResponse(
        "partials/student_options.html",
        {"request": request, "students": students},
    )


@router.get("/partials/student-data", response_class=HTMLResponse)
async def partial_student_data(
    request: Request,
    ra: Optional[str] = None,
    ano: Optional[int] = None,
):
    """Retorna partial Alpine.js que pre-preenche os campos do formulário."""
    if not ra:
        return HTMLResponse("")
    students, _ = _get_students(ano)
    student = next((s for s in students if str(s.get("ra")) == str(ra)), None)
    if not student:
        return HTMLResponse("")
    return templates.TemplateResponse(
        "partials/student_data.html",
        {"request": request, "student": student},
    )


# ---------------------------------------------------------------------------
# Prediction partials
# ---------------------------------------------------------------------------

@router.post("/risk/predict", response_class=HTMLResponse)
async def web_predict_risk(
    request: Request,
    aluno_id: str = Form(default="MANUAL"),
    inde: float = Form(...),
    ieg: float = Form(default=FEATURE_MEDIANS["IEG"]),
    ida: float = Form(default=FEATURE_MEDIANS["IDA"]),
    ips: float = Form(default=FEATURE_MEDIANS["IPS"]),
    iaa: float = Form(default=FEATURE_MEDIANS["IAA"]),
    ian: float = Form(default=0.0),
    anos_no_programa: int = Form(default=1),
    tendencia_inde: float = Form(default=0.0),
    pedras_mudadas_total: float = Form(default=0.0),
    delta_inde: float = Form(default=0.0),
    delta_ieg: float = Form(default=0.0),
    delta_ida: float = Form(default=0.0),
):
    error = None
    result = None
    try:
        model = joblib.load(Path(settings.classifier_model))
        features = {
            "INDE": inde, "IEG": ieg, "IDA": ida, "IPS": ips, "IAA": iaa,
            "delta_INDE": delta_inde, "delta_IEG": delta_ieg, "delta_IDA": delta_ida,
            "anos_no_programa": anos_no_programa,
            "tendencia_INDE": tendencia_inde,
            "pedras_mudadas_total": pedras_mudadas_total,
        }
        X = pd.DataFrame([features])
        prob = float(model.predict_proba(X)[0, 1])
        vai_cair = bool(model.predict(X)[0])
        nivel = _risk_class(prob)
        feat_imp = None
        if hasattr(model, "feature_importances_"):
            imp = dict(zip(features.keys(), model.feature_importances_.astype(float)))
            feat_imp = dict(sorted(imp.items(), key=lambda x: -x[1])[:3])
        result = {"prob": prob, "nivel": nivel, "vai_cair": vai_cair, "feat_imp": feat_imp}
    except FileNotFoundError:
        error = "Modelo não encontrado. Execute scripts/train_all.py primeiro."
    except Exception as e:
        error = str(e)

    return templates.TemplateResponse(
        "partials/risk_result.html",
        {"request": request, "result": result, "error": error},
    )


@router.post("/cluster/predict", response_class=HTMLResponse)
async def web_predict_cluster(
    request: Request,
    aluno_id: str = Form(default="MANUAL"),
    inde: float = Form(...),
    ieg: float = Form(default=FEATURE_MEDIANS["IEG"]),
    ida: float = Form(default=FEATURE_MEDIANS["IDA"]),
    ips: float = Form(default=FEATURE_MEDIANS["IPS"]),
    iaa: float = Form(default=FEATURE_MEDIANS["IAA"]),
):
    error = None
    result = None
    try:
        import json
        km = joblib.load(Path(settings.clustering_model))
        scaler = joblib.load(Path(settings.scaler))
        labels_path = Path("models/cluster_labels.json")
        cluster_names = {}
        if labels_path.exists():
            with open(labels_path) as f:
                cluster_names = json.load(f).get("cluster_names", {})
        features = {"INDE": inde, "IEG": ieg, "IDA": ida, "IPS": ips, "IAA": iaa}
        X = pd.DataFrame([features])
        X_s = scaler.transform(X)
        cid = int(km.predict(X_s)[0])
        nome = cluster_names.get(str(cid), f"Cluster {cid}")
        result = {"cluster_id": cid, "cluster_nome": nome}
    except FileNotFoundError:
        error = "Modelo não encontrado. Execute scripts/train_all.py primeiro."
    except Exception as e:
        error = str(e)

    return templates.TemplateResponse(
        "partials/cluster_result.html",
        {"request": request, "result": result, "error": error},
    )


# ---------------------------------------------------------------------------
# Report multi-step
# ---------------------------------------------------------------------------

@router.post("/report/predictions", response_class=HTMLResponse)
async def web_report_predictions(
    request: Request,
    aluno_id: str = Form(default="MANUAL"),
    nome: str = Form(default=""),
    inde: float = Form(...),
    ieg: float = Form(default=FEATURE_MEDIANS["IEG"]),
    ida: float = Form(default=FEATURE_MEDIANS["IDA"]),
    ips: float = Form(default=FEATURE_MEDIANS["IPS"]),
    iaa: float = Form(default=FEATURE_MEDIANS["IAA"]),
    ian: float = Form(default=0.0),
    ipp: float = Form(default=0.0),
    ipv: float = Form(default=0.0),
    anos_no_programa: int = Form(default=1),
    tendencia_inde: float = Form(default=0.0),
    pedras_mudadas_total: float = Form(default=0.0),
    delta_inde: float = Form(default=0.0),
    delta_ieg: float = Form(default=0.0),
    delta_ida: float = Form(default=0.0),
    pedra: str = Form(default=""),
    idade: int = Form(default=15),
    feedback: str = Form(default=""),
):
    risk_result = None
    cluster_result = None
    try:
        model = joblib.load(Path(settings.classifier_model))
        features = {
            "INDE": inde, "IEG": ieg, "IDA": ida, "IPS": ips, "IAA": iaa,
            "delta_INDE": delta_inde, "delta_IEG": delta_ieg, "delta_IDA": delta_ida,
            "anos_no_programa": anos_no_programa,
            "tendencia_INDE": tendencia_inde,
            "pedras_mudadas_total": pedras_mudadas_total,
        }
        X = pd.DataFrame([features])
        prob = float(model.predict_proba(X)[0, 1])
        nivel = _risk_class(prob)
        feat_imp = None
        if hasattr(model, "feature_importances_"):
            imp = dict(zip(features.keys(), model.feature_importances_.astype(float)))
            feat_imp = dict(sorted(imp.items(), key=lambda x: -x[1])[:3])
        risk_result = {"prob": prob, "nivel": nivel, "feat_imp": feat_imp}
    except Exception as e:
        risk_result = {"error": str(e)}

    try:
        import json
        km = joblib.load(Path(settings.clustering_model))
        scaler = joblib.load(Path(settings.scaler))
        labels_path = Path("models/cluster_labels.json")
        cluster_names = {}
        if labels_path.exists():
            with open(labels_path) as f:
                cluster_names = json.load(f).get("cluster_names", {})
        X_cl = pd.DataFrame([{"INDE": inde, "IEG": ieg, "IDA": ida, "IPS": ips, "IAA": iaa}])
        cid = int(km.predict(scaler.transform(X_cl))[0])
        cluster_result = {"cluster_nome": cluster_names.get(str(cid), f"Cluster {cid}")}
    except Exception as e:
        cluster_result = {"error": str(e)}

    # Dados completos para passar para a etapa LLM via form hidden fields
    form_data = {
        "aluno_id": aluno_id, "nome": nome, "inde": inde, "ieg": ieg, "ida": ida,
        "ips": ips, "iaa": iaa, "ian": ian, "ipp": ipp, "ipv": ipv,
        "anos_no_programa": anos_no_programa, "tendencia_inde": tendencia_inde,
        "pedras_mudadas_total": pedras_mudadas_total, "pedra": pedra,
        "idade": idade, "feedback": feedback,
        "risco_percentual": round(risk_result.get("prob", 0) * 100) if risk_result else 0,
        "risco_classe": risk_result.get("nivel", "N/A") if risk_result else "N/A",
        "cluster_nome": cluster_result.get("cluster_nome", "") if cluster_result else "",
    }

    return templates.TemplateResponse(
        "partials/report_predictions.html",
        {
            "request": request,
            "risk": risk_result,
            "cluster": cluster_result,
            "form_data": form_data,
        },
    )


@router.post("/report/llm", response_class=HTMLResponse)
async def web_report_llm(
    request: Request,
    aluno_id: str = Form(default="MANUAL"),
    nome: str = Form(default=""),
    inde: float = Form(...),
    ieg: float = Form(default=0.0),
    ida: float = Form(default=0.0),
    ips: float = Form(default=0.0),
    iaa: float = Form(default=0.0),
    ian: float = Form(default=0.0),
    ipp: float = Form(default=0.0),
    ipv: float = Form(default=0.0),
    anos_no_programa: int = Form(default=1),
    tendencia_inde: float = Form(default=0.0),
    pedra: str = Form(default=""),
    idade: int = Form(default=15),
    feedback: str = Form(default=""),
    risco_percentual: float = Form(default=0.0),
    risco_classe: str = Form(default="N/A"),
    cluster_nome: str = Form(default=""),
):
    error = None
    relatorio_html = None
    relatorio_raw = None
    try:
        if not settings.openrouter_api_key or len(settings.openrouter_api_key) < 10:
            raise ValueError("OPENROUTER_API_KEY não configurada.")

        from app.services.llm_service import LLMService
        llm = LLMService(api_key=settings.openrouter_api_key)
        aluno_data = {
            "nome": nome or aluno_id,
            "idade": idade,
            "pedra": pedra,
            "inde": inde, "ieg": ieg, "ida": ida, "ips": ips, "iaa": iaa,
            "ian": ian, "ipv": ipv, "ipp": ipp,
            "anos_no_programa": anos_no_programa,
            "tendencia_inde": _tendencia_str(tendencia_inde),
            "cluster_nome": cluster_nome or "Não identificado",
            "risco_percentual": int(risco_percentual),
            "risco_classe": risco_classe,
            "feedback_texto": feedback or "Não disponível",
        }
        try:
            relatorio_raw = await asyncio.wait_for(
                llm.generate_student_report(aluno_data),
                timeout=85.0,
            )
        except asyncio.TimeoutError:
            raise ValueError("Tempo limite excedido ao gerar relatório (>85s).")

        # Limpa cabeçalhos duplicados
        lines = relatorio_raw.splitlines()
        seen = set()
        clean = []
        for line in lines:
            if line.startswith("# "):
                if line in seen:
                    continue
                seen.add(line)
            clean.append(line)
        relatorio_raw = "\n".join(clean)
        relatorio_html = md_lib.markdown(relatorio_raw, extensions=["nl2br"])
    except Exception as e:
        error = str(e)

    return templates.TemplateResponse(
        "partials/report_content.html",
        {
            "request": request,
            "relatorio_html": relatorio_html,
            "relatorio_raw": relatorio_raw,
            "aluno_id": aluno_id,
            "nome": nome,
            "error": error,
        },
    )


@router.get("/report/download", response_class=PlainTextResponse)
async def web_report_download(
    aluno_id: str = "aluno",
    conteudo: str = "",
):
    return PlainTextResponse(
        content=conteudo,
        headers={"Content-Disposition": f'attachment; filename="relatorio_{aluno_id}.txt"'},
    )


# ---------------------------------------------------------------------------
# Monitoring partials
# ---------------------------------------------------------------------------

@router.get("/partials/drift-table", response_class=HTMLResponse)
async def partial_drift_table(request: Request, comparacao: Optional[str] = None):
    from app.routes.health import _load_processed_data
    from src.monitoring.drift import compare_year_drift

    error = None
    drift_data = {}
    comparacoes = []
    selected = comparacao

    try:
        df = _load_processed_data()
        feature_cols = ["INDE", "IEG", "IDA", "IPS", "IAA"]
        available = [c for c in feature_cols if c in df.columns]
        anos = sorted(df["ano"].unique().tolist())

        if len(anos) >= 2:
            ref = anos[0]
            for cur in anos[1:]:
                key = f"{int(ref)}_vs_{int(cur)}"
                comparacoes.append(key)
                try:
                    drift_data[key] = compare_year_drift(df, int(ref), int(cur), feature_cols=available)
                except Exception as e:
                    drift_data[key] = {"error": str(e)}

        if not selected and comparacoes:
            selected = comparacoes[-1]
    except Exception as e:
        error = str(e)

    selected_data = drift_data.get(selected, {}) if selected else {}
    return templates.TemplateResponse(
        "partials/drift_table.html",
        {
            "request": request,
            "comparacoes": comparacoes,
            "selected": selected,
            "drift": selected_data,
            "error": error,
        },
    )


@router.get("/partials/quality-table", response_class=HTMLResponse)
async def partial_quality_table(request: Request):
    from app.routes.health import _load_processed_data

    error = None
    quality = {}
    try:
        df = _load_processed_data()
        feature_cols = ["INDE", "IEG", "IDA", "IPS", "IAA", "IAN", "IPV", "IPP"]
        available = [c for c in feature_cols if c in df.columns]
        total = len(df)
        n_dup = int(df.duplicated().sum())
        total_cells = len(df) * len(df.columns)
        total_missing = df.isnull().sum().sum()
        missing_rate = total_missing / total_cells if total_cells > 0 else 0

        feature_stats = {}
        for col in available:
            col_data = df[col]
            feature_stats[col] = {
                "mean": round(float(col_data.mean()), 2) if not col_data.isna().all() else 0,
                "std": round(float(col_data.std()), 2) if not col_data.isna().all() else 0,
                "min": round(float(col_data.min()), 2) if not col_data.isna().all() else 0,
                "max": round(float(col_data.max()), 2) if not col_data.isna().all() else 0,
                "missing_pct": round(col_data.isna().mean() * 100, 1),
            }
        quality = {
            "total": total,
            "n_duplicados": n_dup,
            "missing_rate": round(missing_rate * 100, 1),
            "n_colunas": len(df.columns),
            "feature_stats": feature_stats,
        }
    except Exception as e:
        error = str(e)

    return templates.TemplateResponse(
        "partials/quality_table.html",
        {"request": request, "quality": quality, "error": error},
    )


@router.get("/partials/drift-llm", response_class=HTMLResponse)
async def partial_drift_llm(request: Request):
    from app.routes.health import _load_processed_data, _generate_drift_llm_analysis
    from src.monitoring.drift import check_drift

    error = None
    llm_html = None
    try:
        df = _load_processed_data()
        feature_cols = ["INDE", "IEG", "IDA", "IPS", "IAA"]
        available = [c for c in feature_cols if c in df.columns]
        anos = sorted(df["ano"].unique().tolist())
        if len(anos) < 2:
            raise ValueError("Dados insuficientes para análise de drift.")
        ref_year, cur_year = int(anos[0]), int(anos[-1])
        ref = df[df["ano"] == ref_year][available].dropna()
        cur = df[df["ano"] == cur_year][available].dropna()
        drift_result = check_drift(ref, cur, feature_cols=available)
        stats_summary = {
            col: {
                "ref_mean": round(float(ref[col].mean()), 2),
                "ref_std": round(float(ref[col].std()), 2),
                "cur_mean": round(float(cur[col].mean()), 2),
                "cur_std": round(float(cur[col].std()), 2),
                "ref_count": len(ref),
                "cur_count": len(cur),
            }
            for col in available
        }
        raw = _generate_drift_llm_analysis(drift_result, stats_summary, ref_year, cur_year)
        if raw:
            llm_html = md_lib.markdown(raw, extensions=["nl2br"])
        else:
            error = "LLM não configurado. Configure OPENROUTER_API_KEY."
    except Exception as e:
        error = str(e)

    return templates.TemplateResponse(
        "partials/drift_llm.html",
        {"request": request, "llm_html": llm_html, "error": error},
    )
