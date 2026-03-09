# Project Guidelines — Datathon Passos Mágicos

ML system for the Passos Mágicos NGO: risk prediction (XGBoost), student clustering (K-Means), and LLM-generated pedagogical reports. All documentation, docstrings, and UI text are in **Portuguese (pt-BR)**. For comprehensive project context, see `CLAUDE.md` at the repo root.

## Architecture

Two decoupled top-level packages:

- **`app/`** — FastAPI runtime (inference API + Streamlit proxy endpoints)
- **`src/`** — Offline data pipeline & training code

Models are serialized as `.pkl` via `joblib` in `models/`. They are **lazy-loaded per request** in route handlers (no startup preloading, no `Depends()` injection). See `app/routes/predict.py::get_classifier_model()` for the pattern.

Routes are organized as separate `APIRouter()` instances in `app/routes/` and mounted with prefixes in `app/main.py`. The `health.py` route is large (~700 lines) because it also serves student data and dashboard statistics.

LLM integration uses **raw `httpx`** to call OpenRouter (not the OpenAI SDK). Prompt templates live in `prompts/`.

## Code Style

- **Logging:** Always `from loguru import logger` — never stdlib `logging`.
- **Config:** `from app.core.config import get_settings` — pydantic-settings `BaseSettings` with `@lru_cache`. All settings have defaults; app starts without `.env`.
- **Type hints** on all functions. Pydantic `Field(...)` with `description`, `ge`/`le` for schemas.
- **Docstrings:** Google-style (`Args:`, `Returns:`) in Portuguese.
- **Section separators:** `# ====...====` blocks inside larger files.
- **Imports:** stdlib → third-party → local. In-function imports for lazy/conditional deps (e.g., MLflow).
- **MLflow is optional:** Guarded with `try: import mlflow; MLFLOW_AVAILABLE = True except ImportError`.
- **Constants** are centralized in `src/utils/constants.py` (feature lists, pedra mappings, CSV config, paths, thresholds).

## Build and Test

```bash
# Setup
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

# Train models (all flags optional)
python scripts/train_all.py [--skip-classifier] [--skip-clustering] [--no-mlflow]

# Run API
uvicorn app.main:app --reload

# Run frontend (separate terminal)
streamlit run frontend/main.py

# Tests (asyncio_mode=auto — no @pytest.mark.asyncio needed)
pytest tests/ -v
pytest tests/ --cov=app --cov=src --cov-report=term-missing  # coverage ≥ 80%

# Docker (5 services: api, streamlit, postgres, mlflow, minio)
cd docker && docker-compose up --build
```

## Project Conventions

- **CSV files** use semicolon separator (`;`), UTF-8 encoding. Column names vary by year and are harmonized in `src/data/preprocessing.py` via uppercasing and accent removal.
- **Data pipeline** is a linear chain: `loader.py` → `preprocessing.py` → `feature_engineering.py`. Each step is pure pandas, no ML logic.
- **Target variable** (risk of "Pedra" level drop) is created on-the-fly during training, never pre-stored.
- **Cold start handling:** New students get zeros for delta features and `anos_no_programa=1`.
- **No Poetry** — `pyproject.toml` holds only pytest/coverage config. Use `pip install -r requirements.txt`.
- **`models/` directory is git-tracked** with `.pkl` files. Cluster labels are separate in `models/cluster_labels.json`.
- **`sys.path.insert(0, ...)`** hack is used in `scripts/train_all.py` and `tests/conftest.py` for root-relative imports.
- **Test fixtures** in `tests/conftest.py` build synthetic DataFrames with `np.random.seed()` and progressively enrich them. API tests use `httpx.AsyncClient` with `ASGITransport`.

## Key Files

| Purpose | Location |
|---------|----------|
| API entry point | `app/main.py` |
| All Pydantic schemas | `app/models/schemas.py` |
| Settings + env vars | `app/core/config.py`, `.env.example` |
| All project constants | `src/utils/constants.py` |
| Feature engineering | `src/data/feature_engineering.py` |
| Risk classifier training | `src/models/train_classifier.py` |
| Clustering training | `src/models/train_clustering.py` |
| LLM service | `app/services/llm_service.py` |
| Drift monitoring | `src/monitoring/drift.py` |
| Test fixtures | `tests/conftest.py` |
| LLM prompt template | `prompts/relatorio_aluno.txt` |

## Integration Points

- **OpenRouter API** — LLM reports via `OPENROUTER_API_KEY` env var. Model configurable via `OPENROUTER_MODEL` (default: `anthropic/claude-3.5-sonnet`).
- **MLflow** — Optional experiment tracking at `MLFLOW_TRACKING_URI` (default: `http://localhost:5001`). MinIO for artifact storage.
- **Evidently AI** — Data drift detection comparing distributions across years (`src/monitoring/drift.py`), served via `GET /health/drift`.
- **Note:** `DATABASE_URL` and `sqlalchemy` exist in config/requirements but are currently unused at runtime — API reads `.pkl` and CSV files directly.

## Security

- **`OPENROUTER_API_KEY`** is the only secret. Never log, commit, or expose it in responses. Always load from `.env` via `get_settings()`.
- **CORS is wide open** (`allow_origins=["*"]`) in `app/main.py`. Acceptable for local/dev; must be restricted before any public deployment.
- **No authentication** on any endpoint — the API is designed for internal/local use behind a private network.
- **Exception handler** in `app/main.py` hides error details in production (`settings.is_development` gate). Preserve this behavior.

## Further Context

See `CLAUDE.md` at the project root for deeper details on data notes, API endpoint reference, ML model metrics, and the full implementation plan in `PLANO_PROJETO.md`.
