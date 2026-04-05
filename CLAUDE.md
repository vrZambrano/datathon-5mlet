# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ML Engineering Datathon for the Passos Mágicos NGO. Predicts student performance risks, clusters student profiles, and generates AI-powered pedagogical reports via LLM.

**Stack:** Python, FastAPI, Streamlit, XGBoost, Scikit-learn, MLflow, Evidently AI, OpenRouter (Claude Sonnet 4)

## Development Commands

### Setup
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # Set OPENROUTER_API_KEY
```

### Training Models
```bash
python scripts/train_all.py              # Train all models
python scripts/train_all.py --skip-classifier  # Only clustering
python scripts/train_all.py --no-mlflow  # Skip MLflow tracking
```

### Running
```bash
uvicorn app.main:app --reload            # API on :8000
streamlit run frontend/main.py           # Frontend on :8501

cd docker && docker-compose up --build   # All services via Docker
```

### Testing
```bash
pytest tests/ -v
pytest tests/ --cov=app --cov=src --cov-report=term-missing
pytest tests/test_models/test_classifier.py -v   # Single file
```

Coverage must stay above 80% (`fail_under = 80` in `pyproject.toml`).

## Architecture

### Data Flow
1. **Ingestion:** CSVs (2022–2024) → `src/data/loader.py`
2. **Preprocessing:** Schema harmonization across years → `src/data/preprocessing.py`
3. **Feature Engineering:** Temporal features (deltas, trends) → `src/data/feature_engineering.py`
4. **Training:** XGBoost classifier + K-Means clustering → `src/models/`
5. **Inference:** API loads `.pkl` models → `/predict` endpoints
6. **Enrichment:** OpenRouter → LLM-generated pedagogical reports

### Module Organization
- **`src/data/`** — Data pipeline only (load → preprocess → feature_engineering). Pure pandas, no ML logic.
- **`src/models/`** — Training scripts. Each outputs a `.pkl` to `models/`.
- **`src/monitoring/`** — Data drift via Evidently AI.
- **`src/utils/constants.py`** — All shared constants (feature names, paths, medians, thresholds).
- **`app/`** — FastAPI app: `core/` (config/logger), `services/` (LLM), `routes/` (endpoints), `models/` (Pydantic schemas).
- **`frontend/`** — Streamlit dashboard calling the API via `requests`.
- **`prompts/`** — LLM prompt templates (edit here to change report format).

### Key Design Decisions

**Data Harmonization:** Only 2022 has complete data (subject grades, text feedbacks). Features common to all years: `INDE, IEG, IDA, IPS, IAA, IAN, IPP, IPV`. Column name normalization is handled by `clean_column_names()` + `COLUNA_MAP_PER_YEAR` in constants.

**Feature Engineering Focus:** Most predictive features are temporal — `delta_INDE`, `delta_IEG`, `delta_IDA`, `tendencia_INDE`, `pedras_mudadas_total`. Static values alone are weak predictors.

**Cold Start Handling:** New students lack deltas. Use `FEATURE_MEDIANS` from `src/utils/constants.py` for base features and `0` for deltas (semantically: "no change yet"). Do NOT use arbitrary values — XGBoost would misinterpret them as outliers.

**Pedra Encoding:** `encode_pedra()` returns `np.nan` (not `-1`) for null/invalid values. Using `-1` would corrupt the target variable (false drops) and inflate `pedras_mudadas_total`.

**Model Loading:** Models are loaded lazily on first request to speed up API startup.

**LLM Integration:** OpenRouter as gateway, async to prevent blocking. Primary model: `anthropic/claude-sonnet-4.6`.

## Data Notes

- CSV separator: `;` (not comma)
- Encoding: UTF-8
- "Pedra" classification order: Quartzo < Ágata < Ametista < Topázio
- Target variable is derived at training time: whether a student drops in Pedra level the next year
- `GroupShuffleSplit` by student RA prevents data leakage between train/test

## Important File Locations

- **Trained models:** `models/classifier.pkl`, `models/clustering_model.pkl`, `models/scaler.pkl`
- **Cluster labels:** `models/cluster_labels.json`
- **Prompt templates:** `prompts/relatorio_aluno.txt`
- **Raw data:** `data/raw/` (not tracked in git)
- **Settings:** `app/core/config.py` (uses `pydantic-settings`, reads from `.env`)

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/predict/risk` | Risk prediction → probability + BAIXO/MEDIO/ALTO |
| POST | `/predict/cluster` | Cluster assignment → cluster ID + name |
| POST | `/enrich/report` | LLM pedagogical report |
| GET | `/health/drift` | Evidently data drift metrics (2022 vs 2023/2024) |
| GET | `/health/stats` | Dashboard stats (INDE avg, pedra distribution) |
| GET | `/health/students` | Student list with indicators (`?ano=` filter) |

## Environment Variables

```bash
OPENROUTER_API_KEY=sk-or-v1-...        # Required for LLM features
OPENROUTER_MODEL=anthropic/claude-sonnet-4.6
CLASSIFIER_MODEL=models/classifier.pkl
CLUSTERING_MODEL=models/clustering_model.pkl
SCALER=models/scaler.pkl
MLFLOW_TRACKING_URI=http://localhost:5001
```
