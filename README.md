# 🎓 Passos Mágicos - Assistente Pedagógico Inteligente

Sistema de Machine Learning para a ONG Passos Mágicos, desenvolvido para o Datathon da Pos Tech Machine Learning Engineering - FIAP.

Vídeo: https://youtu.be/tbBCSgQ7yOk

GitHub: https://github.com/vrZambrano/datathon-5mlet



## 📋 Descrição

Este sistema utiliza Inteligência Artificial para auxiliar professores e educadores da ONG Passos Mágicos a:

- 📊 **Prever riscos de queda de desempenho** - Identificar alunos com risco de cair de nível (pedra)
- 👥 **Agrupar perfis de alunos** - Clusterização para ações pedagógicas personalizadas
- 📝 **Gerar relatórios personalizados** - Uso de LLM (Claude Sonnet 4) para criar recomendações pedagógicas

## 🏗️ Arquitetura

```
                              ┌─────────────────────────────────────────────────────────┐
                              │                    Docker Compose                       │
                              │                                                         │
┌──────────────┐              │  ┌──────────────┐     ┌──────────────┐                  │
│   Browser    │──────────────┼─▶│  Streamlit   │────▶│   FastAPI    │                  │
│   :8501      │              │  │  (Frontend)  │     │   (API)      │                  │
└──────────────┘              │  └──────────────┘     └──────┬───────┘                  │
                              │                              │                          │
                              │         ┌────────────────────┼────────────────┐         │
                              │         │                    │                │         │
                              │         ▼                    ▼                ▼         │
                              │  ┌──────────────┐     ┌──────────────┐ ┌──────────────┐ │
                              │  │  ML Models   │     │  Evidently   │ │    LLM       │ │
                              │  │  .pkl files  │     │  (Drift)     │ │  (OpenRouter)│ │
                              │  │  XGBoost     │     └──────────────┘ └──────────────┘ │
                              │  │  K-Means     │                             │         │
                              │  └──────────────┘                             │         │
                              │         │                                     ▼         │
                              │         │ logs metrics              ┌──────────────┐    │
                              │         └────────────────────────── │   Claude     │    │
                              │                                     │   Sonnet 4   │    │
                              │  ┌──────────────┐     ┌──────────────┐             │    │
                              │  │   MLflow     │     │  PostgreSQL  │ (external)  │    │
                              │  │   :5001      │     │   :5432      │             │    │
                              │  │  (Tracking)  │     │  (Opcional)  │             │    │
                              │  └──────────────┘     └──────────────┘             │    │
                              │         │                                          │    │
                              │         ▼                                          │    │
                              │  ┌──────────────┐                                  │    │
                              │  │    MinIO     │                                  │    │
                              │  │   :9001      │                                  │    │
                              │  │  (Artifacts) │                                  │    │
                              │  └──────────────┘                                  │    │
                              └─────────────────────────────────────────────────────────┘

Fluxo de dados:
1. Browser → Streamlit (UI) → FastAPI (REST API)
2. FastAPI → ML Models (.pkl) → Predições de risco/cluster
3. FastAPI → Evidently AI → Análise de data drift
4. FastAPI → OpenRouter → Claude Sonnet 4 (relatórios LLM)
5. Train script → MLflow → Métricas e parâmetros dos modelos
```

## 🛠️ Tecnologias

| Componente | Tecnologia |
|------------|-----------|
| **API** | FastAPI |
| **Frontend** | Streamlit |
| **ML** | XGBoost, Scikit-learn, K-Means |
| **MLOps** | MLflow, Docker |
| **Monitoramento** | Evidently AI (Data Drift) |
| **LLM** | Claude Sonnet 4 via OpenRouter |
| **Testes** | pytest (151 testes, 84% cobertura) |

## 📦 Estrutura do Projeto

```
datathon-5mlet/
├── app/                    # API FastAPI
│   ├── main.py            # Entry point
│   ├── routes/            # Endpoints (/predict, /enrich, /health)
│   ├── models/            # Pydantic schemas
│   ├── services/          # LLM service
│   └── core/              # Config e logger
├── frontend/              # Streamlit
│   └── main.py            # Aplicação web (dashboard, predições, relatórios)
├── src/                   # ML Pipeline
│   ├── data/              # Loader, preprocessing, feature engineering
│   ├── models/            # Train classifier e clustering
│   ├── monitoring/        # Data drift (Evidently AI)
│   └── utils/             # Constants e helpers
├── scripts/               # Scripts de execução
│   └── train_all.py       # Treina todos os modelos
├── tests/                 # Testes unitários (151 testes, 84% cobertura)
│   ├── test_data/         # Testes de loader, preprocessing, feature eng.
│   ├── test_models/       # Testes de classifier e clustering
│   ├── test_api/          # Testes de endpoints
│   ├── test_services/     # Testes de LLM service
│   ├── test_monitoring/   # Testes de drift
│   └── test_core/         # Testes de config e logger
├── docker/                # Docker configuration
│   ├── docker-compose.yml # Orquestração (5 serviços)
│   ├── Dockerfile.api     # Build da API
│   └── Dockerfile.streamlit # Build do frontend
├── data/                  # Datasets (não versionado)
├── models/                # Modelos treinados (.pkl)
├── prompts/               # Templates para LLM
├── requirements.txt       # Dependências Python
├── pyproject.toml         # Configuração pytest + coverage
└── .env.example           # Variáveis de ambiente
```

## 🚀 Guia de Instalação Rápida

### 1. Clonar o repositório

```bash
cd /path/to/datathon-5mlet
```

### 2. Criar ambiente virtual

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows
```

### 3. Instalar dependências

```bash
pip install -r requirements.txt
```

### 4. Configurar variáveis de ambiente

```bash
cp .env.example .env
# Edite .env com suas configurações, especialmente:
# OPENROUTER_API_KEY=sk-or-v1-sua-chave-aqui
```

### 5. Mover os datasets para o diretório data/raw/

```bash
# Coloque os CSVs em data/raw/
```

### 6. Treinar os modelos

```bash
python scripts/train_all.py
```

### 7. Executar a API

```bash
uvicorn app.main:app --reload
```

A API estará disponível em `http://localhost:8000`

### 8. Executar o Frontend (em outro terminal)

```bash
streamlit run frontend/main.py
```

O frontend estará disponível em `http://localhost:8501`

## 🐳 Docker

Para subir todos os serviços com Docker:

```bash
cd docker
docker-compose up --build
```

### Serviços

| Serviço | URL | Status |
|---------|-----|--------|
| **API** | http://localhost:8000 | ✅ Core |
| **Frontend** | http://localhost:8501 | ✅ Core |
| **MLflow** | http://localhost:5001 | ✅ Tracking de experimentos |
| **PostgreSQL** | localhost:5432 | ⚠️ Disponível (não utilizado atualmente) |
| **MinIO** | http://localhost:9001 | ⚠️ Storage S3 (opcional para artifacts) |

> **Nota:** A API atualmente lê dados diretamente dos arquivos `.pkl` e CSV. O PostgreSQL está configurado para uso futuro (persistência de predições, logs de usuários, etc.).

## 📚 Endpoints da API

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| GET | `/` | Informações da API |
| GET | `/health` | Health check |
| GET | `/health/metrics` | Uptime, status dos modelos, ambiente |
| GET | `/health/stats` | Estatísticas do dashboard (INDE médio, distribuição de pedras, clusters) |
| GET | `/health/students` | Lista de alunos com indicadores (filtro por `?ano=`) |
| GET | `/health/drift` | Análise de data drift entre anos (Evidently AI) |
| GET | `/health/quality` | Métricas de qualidade dos dados |
| GET | `/health/drift/report` | Relatório visual Evidently (HTML) |
| GET | `/health/drift/llm-analysis` | Análise de drift via LLM |
| POST | `/predict/risk` | Predição de risco de queda |
| POST | `/predict/cluster` | Predição de cluster/perfil |
| POST | `/enrich/report` | Geração de relatório LLM |
| GET | `/enrich/models` | Modelos LLM disponíveis |

### Exemplo de Uso - Predição de Risco

```bash
curl -X POST http://localhost:8000/predict/risk \
  -H "Content-Type: application/json" \
  -d '{
    "aluno_id": "RA-1234",
    "ano": 2024,
    "inde": 7.5,
    "ieg": 6.2,
    "ida": 7.0,
    "anos_no_programa": 3
  }'
```

**Resposta:**
```json
{
  "aluno_id": "RA-1234",
  "risco_probabilidade": 0.35,
  "risco_classe": "BAIXO",
  "vai_cair": false,
  "features_importantes": {
    "tendencia_INDE": 0.262,
    "delta_INDE": 0.137,
    "pedras_mudadas_total": 0.115
  }
}
```

### Exemplo de Uso - Clusterização

```bash
curl -X POST http://localhost:8000/predict/cluster \
  -H "Content-Type: application/json" \
  -d '{
    "aluno_id": "RA-1234",
    "inde": 7.5,
    "ieg": 6.2,
    "ida": 7.0,
    "ips": 5.5,
    "iaa": 6.0
  }'
```

### Exemplo de Uso - Health Check

```bash
# Status geral
curl http://localhost:8000/health

# Estatísticas do dashboard
curl http://localhost:8000/health/stats

# Análise de data drift
curl http://localhost:8000/health/drift

# Lista de alunos (filtrando por ano)
curl http://localhost:8000/health/students?ano=2024
```

## 📊 Métricas dos Modelos

### Classificador de Risco (XGBoost)
- **Modelo:** XGBoost Classifier
- **F1-Score:** 0.887
- **ROC AUC:** 0.938
- **Precisão:** ~0.89
- **Recall:** ~0.89
- **Features (11):** INDE, IEG, IDA, IPS, IAA, delta_INDE, delta_IEG, delta_IDA, anos_no_programa, tendencia_INDE, pedras_mudadas_total
- **Target:** Queda de pedra no próximo ano
- **Split:** GroupShuffleSplit (80/20) por aluno para evitar vazamento

### Clusterização de Perfis (K-Means)
- **Modelo:** K-Means (k=4)
- **Silhouette Score:** 0.352
- **Davies-Bouldin Index:** 1.02
- **Features (5):** INDE, IEG, IDA, IPS, IAA
- **Clusters identificados:**
  - **Desmotivados Crônicos** — INDE baixo, necessitam intervenção urgente
  - **Em Risco** — Desempenho abaixo da média, atenção redobrada
  - **Engajados com Dificuldade** — Boa vontade mas indicadores medianos
  - **Alto Desempenho** — INDE elevado, exemplos de sucesso

### Gerador de Relatórios (LLM)
- **Modelo:** Claude Sonnet 4 via OpenRouter
- **Formato:** Relatório pedagógico com resumo, análise de indicadores, pontos fortes/atenção e recomendações
- **Integração:** Recebe automaticamente risco predito + cluster do aluno

## 🧪 Testes

O projeto conta com **151 testes unitários** e **83.75% de cobertura** de código.

```bash
# Rodar todos os testes
pytest tests/ -v

# Verificar cobertura
pytest tests/ --cov=app --cov=src --cov-report=term-missing

# Gerar relatório HTML de cobertura
pytest tests/ --cov=app --cov=src --cov-report=html
```

### Cobertura por Módulo

| Módulo | Cobertura |
|--------|-----------|
| `src/data/` (loader, preprocessing, feature_engineering) | 93-98% |
| `src/models/` (classifier, clustering) | 71-81% |
| `src/monitoring/` (drift) | 97% |
| `app/routes/` (health, predict, cluster, enrich) | 78-81% |
| `app/models/schemas.py` | 95% |
| `app/services/llm_service.py` | 74% |
| `app/core/` (config, logger) | 87-100% |
| **TOTAL** | **83.75%** |

## 📈 Monitoramento e MLOps

### Data Drift (Evidently AI)

O sistema inclui monitoramento contínuo de data drift:

- **`GET /health/drift`** — Métricas de drift entre anos (2022 vs 2023, 2022 vs 2024)
- **`GET /health/drift/report`** — Relatório visual Evidently (HTML interativo)
- **`GET /health/drift/llm-analysis`** — Análise interpretativa via LLM
- **`GET /health/quality`** — Métricas de qualidade (missing values, outliers)
- **Features monitoradas:** INDE, IEG, IDA, IPS, IAA

### MLflow Tracking

Os modelos são versionados e rastreados via **MLflow**:

- **Experiment:** `passos_magicos`
- **Runs:** `risk_classifier`, `student_clustering`
- **Métricas logadas:** F1, Precision, Recall, ROC AUC (classificador); Silhouette, Davies-Bouldin (clustering)
- **Parâmetros logados:** Hiperparâmetros do modelo, features utilizadas
- **UI:** http://localhost:5001 (via Docker)

Variáveis de ambiente principais:

```bash
# API
API_HOST=0.0.0.0
API_PORT=8000

# MLflow
MLFLOW_TRACKING_URI=http://localhost:5001

# OpenRouter (LLM)
OPENROUTER_API_KEY=sk-or-v1-sua-chave
OPENROUTER_MODEL=anthropic/claude-sonnet-4.6

# Modelos
CLASSIFIER_MODEL=models/classifier.pkl
CLUSTERING_MODEL=models/clustering_model.pkl
SCALER=models/scaler.pkl
```

## � Acesso à Aplicação (Local)

Esta é uma entrega local. Após seguir o [guia de instalação](#-guia-de-instalação-rápida) ou subir via [Docker](#-docker):

| Serviço | URL | Descrição |
|---------|-----|-----------|
| **API** | http://localhost:8000 | FastAPI — endpoints REST |
| **Docs (Swagger)** | http://localhost:8000/docs | Documentação interativa da API |
| **Frontend** | http://localhost:8501 | Streamlit — dashboard, predições, relatórios |
| **MLflow** | http://localhost:5001 | Tracking de experimentos (via Docker) |
| **MinIO** | http://localhost:9001 | Console de artefatos (via Docker) |


## 👥 Time

Desenvolvido por Victor Zambrano para o Datathon de Machine Learning Engineering - FIAP.

## 📄 Licença

Este projeto foi desenvolvido para fins educacionais.
