# 🎓 Passos Mágicos - Assistente Pedagógico Inteligente

Sistema de Machine Learning para a ONG Passos Mágicos, desenvolvido para o Datathon de Engenharia de Machine Learning.

## 📋 Descrição

Este sistema utiliza Inteligência Artificial para auxiliar professores e educadores da ONG Passos Mágicos a:

- 📊 **Prever riscos de queda de desempenho** - Identificar alunos com risco de cair de nível (pedra)
- 👥 **Agrupar perfis de alunos** - Clusterização para ações pedagógicas personalizadas
- 📝 **Gerar relatórios personalizados** - Uso de LLM (Claude 3.5 Sonnet) para criar recomendações pedagógicas

## 🏗️ Arquitetura

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  Streamlit UI    │────▶│   FastAPI        │────▶│   ML Models      │
│  (Frontend)      │     │   (API Gateway)  │     │  XGBoost/K-Means │
└──────────────────┘     └──────────────────┘     └──────────────────┘
                                │
                                ▼
                         ┌──────────────────┐
                         │  LLM Service     │
                         │  (OpenRouter +   │
                         │   Claude 3.5)    │
                         └──────────────────┘
```

## 🛠️ Tecnologias

| Componente | Tecnologia |
|------------|-----------|
| **API** | FastAPI |
| **Frontend** | Streamlit |
| **ML** | XGBoost, Scikit-learn, K-Means |
| **MLOps** | MLflow, Docker |
| **Monitoramento** | Evidently AI (Data Drift) |
| **LLM** | Claude 3.5 Sonnet via OpenRouter |
| **Testes** | pytest (141 testes, 85%+ cobertura) |

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
├── tests/                 # Testes unitários (141 testes, 85%+ cobertura)
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

Serviços:
- API: http://localhost:8000
- Frontend: http://localhost:8501
- MLflow: http://localhost:5001
- MinIO: http://localhost:9001

## 📚 Endpoints da API

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| GET | `/` | Informações da API |
| GET | `/health` | Health check |
| GET | `/health/metrics` | Uptime, status dos modelos, ambiente |
| GET | `/health/stats` | Estatísticas do dashboard (INDE médio, distribuição de pedras, clusters) |
| GET | `/health/students` | Lista de alunos com indicadores (filtro por `?ano=`) |
| GET | `/health/drift` | Análise de data drift entre anos (Evidently AI) |
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
  "risco_probabilidade": 0.73,
  "risco_classe": "ALTO",
  "vai_cair": true,
  "features_importantes": {
    "delta_INDE": -0.45,
    "IEG": 0.32
  },
  "timestamp": "2025-02-28T10:30:00Z"
}
```

## 📊 Métricas dos Modelos

### Classificador de Risco (XGBoost)
- **Modelo:** XGBoost Classifier
- **F1-Score:** 0.844
- **ROC AUC:** 0.888
- **Precisão:** 0.848
- **Recall:** 0.844
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
- **Modelo:** Claude 3.5 Sonnet via OpenRouter
- **Formato:** Relatório pedagógico com resumo, análise de indicadores, pontos fortes/atenção e recomendações
- **Integração:** Recebe automaticamente risco predito + cluster do aluno

## 🧪 Testes

O projeto conta com **141 testes unitários** e **85.95% de cobertura** de código.

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
| `src/data/` (loader, preprocessing, feature_engineering) | 96-98% |
| `src/models/` (classifier, clustering) | 70-80% |
| `src/monitoring/` (drift) | 83% |
| `app/routes/` (health, predict, cluster, enrich) | 58-86% |
| `app/models/schemas.py` | 95% |
| `app/services/llm_service.py` | 81% |
| **TOTAL** | **85.95%** |

## 📈 Monitoramento e Data Drift

O sistema inclui monitoramento contínuo usando **Evidently AI**:

- **Endpoint:** `GET /health/drift` — Compara distribuições entre anos (2022 vs 2023, 2022 vs 2024)
- **Features monitoradas:** INDE, IEG, IDA, IPS, IAA
- **Métricas:** Dataset Drift (proporção de features com drift), Drift por feature (p-value e estatística de teste)
- **Relatório HTML:** Gerado via `src/monitoring/drift.py` para análise visual

Variáveis de ambiente principais:

```bash
# API
API_HOST=0.0.0.0
API_PORT=8000

# MLflow
MLFLOW_TRACKING_URI=http://localhost:5001

# OpenRouter (LLM)
OPENROUTER_API_KEY=sk-or-v1-sua-chave
OPENROUTER_MODEL=anthropic/claude-3.5-sonnet

# Modelos
CLASSIFIER_MODEL=models/classifier.pkl
CLUSTERING_MODEL=models/clustering_model.pkl
SCALER=models/scaler.pkl
```

## 📖 Documentação Adicional

- [PLANO_CLAUDE.md](PLANO_CLAUDE.md) - Plano detalhado de implementação
- [instrucoes.md](instrucoes.md) - Instruções originais do Datathon

## 👥 Time

Desenvolvido para o Datathon de Engenharia de Machine Learning - 5MLET.

## 📄 Licença

Este projeto foi desenvolvido para fins educacionais.

## 🙏 Agradecimentos

- ONG Passos Mágicos
- Professores da pós-graduação
- OpenRouter pelo acesso aos LLMs
