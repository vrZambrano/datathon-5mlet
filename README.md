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
| **LLM** | Claude 3.5 Sonnet via OpenRouter |

## 📦 Estrutura do Projeto

```
datathon-5mlet/
├── app/                    # API FastAPI
│   ├── main.py            # Entry point
│   ├── routes/            # Endpoints (/predict, /enrich)
│   ├── models/            # Pydantic schemas
│   ├── services/          # ML e LLM services
│   └── core/              # Config e logger
├── frontend/              # Streamlit
│   └── main.py            # Aplicação web
├── src/                   # ML Pipeline
│   ├── data/              # Loader, preprocessing, feature engineering
│   ├── models/            # Train classifier e clustering
│   └── utils/             # Constants e helpers
├── scripts/               # Scripts de execução
│   └── train_all.py       # Treina todos os modelos
├── docker/                # Docker configuration
│   └── docker-compose.yml # Orquestração
├── data/                  # Datasets (não versionado)
├── models/                # Modelos treinados (.pkl)
├── prompts/               # Templates para LLM
├── requirements.txt       # Dependências Python
├── .env.example          # Variáveis de ambiente
└── PLANO_CLAUDE.md       # Plano detalhado do projeto
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
| POST | `/predict/risk` | Predição de risco de queda |
| POST | `/predict/cluster` | Predição de cluster/perfil |
| POST | `/enrich/report` | Geração de relatório LLM |

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

## 🧪 Testes

```bash
# Rodar todos os testes
pytest tests/ -v

# Verificar cobertura
pytest tests/ --cov=app --cov=src --cov-report=html
```

## 📊 Métricas dos Modelos

### Classificador de Risco
- **Modelo:** XGBoost Classifier
- **Features:** INDE, IEG, IDA, IPS, deltas, anos no programa
- **Target:** Queda de pedra no próximo ano

### Clusterização
- **Modelo:** K-Means (k=4)
- **Features:** INDE, IEG, IDA, IPS, IAA
- **Clusters:** Alto Desempenho, Engajados com Dificuldade, Em Risco, Desmotivados Crônicos

## 🔧 Configurações

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
