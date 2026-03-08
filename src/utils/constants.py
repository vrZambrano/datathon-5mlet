"""
Constantes utilizadas em todo o projeto Passos Mágicos Datathon.
"""

# Features disponíveis em todos os anos (denominador comum)
COMMON_FEATURES = [
    "INDE",  # Índice do Desenvolvimento Educacional
    "IAN",   # Indicador de Adequação ao Nível
    "IDA",   # Indicador de Desempenho Acadêmico
    "IEG",   # Indicador de Engajamento
    "IAA",   # Indicador de Autoavaliação
    "IPS",   # Indicador Psicossocial
    "IPP",   # Indicador Psicopedagógico
    "IPV",   # Indicador de Ponto de Virada
]

# Colunas pedras (classificações)
PEDRA_COLUNAS = ["Pedra 20", "Pedra 21", "Pedra 22", "Pedra 23", "Pedra 24"]

# Valores possíveis para Pedra
PEDRA_VALORES = ["Quartzo", "Ágata", "Ametista", "Topázio"]

# Ordem de pedras (menor para maior)
PEDRA_ORDEM = {
    "Quartzo": 0,
    "Ágata": 1,
    "Ametista": 2,
    "Topázio": 3
}

# Features disponíveis apenas em 2022
FEATURES_2022_ONLY = [
    "Matem",      # Nota de Matemática
    "Portug",     # Nota de Português
    "Inglês",     # Nota de Inglês
    "Destaque IEG",  # Feedback qualitativo
    "Destaque IDA",
    "Destaque IPV"
]

# Features temporais para engenharia
TEMPORAL_FEATURES = [
    "delta_INDE",
    "delta_IEG",
    "delta_IDA",
    "tendencia_INDE",
    "anos_no_programa",
    "pedras_mudadas",
]

# Coluna de identificação
ID_COLUNA = "RA"

# Coluna de ano
ANO_COLUNA = "ano"

# Mapeamento de colunas para padronização
# NOTA: As chaves aqui estão em UPPERCASE pois clean_column_names() converte tudo.
# Mapeamento genérico (aplicado a todos os anos)
# 2022 usa "Nome", 2023/2024 usam "Nome Anonimizado" - unificamos para NOME
COLUNA_MAP = {
    "NOME ANONIMIZADO": "NOME",
}

# Mapeamento por ano: mapeia a coluna INDE/Pedra do ano corrente para nome padrão
COLUNA_MAP_PER_YEAR = {
    2022: {
        "INDE 22": "INDE",
        "PEDRA 22": "PEDRA",
    },
    2023: {
        "INDE 2023": "INDE",
        "PEDRA 2023": "PEDRA",
    },
    2024: {
        "INDE 2024": "INDE",
        "PEDRA 2024": "PEDRA",
    },
}

# Nomes de clusters (será definido após treinar o modelo)
CLUSTER_NAMES = {
    0: "Desmotivados Crônicos",
    1: "Engajados com Dificuldade",
    2: "Alto Desempenho",
    3: "Em Risco",
}

# Medianas populacionais para imputação de cold start
# Calculadas sobre a base PEDE 2020-2024 (denominador comum)
# Usadas quando um aluno novo não tem histórico para uma feature
FEATURE_MEDIANS = {
    "INDE": 6.354,
    "IEG": 6.472,
    "IDA": 6.751,
    "IPS": 5.893,
    "IAA": 6.123,
}

# Thresholds para classificação de risco
RISK_THRESHOLD_HIGH = 0.7
RISK_THRESHOLD_MEDIUM = 0.4

# Caminhos dos dados
DATA_DIR = "data"
RAW_DIR = f"{DATA_DIR}/raw"
PROCESSED_DIR = f"{DATA_DIR}/processed"
FEATURES_DIR = f"{DATA_DIR}/features"

# Arquivos CSV
CSV_2022 = f"{RAW_DIR}/BASE DE DADOS PEDE 2022 - DATATHON.csv"
CSV_2023 = f"{RAW_DIR}/BASE DE DADOS PEDE 2023 - DATATHON.csv"
CSV_2024 = f"{RAW_DIR}/BASE DE DADOS PEDE 2024 - DATATHON.csv"

# Separador dos CSVs (os dados usam ;)
CSV_SEPARATOR = ";"

# Encoding dos CSVs
CSV_ENCODING = "utf-8"
