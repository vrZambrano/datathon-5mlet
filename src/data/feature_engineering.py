"""
Feature Engineering para dados da Passos Mágicos.

Este módulo cria features temporais e derivadas a partir dos dados brutos,
incluindo deltas, tendências e histórico de alunos.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from loguru import logger

from src.utils.constants import (
    ID_COLUNA,
    ANO_COLUNA,
    COMMON_FEATURES,
    TEMPORAL_FEATURES,
)


def calculate_deltas(
    df: pd.DataFrame,
    feature_cols: List[str],
    id_col: str = ID_COLUNA,
    ano_col: str = ANO_COLUNA
) -> pd.DataFrame:
    """
    Calcula deltas (variações) de features entre anos consecutivos.

    Args:
        df: DataFrame com dados de múltiplos anos
        feature_cols: Lista de features para calcular delta
        id_col: Coluna de identificação do aluno
        ano_col: Coluna de ano

    Returns:
        DataFrame com colunas delta_* adicionadas
    """
    df = df.copy()

    # Ordena por aluno e ano
    df = df.sort_values(by=[id_col, ano_col])

    # Calcula deltas para cada feature
    for feature in feature_cols:
        if feature not in df.columns:
            continue

        delta_col = f"delta_{feature}"
        # Delta = valor atual - valor anterior
        df[delta_col] = df.groupby(id_col)[feature].diff()

        # Para registros sem ano anterior (primeiro ano), delta é NaN
        # Isso será tratado depois (imputação)

    logger.info(f"Deltas calculados para {len(feature_cols)} features")
    return df


def calculate_inde_trend(
    df: pd.DataFrame,
    id_col: str = ID_COLUNA,
    ano_col: str = ANO_COLUNA,
    inde_col: str = "INDE"
) -> pd.DataFrame:
    """
    Calcula a tendência do INDE usando regressão linear simples.

    Tendência positiva = crescendo
    Tendência negativa = decaindo
    Tendência próxima de 0 = estável

    Args:
        df: DataFrame com dados de múltiplos anos
        id_col: Coluna de identificação do aluno
        ano_col: Coluna de ano
        inde_col: Coluna do INDE

    Returns:
        DataFrame com coluna tendencia_INDE
    """
    df = df.copy()

    def compute_trend(group: pd.DataFrame) -> float:
        """Computa tendência linear para um grupo."""
        group = group.dropna(subset=[inde_col]).sort_values(by=ano_col)

        if len(group) < 2:
            return 0.0

        x = group[ano_col].values
        y = group[inde_col].values

        # Regressão linear simples (coeficiente angular)
        if len(x) == 2:
            slope = (y[1] - y[0]) / (x[1] - x[0])
        else:
            # Para mais de 2 pontos, usa polyfit grau 1
            try:
                slope = np.polyfit(x, y, 1)[0]
            except:
                slope = 0.0

        return slope

    # Calcula tendência por aluno e mapeia de volta para cada linha
    trend_per_student = df.groupby(id_col).apply(compute_trend)
    df["tendencia_INDE"] = df[id_col].map(trend_per_student)

    logger.info("Tendência de INDE calculada")
    return df


def calculate_years_in_program(
    df: pd.DataFrame,
    id_col: str = ID_COLUNA,
    ano_col: str = ANO_COLUNA
) -> pd.DataFrame:
    """
    Calcula quantos anos cada aluno está no programa.

    Args:
        df: DataFrame com dados de múltiplos anos
        id_col: Coluna de identificação do aluno
        ano_col: Coluna de ano

    Returns:
        DataFrame com coluna anos_no_programa
    """
    df = df.copy()

    # Conta registros por aluno
    df["anos_no_programa"] = df.groupby(id_col).cumcount() + 1

    # Alternativa: usar o mínimo e máximo de ano
    # anos_no_programa = max_ano - min_ano + 1

    logger.info("Anos no programa calculados")
    return df


def calculate_pedra_changes(
    df: pd.DataFrame,
    id_col: str = ID_COLUNA,
    ano_col: str = ANO_COLUNA,
    pedra_codigo_col: str = "PEDRA_CODIGO"
) -> pd.DataFrame:
    """
    Calcula quantas vezes o aluno mudou de pedra (nível).

    Args:
        df: DataFrame com dados de múltiplos anos
        id_col: Coluna de identificação do aluno
        ano_col: Coluna de ano
        pedra_codigo_col: Coluna com código numérico da pedra

    Returns:
        DataFrame com coluna pedras_mudadas
    """
    df = df.copy()

    if pedra_codigo_col not in df.columns:
        logger.warning(f"Coluna {pedra_codigo_col} não encontrada")
        df["pedras_mudadas"] = 0
        return df

    # Ordena por aluno e ano
    df = df.sort_values(by=[id_col, ano_col])

    # Calcula mudanças — ignora registros com pedra NaN para não inflar diff
    df["pedras_mudadas"] = df.groupby(id_col)[pedra_codigo_col].diff().abs()
    # NaN diff (primeiro ano ou pedra nula) fica 0, não infla contagem
    df.loc[df[pedra_codigo_col].isna(), "pedras_mudadas"] = np.nan
    df["pedras_mudadas"] = df["pedras_mudadas"].fillna(0)

    # Cumulativo de mudanças
    df["pedras_mudadas_total"] = df.groupby(id_col)["pedras_mudadas"].cumsum()

    logger.info("Mudanças de pedra calculadas")
    return df


def identify_new_students(
    df: pd.DataFrame,
    id_col: str = ID_COLUNA,
    ano_col: str = ANO_COLUNA
) -> pd.DataFrame:
    """
    Identifica alunos ingressantes (apenas 1 registro).

    Args:
        df: DataFrame com dados de múltiplos anos
        id_col: Coluna de identificação do aluno
        ano_col: Coluna de ano

    Returns:
        DataFrame com coluna ingressante (bool)
    """
    df = df.copy()

    # Conta registros por aluno
    count_por_aluno = df.groupby(id_col).size()

    # Marca ingressantes
    df["ingressante"] = df[id_col].map(count_por_aluno) == 1

    logger.info(f"{df['ingressante'].sum()} alunos ingressantes identificados")
    return df


def create_historical_features(
    df: pd.DataFrame,
    id_col: str = ID_COLUNA,
    ano_col: str = ANO_COLUNA
) -> pd.DataFrame:
    """
    Cria features históricas para cada aluno.

    Inclui médias, mínimos, máximos de features ao longo dos anos.

    Args:
        df: DataFrame com dados de múltiplos anos
        id_col: Coluna de identificação do aluno
        ano_col: Coluna de ano

    Returns:
        DataFrame com features históricas adicionadas
    """
    df = df.copy()

    # Features para calcular estatísticas
    stat_features = ["INDE", "IEG", "IDA", "IPS"]
    stat_features = [f for f in stat_features if f in df.columns]

    for feature in stat_features:
        # Média histórica
        df[f"{feature}_media_hist"] = df.groupby(id_col)[feature].transform(
            lambda x: x.expanding().mean()
        )

        # Mínimo histórico
        df[f"{feature}_min_hist"] = df.groupby(id_col)[feature].transform(
            lambda x: x.expanding().min()
        )

        # Máximo histórico
        df[f"{feature}_max_hist"] = df.groupby(id_col)[feature].transform(
            lambda x: x.expanding().max()
        )

    logger.info("Features históricas criadas")
    return df


def create_all_temporal_features(
    df: pd.DataFrame,
    id_col: str = ID_COLUNA,
    ano_col: str = ANO_COLUNA
) -> pd.DataFrame:
    """
    Cria todas as features temporais de uma vez.

    Args:
        df: DataFrame harmonizado com múltiplos anos
        id_col: Coluna de identificação do aluno
        ano_col: Coluna de ano

    Returns:
        DataFrame com todas as features temporais
    """
    logger.info("Criando features temporais...")

    # Features para calcular deltas
    delta_features = ["INDE", "IEG", "IDA"]
    delta_features = [f for f in delta_features if f in df.columns]

    # Aplica todas as transformações
    df = calculate_deltas(df, delta_features, id_col, ano_col)
    df = calculate_inde_trend(df, id_col, ano_col)
    df = calculate_years_in_program(df, id_col, ano_col)
    df = calculate_pedra_changes(df, id_col, ano_col)
    df = identify_new_students(df, id_col, ano_col)
    df = create_historical_features(df, id_col, ano_col)

    # Lista de features criadas
    temporal_cols = [col for col in df.columns if col not in df.columns[:20]]  # Simplificação
    logger.info(f"Features temporais criadas. Total de colunas: {len(df.columns)}")

    return df


def create_target_variable(
    df: pd.DataFrame,
    id_col: str = ID_COLUNA,
    ano_col: str = ANO_COLUNA,
    pedra_col: str = "PEDRA_CODIGO",
    target_col: str = "target_queda_prox_ano"
) -> pd.DataFrame:
    """
    Cria variável target: se o aluno vai cair de pedra no ano seguinte.

    Args:
        df: DataFrame com dados de múltiplos anos
        id_col: Coluna de identificação do aluno
        ano_col: Coluna de ano
        pedra_col: Coluna com código da pedra
        target_col: Nome da coluna target

    Returns:
        DataFrame com coluna target adicionada
    """
    df = df.copy()

    if pedra_col not in df.columns:
        logger.warning(f"Coluna {pedra_col} não encontrada, não é possível criar target")
        df[target_col] = np.nan
        return df

    # Ordena por aluno e ano
    df = df.sort_values(by=[id_col, ano_col])

    # Shift para pegar pedra do próximo ano
    df["prox_ano_pedra"] = df.groupby(id_col)[pedra_col].shift(-1)

    # Target = 1 se caiu de pedra (próximo < atual)
    # Registros com pedra atual ou próxima NaN recebem target NaN (não é queda real)
    df[target_col] = np.nan
    mask_valid = df[pedra_col].notna() & df["prox_ano_pedra"].notna()
    df.loc[mask_valid, target_col] = (df.loc[mask_valid, "prox_ano_pedra"] < df.loc[mask_valid, pedra_col]).astype(float)

    # Remove linhas sem próximo ano (último ano de cada aluno — já NaN)
    df.loc[df["prox_ano_pedra"].isna(), target_col] = np.nan

    # Remove coluna temporária
    df = df.drop(columns=["prox_ano_pedra"])

    logger.info(f"Target criado: {df[target_col].sum()} alunos com queda (target=1)")
    return df


def prepare_features_for_modeling(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "target_queda_prox_ano"
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepara X e y para modelagem.

    Args:
        df: DataFrame com features e target
        feature_cols: Lista de colunas de features
        target_col: Nome da coluna target

    Returns:
        Tuple (X, y) onde X é DataFrame de features e y é Series com target
    """
    # Remove linhas com target NaN
    df_model = df.dropna(subset=[target_col]).copy()

    # Seleciona features
    X = df_model[feature_cols].copy()

    # Remove colunas com todos NaN
    X = X.dropna(axis=1, how="all")

    # Para features com alguns NaN, imputa com 0 (pode ser melhorado)
    X = X.fillna(0)

    # Target
    y = df_model[target_col]

    logger.info(f"Preparado para modelagem: X={X.shape}, y={y.shape}")
    logger.info(f"Target distribution: {y.value_counts().to_dict()}")

    return X, y
