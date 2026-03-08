"""
Pré-processamento e harmonização dos datasets da Passos Mágicos.

Este módulo fornece funções para limpar, padronizar e unir os datasets
de diferentes anos (2022, 2023, 2024).
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from loguru import logger

from src.utils.constants import (
    COMMON_FEATURES,
    FEATURES_2022_ONLY,
    PEDRA_VALORES,
    PEDRA_ORDEM,
    ID_COLUNA,
    ANO_COLUNA,
    COLUNA_MAP,
    COLUNA_MAP_PER_YEAR,
)


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpa e padroniza nomes das colunas.

    - Remove espaços extras
    - Converte para uppercase
    - Remove acentos (básico)

    Args:
        df: DataFrame com colunas originais

    Returns:
        DataFrame com colunas padronizadas
    """
    df = df.copy()

    # Remove espaços e padroniza
    novas_colunas = []
    for col in df.columns:
        col = str(col).strip()
        col = col.upper()
        # Remove acentos básicos
        col = col.replace("Á", "A").replace("Ã", "A").replace("É", "E").replace("Í", "I")
        col = col.replace("Ó", "O").replace("Õ", "O").replace("Ú", "U").replace("Ç", "C")
        novas_colunas.append(col)

    df.columns = novas_colunas
    return df


def apply_column_mapping(df: pd.DataFrame, mapping: Dict[str, str] = None) -> pd.DataFrame:
    """
    Aplica mapeamento de colunas para padronização.

    Args:
        df: DataFrame com colunas originais
        mapping: Dicionário de mapeamento (opcional, usa COLUNA_MAP se não fornecido)

    Returns:
        DataFrame com colunas renomeadas
    """
    if mapping is None:
        mapping = COLUNA_MAP

    df = df.copy()
    rename_map = {}

    for col in df.columns:
        if col in mapping:
            rename_map[col] = mapping[col]

    if rename_map:
        df = df.rename(columns=rename_map)
        logger.debug(f"Colunas renomeadas: {rename_map}")

    return df


def extract_pedra_value(pedra_str: str) -> Optional[str]:
    """
    Extrai o valor da pedra de uma string.

    Args:
        pedra_str: String contendo o nome da pedra

    Returns:
        Valor da pedra (Quartzo, Ágata, Ametista, Topázio) ou None
    """
    if pd.isna(pedra_str):
        return None

    pedra_str = str(pedra_str).strip().upper()

    # Remove todos os acentos para comparação uniforme
    import unicodedata
    def _strip_accents(s: str) -> str:
        return "".join(
            c for c in unicodedata.normalize("NFD", s)
            if unicodedata.category(c) != "Mn"
        )

    pedra_norm = _strip_accents(pedra_str)

    for valor in PEDRA_VALORES:
        valor_norm = _strip_accents(valor.upper())
        if valor_norm in pedra_norm:
            return valor

    return None


def encode_pedra(pedra: Optional[str]) -> float:
    """
    Converte o valor da pedra para um código numérico.

    Args:
        pedra: Valor da pedra (Quartzo, Ágata, Ametista, Topázio)

    Returns:
        Código numérico (0-3) ou NaN se nulo/inválido
    """
    if pd.isna(pedra) or pedra is None:
        return np.nan
    code = PEDRA_ORDEM.get(pedra)
    if code is None:
        return np.nan
    return float(code)


def clean_numeric_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Limpa colunas numéricas, substituindo vírgulas por pontos e convertendo.

    Args:
        df: DataFrame com colunas numéricas
        columns: Lista de colunas para limpar

    Returns:
        DataFrame com colunas convertidas para numeric
    """
    df = df.copy()

    for col in columns:
        if col not in df.columns:
            continue

        # Converte para string, substitui vírgula por ponto
        df[col] = df[col].astype(str).str.replace(",", ".", regex=False)

        # Converte para numeric
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def handle_missing_values(df: pd.DataFrame, strategy: str = "mean") -> pd.DataFrame:
    """
    Trata valores missing conforme estratégia definida.

    Args:
        df: DataFrame com valores missing
        strategy: Estratégia ('mean', 'median', 'mode', 'zero')

    Returns:
        DataFrame com valores tratados
    """
    df = df.copy()

    # Colunas numéricas
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            if strategy == "mean":
                df[col].fillna(df[col].mean(), inplace=True)
            elif strategy == "median":
                df[col].fillna(df[col].median(), inplace=True)
            elif strategy == "zero":
                df[col].fillna(0, inplace=True)

    # Colunas categóricas
    categorical_cols = df.select_dtypes(include=["object"]).columns

    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna("Desconhecido", inplace=True)

    logger.debug(f"Missing values tratados com estratégia: {strategy}")
    return df


def add_year_column(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    Adiciona coluna de ano ao DataFrame.

    Args:
        df: DataFrame original
        year: Ano dos dados

    Returns:
        DataFrame com coluna ANO adicionada
    """
    df = df.copy()
    df[ANO_COLUNA] = year
    return df


def harmonize_single_year(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    Aplica toda a harmonização para um único ano.

    Args:
        df: DataFrame do ano
        year: Ano dos dados

    Returns:
        DataFrame harmonizado
    """
    logger.info(f"Harmonizando dados de {year}")

    # Limpa nomes das colunas (uppercase + remove acentos)
    df = clean_column_names(df)

    # Aplica mapeamento genérico
    df = apply_column_mapping(df)

    # Aplica mapeamento por ano (INDE 2023 -> INDE, PEDRA 2024 -> PEDRA, etc.)
    year_map = COLUNA_MAP_PER_YEAR.get(year, {})
    if year_map:
        # As chaves do year_map já estão em uppercase
        df = apply_column_mapping(df, mapping=year_map)

    # Remove colunas duplicadas (mantém primeira ocorrência)
    df = df.loc[:, ~df.columns.duplicated()]

    # Converte colunas numéricas que podem estar como string
    numeric_candidates = ["INDE", "IEG", "IDA", "IPS", "IAA", "IPP", "IPV", "IAN",
                          "INDE 22", "INDE 23", "INDE 2023", "INDE 2024"]
    for col in numeric_candidates:
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].astype(str).str.replace(",", ".", regex=False)
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Adiciona coluna de ano
    df = add_year_column(df, year)

    return df


def harmonize_datasets(
    df_2022: Optional[pd.DataFrame] = None,
    df_2023: Optional[pd.DataFrame] = None,
    df_2024: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Harmoniza e une os datasets de diferentes anos em um único DataFrame.

    Colunas que não existem em todos os anos são mantidas, mas com valores NaN
    para os anos onde não existem.

    Args:
        df_2022: DataFrame de 2022 (opcional)
        df_2023: DataFrame de 2023 (opcional)
        df_2024: DataFrame de 2024 (opcional)

    Returns:
        DataFrame único com dados de todos os anos
    """
    dfs = []

    # Processa cada ano disponível
    if df_2022 is not None:
        dfs.append(harmonize_single_year(df_2022, 2022))

    if df_2023 is not None:
        dfs.append(harmonize_single_year(df_2023, 2023))

    if df_2024 is not None:
        dfs.append(harmonize_single_year(df_2024, 2024))

    if not dfs:
        raise ValueError("Pelo menos um DataFrame deve ser fornecido")

    # Une todos os DataFrames
    result = pd.concat(dfs, ignore_index=True, sort=False)

    logger.info(f"Datasets harmonizados: {len(result)} registros totais")
    logger.info(f"Colunas finais: {len(result.columns)}")

    return result


def filter_common_features_only(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filtra o DataFrame para manter apenas as features comuns a todos os anos.

    Args:
        df: DataFrame completo

    Returns:
        DataFrame apenas com features comuns
    """
    # Adiciona ID e ANO se existirem
    cols_to_keep = [ID_COLUNA, ANO_COLUNA] + COMMON_FEATURES

    # Verifica quais colunas existem
    cols_to_keep = [col for col in cols_to_keep if col in df.columns]

    result = df[cols_to_keep].copy()
    logger.info(f"DataFrame filtrado para features comuns: {len(cols_to_keep)} colunas")

    return result


def normalize_pedra_column(df: pd.DataFrame, pedra_col: str = "PEDRA") -> pd.DataFrame:
    """
    Normaliza a coluna de pedra, extraindo valores válidos.

    Args:
        df: DataFrame com coluna de pedra
        pedra_col: Nome da coluna de pedra

    Returns:
        DataFrame com coluna PEDRA normalizada e PEDRA_CODIGO
    """
    df = df.copy()

    if pedra_col not in df.columns:
        logger.warning(f"Coluna {pedra_col} não encontrada")
        return df

    # Extrai valor da pedra
    df["PEDRA"] = df[pedra_col].apply(extract_pedra_value)

    # Cria código numérico
    df["PEDRA_CODIGO"] = df["PEDRA"].apply(encode_pedra)

    # Remove coluna original se diferente
    if pedra_col != "PEDRA":
        df = df.drop(columns=[pedra_col])

    logger.info("Coluna PEDRA normalizada")
    return df
