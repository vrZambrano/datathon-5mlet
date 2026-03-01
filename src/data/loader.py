"""
Carregamento de dados dos CSVs da Passos Mágicos.

Este módulo fornece funções para carregar e inspecionar os datasets
de 2022, 2023 e 2024.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from loguru import logger

from src.utils.constants import (
    CSV_SEPARATOR,
    CSV_ENCODING,
    CSV_2022,
    CSV_2023,
    CSV_2024,
    ID_COLUNA,
)


def load_csv(filepath: str, separator: str = CSV_SEPARATOR, encoding: str = CSV_ENCODING) -> pd.DataFrame:
    """
    Carrega um arquivo CSV.

    Args:
        filepath: Caminho do arquivo CSV
        separator: Separador utilizado no CSV (padrão: ;)
        encoding: Encoding do arquivo (padrão: utf-8)

    Returns:
        DataFrame com os dados carregados
    """
    logger.info(f"Carregando CSV: {filepath}")
    try:
        df = pd.read_csv(filepath, sep=separator, encoding=encoding)
        logger.info(f"CSV carregado: {df.shape[0]} linhas, {df.shape[1]} colunas")
        return df
    except Exception as e:
        logger.error(f"Erro ao carregar CSV {filepath}: {e}")
        raise


def load_2022() -> pd.DataFrame:
    """Carrega o dataset de 2022."""
    return load_csv(CSV_2022)


def load_2023() -> pd.DataFrame:
    """Carrega o dataset de 2023."""
    return load_csv(CSV_2023)


def load_2024() -> pd.DataFrame:
    """Carrega o dataset de 2024."""
    return load_csv(CSV_2024)


def load_all_years() -> Dict[int, pd.DataFrame]:
    """
    Carrega todos os datasets disponíveis (2022, 2023, 2024).

    Returns:
        Dicionário com ano como chave e DataFrame como valor
    """
    datasets = {}

    # Verifica quais arquivos existem
    for ano, caminho in [(2022, CSV_2022), (2023, CSV_2023), (2024, CSV_2024)]:
        if Path(caminho).exists():
            try:
                datasets[ano] = load_csv(caminho)
            except Exception as e:
                logger.warning(f"Não foi possível carregar dados de {ano}: {e}")
        else:
            logger.warning(f"Arquivo não encontrado: {caminho}")

    logger.info(f"Datasets carregados para os anos: {list(datasets.keys())}")
    return datasets


def compare_schemas(dfs: Dict[int, pd.DataFrame]) -> pd.DataFrame:
    """
    Compara os schemas (colunas) dos DataFrames de diferentes anos.

    Args:
        dfs: Dicionário com ano -> DataFrame

    Returns:
        DataFrame com comparação das colunas presentes em cada ano
    """
    anos = sorted(dfs.keys())
    todas_colunas = set()

    for df in dfs.values():
        todas_colunas.update(df.columns)

    # Cria matriz de presença de colunas
    comparacao = []
    for coluna in sorted(todas_colunas):
        row = {"coluna": coluna}
        for ano in anos:
            row[str(ano)] = coluna in dfs[ano].columns
        comparacao.append(row)

    result = pd.DataFrame(comparacao)
    logger.info(f"Comparação de schemas gerada: {len(todas_colunas)} colunas únicas")
    return result


def get_common_columns(dfs: Dict[int, pd.DataFrame]) -> List[str]:
    """
    Retorna as colunas presentes em todos os DataFrames.

    Args:
        dfs: Dicionário com ano -> DataFrame

    Returns:
        Lista de colunas comuns a todos os anos
    """
    if not dfs:
        return []

    # Começa com colunas do primeiro DataFrame
    common = set(dfs[list(dfs.keys())[0]].columns)

    # Intersecção com os demais
    for df in dfs.values():
        common = common.intersection(set(df.columns))

    result = sorted(list(common))
    logger.info(f"Colunas comuns a todos os anos: {len(result)}")
    return result


def get_unique_columns_per_year(dfs: Dict[int, pd.DataFrame]) -> Dict[int, List[str]]:
    """
    Retorna colunas únicas de cada ano (não presentes nos outros).

    Args:
        dfs: Dicionário com ano -> DataFrame

    Returns:
        Dicionário com ano -> lista de colunas únicas
    """
    unique_per_year = {}
    all_columns = {}

    # Coleta todas as colunas por ano
    for ano, df in dfs.items():
        all_columns[ano] = set(df.columns)

    # Encontra as únicas
    for ano, cols in all_columns.items():
        other_years_cols = set()
        for other_ano, other_cols in all_columns.items():
            if other_ano != ano:
                other_years_cols.update(other_cols)

        unique = cols - other_years_cols
        if unique:
            unique_per_year[ano] = sorted(list(unique))
            logger.info(f"Colunas únicas em {ano}: {len(unique)}")

    return unique_per_year


def inspect_dataframe(df: pd.DataFrame, ano: int) -> Dict:
    """
    Retorna informações estatísticas sobre o DataFrame.

    Args:
        df: DataFrame a ser analisado
        ano: Ano dos dados (para log)

    Returns:
        Dicionário com estatísticas
    """
    info = {
        "ano": ano,
        "n_linhas": df.shape[0],
        "n_colunas": df.shape[1],
        "colunas": list(df.columns),
        "tipos": df.dtypes.to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "missing_pct": (df.isnull().sum() / len(df) * 100).to_dict(),
    }

    # Verifica se existe coluna RA
    if ID_COLUNA in df.columns:
        info["n_alunos_unicos"] = df[ID_COLUNA].nunique()

    logger.info(f"Inspeção {ano}: {info['n_linhas']} registros, {info.get('n_alunos_unicos', 'N/A')} alunos únicos")
    return info
