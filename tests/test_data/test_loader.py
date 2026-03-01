"""
Testes para src/data/loader.py
"""

import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.data.loader import (
    load_csv,
    load_all_years,
    compare_schemas,
    get_common_columns,
    get_unique_columns_per_year,
    inspect_dataframe,
)


class TestLoadCsv:
    """Testes para load_csv."""

    def test_load_csv_with_valid_file(self, tmp_path):
        """Testa carregamento de CSV válido."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("RA;INDE;IEG\n1;5.0;6.0\n2;7.0;8.0", encoding="utf-8")

        df = load_csv(str(csv_file), separator=";", encoding="utf-8")

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "RA" in df.columns

    def test_load_csv_with_invalid_file(self):
        """Testa erro ao carregar arquivo inexistente."""
        with pytest.raises(Exception):
            load_csv("arquivo_inexistente.csv")

    def test_load_csv_semicolon_separator(self, tmp_path):
        """Testa que o separador padrão é ;."""
        csv_file = tmp_path / "semicolon.csv"
        csv_file.write_text("col1;col2\na;b\nc;d", encoding="utf-8")

        df = load_csv(str(csv_file))

        assert len(df.columns) == 2
        assert "col1" in df.columns


class TestLoadAllYears:
    """Testes para load_all_years."""

    @patch("src.data.loader.load_csv")
    @patch("src.data.loader.Path")
    def test_load_all_years_with_all_files(self, mock_path, mock_load_csv):
        """Testa carregamento quando todos os CSVs existem."""
        mock_path.return_value.exists.return_value = True
        mock_load_csv.return_value = pd.DataFrame({"RA": [1, 2], "INDE": [5.0, 6.0]})

        datasets = load_all_years()

        assert len(datasets) >= 1

    @patch("src.data.loader.Path")
    def test_load_all_years_no_files(self, mock_path):
        """Testa quando nenhum arquivo existe."""
        mock_path.return_value.exists.return_value = False

        datasets = load_all_years()

        assert len(datasets) == 0


class TestCompareSchemas:
    """Testes para compare_schemas."""

    def test_compare_schemas_basic(self):
        """Testa comparação de schemas entre DataFrames."""
        dfs = {
            2022: pd.DataFrame({"RA": [1], "INDE": [5.0], "IEG": [6.0]}),
            2023: pd.DataFrame({"RA": [1], "INDE": [5.0], "EXTRA": [1.0]}),
        }

        result = compare_schemas(dfs)

        assert isinstance(result, pd.DataFrame)
        assert "coluna" in result.columns
        assert "2022" in result.columns
        assert "2023" in result.columns

    def test_compare_schemas_column_presence(self):
        """Verifica presença correta de colunas."""
        dfs = {
            2022: pd.DataFrame({"A": [1], "B": [2]}),
            2023: pd.DataFrame({"B": [1], "C": [2]}),
        }

        result = compare_schemas(dfs)

        # B deve estar em ambos
        row_b = result[result["coluna"] == "B"]
        assert row_b["2022"].values[0] == True
        assert row_b["2023"].values[0] == True

        # A só em 2022
        row_a = result[result["coluna"] == "A"]
        assert row_a["2022"].values[0] == True
        assert row_a["2023"].values[0] == False


class TestGetCommonColumns:
    """Testes para get_common_columns."""

    def test_common_columns(self):
        """Testa obtenção de colunas comuns."""
        dfs = {
            2022: pd.DataFrame({"RA": [1], "INDE": [5.0], "IEG": [6.0]}),
            2023: pd.DataFrame({"RA": [1], "INDE": [5.0], "EXTRA": [1.0]}),
        }

        common = get_common_columns(dfs)

        assert "RA" in common
        assert "INDE" in common
        assert "IEG" not in common
        assert "EXTRA" not in common

    def test_common_columns_empty(self):
        """Testa com dicionário vazio."""
        result = get_common_columns({})
        assert result == []

    def test_common_columns_single_df(self):
        """Testa com um único DataFrame."""
        dfs = {2022: pd.DataFrame({"RA": [1], "INDE": [5.0]})}
        common = get_common_columns(dfs)
        assert "RA" in common
        assert "INDE" in common


class TestGetUniqueColumnsPerYear:
    """Testes para get_unique_columns_per_year."""

    def test_unique_columns(self):
        """Testa obtenção de colunas únicas por ano."""
        dfs = {
            2022: pd.DataFrame({"RA": [1], "ONLY_22": [5.0]}),
            2023: pd.DataFrame({"RA": [1], "ONLY_23": [5.0]}),
        }

        unique = get_unique_columns_per_year(dfs)

        assert 2022 in unique
        assert "ONLY_22" in unique[2022]
        assert 2023 in unique
        assert "ONLY_23" in unique[2023]


class TestInspectDataframe:
    """Testes para inspect_dataframe."""

    def test_inspect_basic(self):
        """Testa inspeção básica de DataFrame."""
        df = pd.DataFrame({
            "RA": [1, 2, 3],
            "INDE": [5.0, None, 7.0],
        })

        info = inspect_dataframe(df, 2022)

        assert info["ano"] == 2022
        assert info["n_linhas"] == 3
        assert info["n_colunas"] == 2
        assert "INDE" in info["missing_values"]

    def test_inspect_with_ra(self):
        """Testa inspeção com coluna RA."""
        df = pd.DataFrame({"RA": ["RA-1", "RA-1", "RA-2"], "INDE": [5.0, 6.0, 7.0]})

        info = inspect_dataframe(df, 2023)

        assert info["n_alunos_unicos"] == 2
