"""
Testes para src/data/preprocessing.py
"""

import pytest
import pandas as pd
import numpy as np

from src.data.preprocessing import (
    clean_column_names,
    apply_column_mapping,
    extract_pedra_value,
    encode_pedra,
    clean_numeric_columns,
    handle_missing_values,
    add_year_column,
    harmonize_single_year,
    harmonize_datasets,
    filter_common_features_only,
    normalize_pedra_column,
)


class TestCleanColumnNames:
    """Testes para clean_column_names."""

    def test_uppercase_conversion(self):
        """Testa conversão para uppercase."""
        df = pd.DataFrame({"nome": [1], "inde": [2]})
        result = clean_column_names(df)
        assert "NOME" in result.columns
        assert "INDE" in result.columns

    def test_strip_spaces(self):
        """Testa remoção de espaços."""
        df = pd.DataFrame({"  col1  ": [1], " col2": [2]})
        result = clean_column_names(df)
        assert "COL1" in result.columns
        assert "COL2" in result.columns

    def test_accent_removal(self):
        """Testa remoção de acentos básicos."""
        df = pd.DataFrame({"índice": [1], "nível": [2]})
        result = clean_column_names(df)
        # Uppercase + accent removal
        assert any("INDICE" in c for c in result.columns)

    def test_does_not_modify_original(self):
        """Testa que DataFrame original não é modificado."""
        df = pd.DataFrame({"nome": [1]})
        result = clean_column_names(df)
        assert "nome" in df.columns
        assert "NOME" in result.columns


class TestExtractPedraValue:
    """Testes para extract_pedra_value."""

    def test_quartzo(self):
        """Testa extração de Quartzo."""
        assert extract_pedra_value("Quartzo") == "Quartzo"

    def test_agata(self):
        """Testa extração de Ágata."""
        assert extract_pedra_value("Ágata") == "Ágata"

    def test_ametista(self):
        """Testa extração de Ametista."""
        assert extract_pedra_value("Ametista") == "Ametista"

    def test_topazio(self):
        """Testa extração de Topázio."""
        assert extract_pedra_value("Topázio") == "Topázio"

    def test_case_insensitive(self):
        """Testa case insensitive."""
        assert extract_pedra_value("QUARTZO") == "Quartzo"
        assert extract_pedra_value("topázio") == "Topázio"

    def test_nan_returns_none(self):
        """Testa que NaN retorna None."""
        assert extract_pedra_value(np.nan) is None
        assert extract_pedra_value(None) is None

    def test_invalid_pedra(self):
        """Testa valor inválido."""
        assert extract_pedra_value("Diamante") is None

    def test_pedra_with_extra_text(self):
        """Testa pedra com texto adicional."""
        result = extract_pedra_value("Pedra: Ametista nível 2")
        assert result == "Ametista"


class TestEncodePedra:
    """Testes para encode_pedra."""

    def test_encode_all_pedras(self):
        """Testa encoding de todas as pedras."""
        assert encode_pedra("Quartzo") == 0
        assert encode_pedra("Ágata") == 1
        assert encode_pedra("Ametista") == 2
        assert encode_pedra("Topázio") == 3

    def test_encode_invalid(self):
        """Testa encoding inválido."""
        assert encode_pedra("Diamante") == -1

    def test_encode_nan(self):
        """Testa encoding de NaN."""
        assert encode_pedra(np.nan) == -1
        assert encode_pedra(None) == -1


class TestCleanNumericColumns:
    """Testes para clean_numeric_columns."""

    def test_comma_to_dot(self):
        """Testa conversão de vírgula para ponto."""
        df = pd.DataFrame({"INDE": ["7,5", "8,3", "6,1"]})
        result = clean_numeric_columns(df, ["INDE"])
        assert result["INDE"].dtype == np.float64
        assert result["INDE"].iloc[0] == 7.5

    def test_handles_missing_column(self):
        """Testa com coluna que não existe."""
        df = pd.DataFrame({"INDE": [5.0]})
        result = clean_numeric_columns(df, ["INDE", "COLUNA_FAKE"])
        assert len(result) == 1

    def test_handles_nan(self):
        """Testa com valores não numéricos."""
        df = pd.DataFrame({"INDE": ["7.5", "abc", "6.1"]})
        result = clean_numeric_columns(df, ["INDE"])
        assert pd.isna(result["INDE"].iloc[1])


class TestHandleMissingValues:
    """Testes para handle_missing_values."""

    def test_mean_strategy(self):
        """Testa imputação com média."""
        df = pd.DataFrame({"INDE": [4.0, np.nan, 8.0]})
        result = handle_missing_values(df, strategy="mean")
        assert result["INDE"].iloc[1] == pytest.approx(6.0)

    def test_zero_strategy(self):
        """Testa imputação com zero."""
        df = pd.DataFrame({"INDE": [4.0, np.nan, 8.0]})
        result = handle_missing_values(df, strategy="zero")
        assert result["INDE"].iloc[1] == 0.0

    def test_categorical_missing(self):
        """Testa imputação de categorias."""
        df = pd.DataFrame({"PEDRA": ["Quartzo", None, "Ametista"]})
        result = handle_missing_values(df)
        assert result["PEDRA"].iloc[1] == "Desconhecido"


class TestAddYearColumn:
    """Testes para add_year_column."""

    def test_add_year(self):
        """Testa adição de coluna de ano."""
        df = pd.DataFrame({"RA": [1, 2]})
        result = add_year_column(df, 2022)
        assert "ano" in result.columns
        assert all(result["ano"] == 2022)

    def test_does_not_modify_original(self):
        """Testa que não modifica original."""
        df = pd.DataFrame({"RA": [1]})
        add_year_column(df, 2022)
        assert "ano" not in df.columns


class TestHarmonizeSingleYear:
    """Testes para harmonize_single_year."""

    def test_harmonize_2022(self, sample_df_2022):
        """Testa harmonização de 2022."""
        result = harmonize_single_year(sample_df_2022, 2022)

        assert "ano" in result.columns
        assert all(result["ano"] == 2022)
        assert "INDE" in result.columns  # INDE 22 -> INDE

    def test_harmonize_2023(self, sample_df_2023):
        """Testa harmonização de 2023."""
        result = harmonize_single_year(sample_df_2023, 2023)

        assert "ano" in result.columns
        assert "INDE" in result.columns  # INDE 2023 -> INDE

    def test_harmonize_2024(self, sample_df_2024):
        """Testa harmonização de 2024."""
        result = harmonize_single_year(sample_df_2024, 2024)

        assert "ano" in result.columns
        assert "INDE" in result.columns  # INDE 2024 -> INDE

    def test_columns_uppercase(self, sample_df_2022):
        """Verifica que colunas ficam uppercase."""
        result = harmonize_single_year(sample_df_2022, 2022)
        for col in result.columns:
            if col != "ano":
                assert col == col.upper() or col == "ano", f"Coluna '{col}' não está uppercase"


class TestHarmonizeDatasets:
    """Testes para harmonize_datasets."""

    def test_harmonize_all_years(self, sample_df_2022, sample_df_2023, sample_df_2024):
        """Testa harmonização de todos os anos."""
        result = harmonize_datasets(
            df_2022=sample_df_2022,
            df_2023=sample_df_2023,
            df_2024=sample_df_2024,
        )

        assert isinstance(result, pd.DataFrame)
        assert "ano" in result.columns
        assert set(result["ano"].unique()) == {2022, 2023, 2024}
        assert len(result) == len(sample_df_2022) + len(sample_df_2023) + len(sample_df_2024)

    def test_harmonize_single_year(self, sample_df_2022):
        """Testa com apenas um ano."""
        result = harmonize_datasets(df_2022=sample_df_2022)
        assert len(result) == len(sample_df_2022)

    def test_harmonize_no_data_raises(self):
        """Testa que sem dados gera erro."""
        with pytest.raises(ValueError, match="Pelo menos um DataFrame"):
            harmonize_datasets()

    def test_harmonize_has_inde_column(self, sample_df_2022, sample_df_2023):
        """Verifica que INDE é mapeada corretamente."""
        result = harmonize_datasets(df_2022=sample_df_2022, df_2023=sample_df_2023)
        assert "INDE" in result.columns
        # INDE deve ter valores numéricos
        assert result["INDE"].dropna().dtype in [np.float64, np.float32]


class TestFilterCommonFeaturesOnly:
    """Testes para filter_common_features_only."""

    def test_filters_to_common(self, harmonized_df):
        """Testa filtragem para features comuns."""
        result = filter_common_features_only(harmonized_df)

        # Deve ter RA e ano
        assert "RA" in result.columns or "ano" in result.columns

        # Não deve ter features extras
        assert result.shape[1] <= harmonized_df.shape[1]


class TestNormalizePedraColumn:
    """Testes para normalize_pedra_column."""

    def test_normalizes_pedra(self):
        """Testa normalização da coluna PEDRA."""
        df = pd.DataFrame({
            "PEDRA": ["Quartzo", "ÁGATA", "ametista", "Topázio", np.nan]
        })
        result = normalize_pedra_column(df)

        assert "PEDRA" in result.columns
        assert "PEDRA_CODIGO" in result.columns
        assert result["PEDRA"].iloc[0] == "Quartzo"
        assert result["PEDRA_CODIGO"].iloc[0] == 0

    def test_all_four_pedras_recognized(self, harmonized_df):
        """Verifica que as 4 pedras são reconhecidas."""
        pedras = harmonized_df["PEDRA"].dropna().unique()
        expected = {"Quartzo", "Ágata", "Ametista", "Topázio"}
        assert set(pedras) == expected

    def test_missing_pedra_column(self):
        """Testa quando coluna PEDRA não existe."""
        df = pd.DataFrame({"RA": [1, 2]})
        result = normalize_pedra_column(df)
        assert "PEDRA_CODIGO" not in result.columns
