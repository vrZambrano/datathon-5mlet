"""
Testes para src/data/feature_engineering.py
"""

import pytest
import pandas as pd
import numpy as np

from src.data.feature_engineering import (
    calculate_deltas,
    calculate_inde_trend,
    calculate_years_in_program,
    calculate_pedra_changes,
    identify_new_students,
    create_historical_features,
    create_all_temporal_features,
    create_target_variable,
    prepare_features_for_modeling,
)


class TestCalculateDeltas:
    """Testes para calculate_deltas."""

    def test_basic_delta(self):
        """Testa cálculo de delta simples."""
        df = pd.DataFrame({
            "RA": ["A", "A", "B", "B"],
            "ano": [2022, 2023, 2022, 2023],
            "INDE": [5.0, 7.0, 6.0, 4.0],
        })
        result = calculate_deltas(df, ["INDE"])

        assert "delta_INDE" in result.columns

        # Primeiro ano de cada aluno é NaN
        aluno_a = result[result["RA"] == "A"].sort_values("ano")
        assert pd.isna(aluno_a["delta_INDE"].iloc[0])
        assert aluno_a["delta_INDE"].iloc[1] == pytest.approx(2.0)

        aluno_b = result[result["RA"] == "B"].sort_values("ano")
        assert aluno_b["delta_INDE"].iloc[1] == pytest.approx(-2.0)

    def test_missing_feature_col(self):
        """Testa com feature que não existe no DataFrame."""
        df = pd.DataFrame({
            "RA": ["A"], "ano": [2022], "INDE": [5.0]
        })
        result = calculate_deltas(df, ["INDE", "COLUNA_FAKE"])
        # Não deve ter delta_COLUNA_FAKE
        assert "delta_COLUNA_FAKE" not in result.columns

    def test_single_year(self):
        """Testa com apenas um ano (delta deve ser NaN)."""
        df = pd.DataFrame({
            "RA": ["A"], "ano": [2022], "INDE": [5.0]
        })
        result = calculate_deltas(df, ["INDE"])
        assert pd.isna(result["delta_INDE"].iloc[0])

    def test_multiple_features(self):
        """Testa com múltiplas features."""
        df = pd.DataFrame({
            "RA": ["A", "A"], "ano": [2022, 2023],
            "INDE": [5.0, 8.0], "IEG": [3.0, 6.0],
        })
        result = calculate_deltas(df, ["INDE", "IEG"])
        assert "delta_INDE" in result.columns
        assert "delta_IEG" in result.columns


class TestCalculateIndeTrend:
    """Testes para calculate_inde_trend."""

    def test_positive_trend(self):
        """Testa tendência positiva."""
        df = pd.DataFrame({
            "RA": ["A", "A", "A"],
            "ano": [2022, 2023, 2024],
            "INDE": [4.0, 6.0, 8.0],
        })
        result = calculate_inde_trend(df)
        assert "tendencia_INDE" in result.columns
        assert result["tendencia_INDE"].iloc[0] > 0

    def test_negative_trend(self):
        """Testa tendência negativa."""
        df = pd.DataFrame({
            "RA": ["A", "A", "A"],
            "ano": [2022, 2023, 2024],
            "INDE": [8.0, 6.0, 4.0],
        })
        result = calculate_inde_trend(df)
        assert result["tendencia_INDE"].iloc[0] < 0

    def test_single_year_zero_trend(self):
        """Testa tendência com um só ano = 0."""
        df = pd.DataFrame({
            "RA": ["A"], "ano": [2022], "INDE": [5.0]
        })
        result = calculate_inde_trend(df)
        assert result["tendencia_INDE"].iloc[0] == 0.0


class TestCalculateYearsInProgram:
    """Testes para calculate_years_in_program."""

    def test_cumulative_count(self):
        """Testa contagem cumulativa."""
        df = pd.DataFrame({
            "RA": ["A", "A", "A", "B", "B"],
            "ano": [2022, 2023, 2024, 2023, 2024],
        })
        result = calculate_years_in_program(df)
        assert "anos_no_programa" in result.columns

        aluno_a = result[result["RA"] == "A"].sort_values("ano")
        assert list(aluno_a["anos_no_programa"]) == [1, 2, 3]

    def test_single_year_student(self):
        """Testa aluno com um ano."""
        df = pd.DataFrame({"RA": ["X"], "ano": [2024]})
        result = calculate_years_in_program(df)
        assert result["anos_no_programa"].iloc[0] == 1


class TestCalculatePedraChanges:
    """Testes para calculate_pedra_changes."""

    def test_with_change(self):
        """Testa com mudança de pedra."""
        df = pd.DataFrame({
            "RA": ["A", "A", "A"],
            "ano": [2022, 2023, 2024],
            "PEDRA_CODIGO": [0, 1, 2],  # Quartzo -> Ágata -> Ametista
        })
        result = calculate_pedra_changes(df)
        assert "pedras_mudadas" in result.columns
        assert "pedras_mudadas_total" in result.columns

        sorted_result = result.sort_values("ano")
        assert sorted_result["pedras_mudadas_total"].iloc[2] == 2.0

    def test_no_change(self):
        """Testa sem mudança de pedra."""
        df = pd.DataFrame({
            "RA": ["A", "A"],
            "ano": [2022, 2023],
            "PEDRA_CODIGO": [1, 1],
        })
        result = calculate_pedra_changes(df)
        assert result["pedras_mudadas_total"].iloc[-1] == 0.0

    def test_missing_column(self):
        """Testa quando coluna não existe."""
        df = pd.DataFrame({"RA": ["A"], "ano": [2022]})
        result = calculate_pedra_changes(df)
        assert result["pedras_mudadas"].iloc[0] == 0

    def test_nan_pedra_does_not_inflate(self):
        """Testa que pedra NaN não infla contagem de mudanças."""
        df = pd.DataFrame({
            "RA": ["A", "A", "A"],
            "ano": [2022, 2023, 2024],
            "PEDRA_CODIGO": [2.0, np.nan, 1.0],  # Ametista -> NaN -> Ágata
        })
        result = calculate_pedra_changes(df)
        sorted_r = result.sort_values("ano")
        # Ano 2023 (NaN) deve contribuir 0, não diff com 2
        assert sorted_r["pedras_mudadas"].iloc[1] == 0
        # Total não deve ser inflado por NaN
        assert sorted_r["pedras_mudadas_total"].iloc[2] <= 1.0


class TestIdentifyNewStudents:
    """Testes para identify_new_students."""

    def test_identifies_ingressantes(self):
        """Testa identificação de ingressantes."""
        df = pd.DataFrame({
            "RA": ["A", "A", "B"],
            "ano": [2022, 2023, 2024],
        })
        result = identify_new_students(df)
        assert "ingressante" in result.columns

        # B tem 1 registro, é ingressante
        assert result[result["RA"] == "B"]["ingressante"].iloc[0] == True
        # A tem 2 registros, não é ingressante
        assert result[result["RA"] == "A"]["ingressante"].iloc[0] == False


class TestCreateAllTemporalFeatures:
    """Testes para create_all_temporal_features."""

    def test_creates_all_features(self, harmonized_df):
        """Testa criação de todas features temporais."""
        result = create_all_temporal_features(harmonized_df)

        expected_cols = [
            "tendencia_INDE",
            "anos_no_programa",
            "pedras_mudadas",
            "pedras_mudadas_total",
            "ingressante",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Coluna '{col}' ausente"

    def test_preserves_original_data(self, harmonized_df):
        """Testa que dados originais são preservados."""
        result = create_all_temporal_features(harmonized_df)
        assert len(result) == len(harmonized_df)
        assert "INDE" in result.columns
        assert "RA" in result.columns


class TestCreateTargetVariable:
    """Testes para create_target_variable."""

    def test_creates_target(self):
        """Testa criação do target."""
        df = pd.DataFrame({
            "RA": ["A", "A", "A"],
            "ano": [2022, 2023, 2024],
            "PEDRA_CODIGO": [2, 1, 0],  # Caindo
        })
        result = create_target_variable(df)
        assert "target_queda_prox_ano" in result.columns

        sorted_r = result.sort_values("ano")
        # 2022 → 2023: 2→1 = queda → target=1
        assert sorted_r["target_queda_prox_ano"].iloc[0] == 1.0
        # Último ano → NaN
        assert pd.isna(sorted_r["target_queda_prox_ano"].iloc[2])

    def test_no_queda(self):
        """Testa sem queda."""
        df = pd.DataFrame({
            "RA": ["A", "A"],
            "ano": [2022, 2023],
            "PEDRA_CODIGO": [1, 2],  # Subindo
        })
        result = create_target_variable(df)
        sorted_r = result.sort_values("ano")
        assert sorted_r["target_queda_prox_ano"].iloc[0] == 0.0

    def test_missing_pedra_column(self):
        """Testa quando PEDRA_CODIGO não existe."""
        df = pd.DataFrame({"RA": ["A"], "ano": [2022]})
        result = create_target_variable(df)
        assert pd.isna(result["target_queda_prox_ano"].iloc[0])

    def test_nan_pedra_produces_nan_target(self):
        """Testa que pedra NaN no registro atual gera target NaN, não falso positivo."""
        df = pd.DataFrame({
            "RA": ["A", "A", "A"],
            "ano": [2022, 2023, 2024],
            "PEDRA_CODIGO": [2.0, np.nan, 1.0],
        })
        result = create_target_variable(df)
        sorted_r = result.sort_values("ano")
        # 2023 tem pedra NaN — target deve ser NaN, não 1
        assert pd.isna(sorted_r["target_queda_prox_ano"].iloc[1])
        # 2022→2023: próxima pedra é NaN — target deve ser NaN, não 1
        assert pd.isna(sorted_r["target_queda_prox_ano"].iloc[0])


class TestPrepareFeaturesForModeling:
    """Testes para prepare_features_for_modeling."""

    def test_basic_split(self, harmonized_df_full):
        """Testa preparação de X e y."""
        feature_cols = ["INDE", "IEG", "IDA", "IPS"]
        feature_cols = [f for f in feature_cols if f in harmonized_df_full.columns]

        X, y = prepare_features_for_modeling(harmonized_df_full, feature_cols)

        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert len(X) == len(y)
        assert len(X) > 0

    def test_fills_nan_with_zero(self, harmonized_df_full):
        """Testa que NaN é preenchido com 0."""
        feature_cols = ["INDE", "IEG", "delta_INDE"]
        feature_cols = [f for f in feature_cols if f in harmonized_df_full.columns]

        X, y = prepare_features_for_modeling(harmonized_df_full, feature_cols)
        assert X.isna().sum().sum() == 0
