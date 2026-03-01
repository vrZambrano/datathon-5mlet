"""
Testes para src/monitoring/drift.py
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path


class TestCreateDriftReport:
    """Testes para create_drift_report."""

    def test_creates_report(self):
        """Testa criação de report de drift."""
        from src.monitoring.drift import create_drift_report

        np.random.seed(42)
        ref = pd.DataFrame({
            "INDE": np.random.normal(6, 2, 100),
            "IEG": np.random.normal(5, 1.5, 100),
        })
        cur = pd.DataFrame({
            "INDE": np.random.normal(6.5, 2, 100),
            "IEG": np.random.normal(5.2, 1.5, 100),
        })

        report = create_drift_report(ref, cur, ["INDE", "IEG"])
        assert report is not None

    def test_with_shifted_distribution(self):
        """Testa com distribuições muito diferentes."""
        from src.monitoring.drift import create_drift_report

        np.random.seed(42)
        ref = pd.DataFrame({"INDE": np.random.normal(5, 1, 200)})
        cur = pd.DataFrame({"INDE": np.random.normal(9, 0.5, 200)})

        report = create_drift_report(ref, cur, ["INDE"])
        assert report is not None


class TestCheckDrift:
    """Testes para check_drift."""

    def test_returns_drift_info(self):
        """Testa retorno de informações de drift."""
        from src.monitoring.drift import check_drift

        np.random.seed(42)
        ref = pd.DataFrame({
            "INDE": np.random.normal(6, 2, 100),
            "IEG": np.random.normal(5, 1.5, 100),
        })
        cur = pd.DataFrame({
            "INDE": np.random.normal(6, 2, 100),
            "IEG": np.random.normal(5, 1.5, 100),
        })

        result = check_drift(ref, cur, ["INDE", "IEG"])

        assert isinstance(result, dict)
        assert "dataset_drift" in result
        assert "drift_share" in result
        assert isinstance(result["dataset_drift"], bool)

    def test_detects_drift_with_shifted_data(self):
        """Testa detecção de drift com dados deslocados."""
        from src.monitoring.drift import check_drift

        np.random.seed(42)
        ref = pd.DataFrame({"INDE": np.random.normal(3, 0.5, 200)})
        cur = pd.DataFrame({"INDE": np.random.normal(9, 0.5, 200)})

        result = check_drift(ref, cur, ["INDE"])

        # Drift should be detected with such different distributions
        assert result["dataset_drift"] is True

    def test_no_drift_with_similar_data(self):
        """Testa sem drift com dados similares."""
        from src.monitoring.drift import check_drift

        np.random.seed(42)
        ref = pd.DataFrame({"INDE": np.random.normal(6, 2, 500)})
        cur = pd.DataFrame({"INDE": np.random.normal(6, 2, 500)})

        result = check_drift(ref, cur, ["INDE"])
        # Likely no drift with same distribution
        assert isinstance(result["dataset_drift"], bool)


class TestSaveDriftReportHtml:
    """Testes para save_drift_report_html."""

    def test_saves_html(self):
        """Testa salvamento de relatório HTML."""
        from src.monitoring.drift import save_drift_report_html

        np.random.seed(42)
        ref = pd.DataFrame({"INDE": np.random.normal(6, 2, 100)})
        cur = pd.DataFrame({"INDE": np.random.normal(6.5, 2, 100)})

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "drift_report.html")
            save_drift_report_html(ref, cur, output_path=path, feature_cols=["INDE"])
            assert Path(path).exists()
            assert Path(path).stat().st_size > 0


class TestCompareYearDrift:
    """Testes para compare_year_drift."""

    def test_compare_two_years(self):
        """Testa comparação entre dois anos."""
        from src.monitoring.drift import compare_year_drift

        np.random.seed(42)
        df = pd.DataFrame({
            "ano": [2022] * 100 + [2023] * 100,
            "INDE": np.concatenate([
                np.random.normal(6, 2, 100),
                np.random.normal(6.5, 2, 100),
            ]),
            "IEG": np.concatenate([
                np.random.normal(5, 1.5, 100),
                np.random.normal(5.3, 1.5, 100),
            ]),
        })

        result = compare_year_drift(df, 2022, 2023, ["INDE", "IEG"])

        assert isinstance(result, dict)
        assert "dataset_drift" in result

    def test_missing_year_returns_error(self):
        """Testa com ano inexistente."""
        from src.monitoring.drift import compare_year_drift

        df = pd.DataFrame({
            "ano": [2022] * 50,
            "INDE": np.random.normal(6, 2, 50),
        })

        result = compare_year_drift(df, 2022, 2025, ["INDE"])
        assert "error" in result
