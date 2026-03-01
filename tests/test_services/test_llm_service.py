"""
Testes para app/services/llm_service.py
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock


class TestLLMService:
    """Testes para LLMService."""

    def test_initialization(self):
        """Testa inicialização do serviço."""
        from app.services.llm_service import LLMService

        service = LLMService(api_key="test-key-123")
        assert service.client is not None

    def test_get_default_template(self):
        """Testa template padrão."""
        from app.services.llm_service import LLMService

        service = LLMService(api_key="test-key-123")
        template = service._get_default_template()

        assert isinstance(template, str)
        assert "{nome}" in template
        assert "{inde:.1f}" in template or "{inde}" in template
        assert "{pedra}" in template

    def test_load_prompt_template_fallback(self):
        """Testa fallback quando template não existe."""
        from app.services.llm_service import LLMService

        service = LLMService(api_key="test-key-123")
        template = service._load_prompt_template("nonexistent.txt")

        assert isinstance(template, str)
        assert len(template) > 100

    @pytest.mark.asyncio
    async def test_generate_report_with_mock(self):
        """Testa geração de relatório com mock da API."""
        from app.services.llm_service import LLMService

        service = LLMService(api_key="test-key-123")

        # Mock da resposta da API
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "## Resumo\nRelatório teste gerado."
        mock_response.usage.total_tokens = 150

        with patch.object(service.client.chat.completions, 'create', return_value=mock_response):
            aluno_data = {
                "nome": "Aluno Teste",
                "idade": 12,
                "pedra": "Ametista",
                "inde": 7.5,
                "ieg": 6.0,
                "ida": 7.0,
                "ips": 5.5,
                "iaa": 6.5,
                "ian": 5.0,
                "ipv": 7.0,
                "ipp": 6.0,
                "anos_no_programa": 2,
                "tendencia_inde": "crescendo",
                "cluster_nome": "Alto Desempenho",
                "risco_percentual": 15,
                "risco_classe": "BAIXO",
                "feedback_texto": "Bom desempenho geral."
            }

            relatorio = await service.generate_student_report(aluno_data)

        assert isinstance(relatorio, str)
        assert "Relatório" in relatorio or "Resumo" in relatorio

    @pytest.mark.asyncio
    async def test_generate_report_handles_missing_fields(self):
        """Testa geração com campos faltantes."""
        from app.services.llm_service import LLMService

        service = LLMService(api_key="test-key-123")

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Relatório gerado."
        mock_response.usage.total_tokens = 50

        with patch.object(service.client.chat.completions, 'create', return_value=mock_response):
            # Dados mínimos
            aluno_data = {
                "nome": "Teste",
                "pedra": "Quartzo",
                "inde": 4.0,
            }

            relatorio = await service.generate_student_report(aluno_data)

        assert isinstance(relatorio, str)
