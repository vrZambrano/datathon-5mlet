"""
Serviço de integração com LLM via OpenRouter.

Este módulo fornece funções para gerar relatórios personalizados
usando o Claude 3.5 Sonnet ou outros modelos via OpenRouter.
"""

import openai
from typing import Dict, Optional
from pathlib import Path
from loguru import logger


class LLMService:
    """
    Serviço para interação com LLMs via OpenRouter.

    OpenRouter permite acesso a múltiplos modelos (Claude, GPT-4, Llama, etc.)
    através de uma API única.
    """

    def __init__(self, api_key: str, base_url: str = "https://openrouter.ai/api/v1"):
        """
        Inicializa o serviço LLM.

        Args:
            api_key: Chave de API do OpenRouter
            base_url: URL base da API (padrão: OpenRouter)
        """
        self.client = openai.OpenAI(
            base_url=base_url,
            api_key=api_key
        )
        logger.info("Serviço LLM inicializado")

    def _load_prompt_template(self, template_name: str = "relatorio_aluno.txt") -> str:
        """
        Carrega um template de prompt do arquivo.

        Args:
            template_name: Nome do arquivo de template

        Returns:
            String com o template
        """
        template_path = Path(f"prompts/{template_name}")

        if template_path.exists():
            with open(template_path, "r", encoding="utf-8") as f:
                template = f.read()
            logger.debug(f"Template carregado: {template_name}")
            return template
        else:
            # Template padrão embutido
            logger.warning(f"Template não encontrado: {template_name}, usando padrão")
            return self._get_default_template()

    def _get_default_template(self) -> str:
        """Retorna o template padrão de relatório."""
        return """Você é um assistente pedagógico especializado em educação para jovens em vulnerabilidade social, no contexto da ONG Passos Mágicos.

ALUNO: {nome}
IDADE: {idade} anos
PEDRA ATUAL: {pedra} (INDE: {inde:.1f})

HISTÓRICO:
- Anos no programa: {anos_no_programa}
- Tendência do INDE: {tendencia_inde}
- Perfil (Cluster): {cluster_nome}
- Risco de queda no próximo ano: {risco_percentual}%

FEEDBACKS ANTERIORES:
{feedback_texto}

Por favor, gere um relatório para o professor com o seguinte formato:

## Resumo do Perfil
[2-3 frases acolhedoras sobre o aluno]

## Pontos Fortes
1. [Ponto forte 1]
2. [Ponto forte 2]
3. [Ponto forte 3]

## Pontos de Atenção
1. [Ponto que requer atenção 1]
2. [Ponto que requer atenção 2]
3. [Ponto que requer atenção 3]

## Recomendações para o Professor
1. [Recomendação concreta e acionável 1]
2. [Recomendação concreta e acionável 2]
3. [Recomendação concreta e acionável 3]

Use linguagem clara, empática e construtiva. Evite jargões técnicos.
"""

    async def generate_student_report(
        self,
        aluno_data: Dict,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> str:
        """
        Gera relatório personalizado do aluno.

        Args:
            aluno_data: Dicionário com dados do aluno
            model: Modelo a ser usado (opcional, usa padrão se não fornecido)
            temperature: Temperatura para geração (0-1)
            max_tokens: Máximo de tokens

        Returns:
            String com o relatório gerado
        """
        # Carrega template
        prompt_template = self._load_prompt_template()

        # Formata prompt com dados do aluno
        try:
            prompt = prompt_template.format(**aluno_data)
        except KeyError as e:
            logger.warning(f"Campo faltando nos dados: {e}")
            # Preenche campos vazios
            for key in ["idade", "anos_no_programa", "tendencia_inde", "cluster_nome", "risco_percentual", "feedback_texto"]:
                if key not in aluno_data:
                    aluno_data[key] = "N/A"
            prompt = prompt_template.format(**aluno_data)

        # Modelo padrão se não fornecido
        if model is None:
            from app.core.config import get_settings
            settings = get_settings()
            model = settings.openrouter_model

        # Chama API
        logger.info(f"Gerando relatório com modelo: {model}")

        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "Você é um assistente pedagógico especializado da ONG Passos Mágicos. Gere relatórios construtivos e acolhedores para professores."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )

            relatorio = response.choices[0].message.content
            tokens_usados = response.usage.total_tokens if hasattr(response, "usage") else None

            logger.info(f"Relatório gerado: {len(relatorio)} caracteres, {tokens_usados} tokens")

            return relatorio

        except openai.APIError as e:
            logger.error(f"Erro na API OpenAI/OpenRouter: {e}")
            raise

    async def simple_completion(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.7
    ) -> str:
        """
        Faz uma completion simples.

        Args:
            prompt: Prompt para enviar
            model: Modelo a ser usado
            temperature: Temperatura

        Returns:
            Resposta do modelo
        """
        if model is None:
            from app.core.config import get_settings
            settings = get_settings()
            model = settings.openrouter_model

        response = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature
        )

        return response.choices[0].message.content


# Função de conveniência
async def generate_report_sync(aluno_data: Dict, api_key: str) -> str:
    """
    Função síncrona para geração de relatório (wrapper).

    Args:
        aluno_data: Dados do aluno
        api_key: Chave da API OpenRouter

    Returns:
        Relatório gerado
    """
    service = LLMService(api_key=api_key)
    return await service.generate_student_report(aluno_data)
