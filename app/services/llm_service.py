"""
Serviço de integração com LLM via OpenRouter.

Este módulo fornece funções para gerar relatórios personalizados
usando o Claude 3.5 Sonnet ou outros modelos via OpenRouter.
Usa httpx diretamente para evitar incompatibilidade openai SDK + pydantic.
"""

import httpx
from typing import Dict, Optional
from pathlib import Path
from loguru import logger

_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


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
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.completions_url = f"{self.base_url}/chat/completions"
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

DADOS DO ALUNO:
- Nome: {nome}
- Idade: {idade} anos
- Pedra Atual: {pedra} (INDE: {inde:.1f})

INDICADORES EDUCACIONAIS:
- IEG (Engajamento): {ieg:.1f}
- IDA (Desempenho Acadêmico): {ida:.1f}
- IPS (Psicossocial): {ips:.1f}
- IAA (Autoavaliação): {iaa:.1f}
- IAN (Adequação ao Nível): {ian:.1f}
- IPV (Ponto de Virada): {ipv:.1f}
- IPP (Psicopedagógico): {ipp:.1f}

RESULTADOS DOS MODELOS PREDITIVOS (Machine Learning):
- Predição de Risco de Queda: {risco_classe} ({risco_percentual}%)
- Cluster (Perfil Comportamental): {cluster_nome}

HISTÓRICO:
- Anos no programa: {anos_no_programa}
- Tendência do INDE: {tendencia_inde}

FEEDBACKS ANTERIORES:
{feedback_texto}

Com base nos dados acima, incluindo os resultados dos modelos de Machine Learning (predição de risco e clusterização), gere um relatório pedagógico para o professor com o seguinte formato:

## Resumo do Perfil
[2-3 frases acolhedoras sobre o aluno, mencionando o nível de risco predito e o perfil comportamental identificado pelo modelo de clusterização]

## Análise dos Indicadores
[Análise breve dos indicadores educacionais, destacando pontos acima e abaixo da média (5.0), e como se relacionam com o risco predito]

## Pontos Fortes
1. [Ponto forte 1 - baseado nos indicadores]
2. [Ponto forte 2]
3. [Ponto forte 3]

## Pontos de Atenção
1. [Ponto que requer atenção 1 - considerar o risco predito]
2. [Ponto que requer atenção 2]
3. [Ponto que requer atenção 3]

## Recomendações para o Professor
1. [Recomendação concreta e acionável 1 - alinhada ao cluster e risco]
2. [Recomendação concreta e acionável 2]
3. [Recomendação concreta e acionável 3]

Use linguagem clara, empática e construtiva. Evite jargões técnicos.
As recomendações devem ser práticas e considerar o nível de risco e perfil do aluno.
"""

    async def generate_student_report(
        self,
        aluno_data: Dict,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2500
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
            for key in ["idade", "anos_no_programa", "tendencia_inde", "cluster_nome", "risco_percentual", "risco_classe", "feedback_texto", "ieg", "ida", "ips", "iaa", "ian", "ipv", "ipp"]:
                if key not in aluno_data:
                    aluno_data[key] = "N/A"
            prompt = prompt_template.format(**aluno_data)

        # Modelo padrão se não fornecido
        if model is None:
            from app.core.config import get_settings
            settings = get_settings()
            model = settings.openrouter_model

        # Chama API via httpx
        logger.info(f"Gerando relatório com modelo: {model}")

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    self.completions_url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model,
                        "messages": [
                            {
                                "role": "system",
                                "content": "Você é um assistente pedagógico especializado da ONG Passos Mágicos. Gere relatórios construtivos e acolhedores para professores."
                            },
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ],
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                    },
                )
                response.raise_for_status()
                data = response.json()

            relatorio = data["choices"][0]["message"]["content"]
            tokens_usados = data.get("usage", {}).get("total_tokens")

            logger.info(f"Relatório gerado: {len(relatorio)} caracteres, {tokens_usados} tokens")

            return relatorio

        except httpx.HTTPStatusError as e:
            logger.error(f"Erro na API OpenRouter: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Erro na chamada LLM: {e}")
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

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                self.completions_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                },
            )
            response.raise_for_status()
            data = response.json()

        return data["choices"][0]["message"]["content"]


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
