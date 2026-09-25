"""A DeepEval (https://deepeval.com) judge model backed by litellm, so any
`ModelConfig` eval-bench already knows how to authenticate (OpenRouter,
OpenAI, Anthropic, DeepSeek's own API, ...) can also serve as the LLM-judge
for DeepEval metrics like GEval, instead of being stuck with DeepEval's
OpenAI default.

Implements the six-rule `DeepEvalBaseLLM` contract from DeepEval's own
"Using Custom LLMs for Evaluation" guide, verified against the installed
package (deepeval.models.DeepEvalBaseLLM's actual method signatures), not
just the docs prose.
"""
from __future__ import annotations

import os
from typing import Optional, Type

from deepeval.models import DeepEvalBaseLLM
from pydantic import BaseModel

from evalbench.config import ModelConfig

# litellm expects its own standard env var name per provider prefix,
# regardless of what the user named api_key_env in their config - same
# mapping as evalbench/harnesses/mini_swe_agent.py, which hits this exact
# litellm requirement.
_LITELLM_ENV_VAR = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "deepseek-official": "DEEPSEEK_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "openai-compatible": "OPENAI_API_KEY",
}


class LitellmJudgeModel(DeepEvalBaseLLM):
    def __init__(self, model: ModelConfig):
        self._model = model
        os.environ[_LITELLM_ENV_VAR[model.provider]] = model.resolve_api_key()

    def load_model(self) -> "LitellmJudgeModel":
        return self

    def get_model_name(self) -> str:
        return self._model.as_litellm_model_string()

    def _request_kwargs(self, schema: Optional[Type[BaseModel]]) -> dict:
        kwargs: dict = {"model": self._model.as_litellm_model_string()}
        if base_url := self._model.resolve_base_url():
            kwargs["api_base"] = base_url
        if schema is not None:
            kwargs["response_format"] = schema
        return kwargs

    def generate(self, prompt: str, schema: Optional[Type[BaseModel]] = None):
        import litellm

        response = litellm.completion(
            messages=[{"role": "user", "content": prompt}],
            **self._request_kwargs(schema),
        )
        content = response.choices[0].message.content
        return schema.model_validate_json(content) if schema is not None else content

    async def a_generate(self, prompt: str, schema: Optional[Type[BaseModel]] = None):
        import litellm

        response = await litellm.acompletion(
            messages=[{"role": "user", "content": prompt}],
            **self._request_kwargs(schema),
        )
        content = response.choices[0].message.content
        return schema.model_validate_json(content) if schema is not None else content
