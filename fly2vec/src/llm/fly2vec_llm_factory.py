# Adapted from AERION project (reasoning_agent_node.py LLM_BACKEND pattern)
"""Fly2Vec LLM 백엔드 팩토리."""

import os

from langchain_core.language_models.chat_models import BaseChatModel


def build_fly2vec_llm(backend: str | None = None) -> BaseChatModel:
    """LLM_BACKEND 환경변수 기반 멀티백엔드 LLM 생성.

    Args:
        backend: 'openai' | 'claude' | 'gemini' (None이면 환경변수 사용)

    Returns:
        BaseChatModel 인스턴스
    """
    llm_backend = backend or os.getenv('LLM_BACKEND', 'openai')

    if llm_backend == 'groq':
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=os.getenv('GROQ_MODEL', 'llama-3.3-70b-versatile'),
            temperature=0.1,
            api_key=os.getenv('GROQ_API_KEY', ''),
            base_url='https://api.groq.com/openai/v1',
        )

    elif llm_backend == 'openai':
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=os.getenv('OPENAI_MODEL', 'gpt-4o-mini'),
            temperature=0.1,
        )

    elif llm_backend == 'claude':
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=os.getenv('CLAUDE_MODEL', 'claude-haiku-4-5-20251001'),
            temperature=0.1,
        )

    elif llm_backend == 'gemini':
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=os.getenv('GEMINI_MODEL', 'gemini-2.5-flash'),
            temperature=0.1,
        )

    else:
        raise ValueError(
            f"Unsupported LLM_BACKEND: '{llm_backend}'. "
            "Supported: groq, openai, claude, gemini"
        )
