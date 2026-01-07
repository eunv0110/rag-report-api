#!/usr/bin/env python3
"""LLM Factory - Azure AI와 OpenRouter LLM 생성을 위한 팩토리 함수"""

import os
from typing import Optional
from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI


# Azure AI 모델 ID/이름을 OpenRouter 모델로 매핑
AZURE_TO_OPENROUTER = {
    "azure_ai:gpt-4.1": "openai/gpt-4.1",
    "gpt41": "openai/gpt-4.1",
    "azure_ai:DeepSeek-V3.1": "deepseek/deepseek-chat-v3.1",
    "deepseek_v31": "deepseek/deepseek-chat-v3.1",
}


def get_openrouter_model(model_id: str, model_name: Optional[str] = None) -> str:
    """
    Azure AI 모델 ID 또는 모델 이름을 OpenRouter 모델명으로 변환

    Args:
        model_id: Azure AI 모델 ID (예: "azure_ai:gpt-4.1")
        model_name: 모델 이름 (예: "gpt41", "deepseek_v31")

    Returns:
        OpenRouter 모델명 (예: "openai/gpt-4.1", "deepseek/deepseek-chat-v3.1")
    """
    # 1. model_name으로 먼저 확인
    if model_name and model_name in AZURE_TO_OPENROUTER:
        return AZURE_TO_OPENROUTER[model_name]

    # 2. model_id로 확인
    if model_id in AZURE_TO_OPENROUTER:
        return AZURE_TO_OPENROUTER[model_id]

    # 3. 매핑 없으면 에러
    raise ValueError(f"알 수 없는 모델: model_id={model_id}, model_name={model_name}")


def get_llm(
    model_id: str,
    temperature: float = 0,
    max_tokens: int = 1000,
    use_openrouter: bool = False,
    model_name: Optional[str] = None
):
    """
    LLM 인스턴스를 생성합니다.

    Args:
        model_id: Azure AI 모델 ID (예: "azure_ai:gpt-4.1")
        temperature: 생성 온도
        max_tokens: 최대 토큰 수
        use_openrouter: OpenRouter를 사용할지 여부
        model_name: 모델 이름 (예: "gpt41", "deepseek_v31")

    Returns:
        LangChain ChatModel 인스턴스

    Examples:
        # Azure AI 사용
        >>> model = get_llm("azure_ai:gpt-4.1", temperature=0.1)

        # OpenRouter 사용 (자동 매핑)
        >>> model = get_llm(
        ...     model_id="azure_ai:gpt-4.1",
        ...     model_name="gpt41",
        ...     use_openrouter=True
        ... )
        # -> "openai/gpt-4.1" 모델 사용
    """
    # OpenRouter 사용
    if use_openrouter:
        from app.config.settings import OPENROUTER_API_KEY, OPENROUTER_BASE_URL

        if not OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY가 설정되지 않았습니다.")

        # OpenRouter 모델 자동 매핑
        openrouter_model = get_openrouter_model(model_id, model_name)

        print(f"🔄 Using OpenRouter: {openrouter_model}")

        return ChatOpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=OPENROUTER_API_KEY,
            model=openrouter_model,
            temperature=temperature,
            max_tokens=max_tokens
        )

    # Azure AI 사용 (기본값)
    else:
        from app.config.settings import AZURE_AI_CREDENTIAL, AZURE_AI_ENDPOINT

        if not AZURE_AI_CREDENTIAL or not AZURE_AI_ENDPOINT:
            raise ValueError("Azure AI 설정이 올바르지 않습니다. .env 파일을 확인하세요.")

        os.environ['AZURE_AI_CREDENTIAL'] = AZURE_AI_CREDENTIAL
        os.environ['AZURE_AI_ENDPOINT'] = AZURE_AI_ENDPOINT

        print(f"🔄 Using Azure AI: {model_id}")

        return init_chat_model(
            model_id,
            temperature=temperature,
            max_completion_tokens=max_tokens
        )
