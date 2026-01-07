"""Langfuse 트레이싱 유틸리티

Langfuse는 LLM 애플리케이션의 트레이싱, 모니터링, 평가를 위한 도구입니다.
"""

from typing import Optional, Dict, Any
from contextlib import contextmanager
from dotenv import load_dotenv
from langfuse import Langfuse
from app.config.settings import LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST

load_dotenv()

_langfuse_client: Optional[Langfuse] = None


def get_langfuse_client() -> Optional[Langfuse]:
    """Langfuse 클라이언트 싱글톤 반환

    Returns:
        Langfuse 클라이언트 인스턴스 또는 None (설정되지 않은 경우)
    """
    global _langfuse_client

    if _langfuse_client is None:
        if LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY:
            _langfuse_client = Langfuse(
                public_key=LANGFUSE_PUBLIC_KEY,
                secret_key=LANGFUSE_SECRET_KEY,
                host=LANGFUSE_HOST
            )
            print("✅ Langfuse 연동 완료")
        else:
            print("⚠️  Langfuse 키가 설정되지 않음 (트레이싱 비활성화)")

    return _langfuse_client


@contextmanager
def trace_operation(name: str, metadata: Optional[Dict[str, Any]] = None, user_id: Optional[str] = None):
    """작업을 Langfuse에 트레이싱 (현재 비활성화)

    Args:
        name: 작업 이름
        metadata: 메타데이터
        user_id: 사용자 ID

    Yields:
        None

    Note:
        현재 API 호환성 문제로 트레이싱이 비활성화되어 있습니다.
        추후 활성화 예정입니다.
    """
    # 트레이싱 비활성화 - 추후 활성화 예정
    yield None


def save_feedback(trace_id: str, score: int, comment: Optional[str] = None, feedback_type: str = "user_satisfaction") -> bool:
    """사용자 피드백을 Langfuse에 저장

    Args:
        trace_id: Langfuse Trace ID
        score: 피드백 점수 (0-10)
        comment: 피드백 코멘트 (선택)
        feedback_type: 피드백 타입

    Returns:
        저장 성공 여부

    Raises:
        ValueError: Langfuse 클라이언트가 초기화되지 않은 경우
    """
    client = get_langfuse_client()

    if client is None:
        raise ValueError("Langfuse 클라이언트가 초기화되지 않았습니다. LANGFUSE_PUBLIC_KEY와 LANGFUSE_SECRET_KEY를 설정해주세요.")

    try:
        # 0-10 점수를 0-1로 정규화
        normalized_score = score / 10.0

        client.create_score(
            trace_id=trace_id,
            name=feedback_type,
            value=normalized_score,
            comment=comment
        )

        # 즉시 flush하여 서버에 전송
        client.flush()

        print(f"✅ 피드백 저장 완료 (trace_id: {trace_id}, score: {score}/10 -> {normalized_score})")
        return True

    except Exception as e:
        print(f"❌ 피드백 저장 실패: {e}")
        return False
