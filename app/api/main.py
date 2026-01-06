"""FastAPI 서버 - RAG 보고서 생성 API"""
import os
import sys
import time
import uuid
import getpass
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 환경 변수 로드
load_dotenv()

from app.api.schemas import (
    ReportRequest,
    ReportResponse,
    HealthResponse
)
from app.scripts.report_generator import ReportGenerator
from app.utils.dates import parse_date_range

# FastAPI 앱 생성
app = FastAPI(
    title="RAG Report Generator API",
    description="RAG 기반 보고서 자동 생성 API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 특정 도메인으로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=HealthResponse)
async def root():
    """루트 엔드포인트 - API 상태 확인"""
    return HealthResponse(
        status="ok",
        version="1.0.0",
        timestamp=datetime.now().isoformat()
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """헬스체크 엔드포인트"""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        timestamp=datetime.now().isoformat()
    )


def calculate_generation_cost(report_data: Dict[str, Any]) -> float:
    """보고서 생성 비용 계산 (예시)"""
    # 실제로는 LLM API 호출 비용을 계산해야 함
    # 여기서는 간단하게 문서 수 기반으로 예시 비용 계산
    total_docs = sum(result.get('num_docs', 0) for result in report_data.get('results', []))
    cost_per_doc = 0.001  # 문서당 예상 비용
    return round(total_docs * cost_per_doc, 4)


@app.post("/api/v1/generate-report", response_model=ReportResponse)
async def generate_report(request: ReportRequest):
    """
    보고서 생성 API

    - **report_type**: 보고서 타입 (weekly 또는 executive)
    - **questions**: 질문 리스트 (선택사항, 미지정 시 기본 질문 사용)
    - **date_range**: 날짜 범위 (예: '이번 주', '12월 2주차')
    - **start_date**: 시작 날짜 (YYYY-MM-DD)
    - **end_date**: 종료 날짜 (YYYY-MM-DD)
    - **author**: 보고서 작성자 (선택사항)
    """
    try:
        start_time = time.time()
        trace_id = str(uuid.uuid4())

        # 날짜 필터 파싱
        date_filter = parse_date_range(
            date_input=request.date_range,
            start_date=request.start_date,
            end_date=request.end_date
        )

        # 보고서 생성기 초기화
        generator = ReportGenerator(
            config_path=None,  # 기본 설정 파일 사용
            report_type=request.report_type
        )

        # 질문 설정
        if request.questions:
            questions = request.questions
        else:
            # 설정 파일의 기본 질문 사용
            questions = generator.default_questions

        if not questions:
            raise HTTPException(
                status_code=400,
                detail="질문이 제공되지 않았으며, 기본 질문도 없습니다."
            )

        # 작성자 정보 설정
        author = request.author if request.author else getpass.getuser()

        # 보고서 생성
        report_data = generator.generate_report(questions, date_filter)

        # 작성자 및 작성일자 정보 추가
        report_data["author"] = author
        report_data["created_date"] = datetime.now().strftime("%Y-%m-%d")
        report_data["created_datetime"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 마크다운 형식으로 변환 (결과 조합)
        report_title = "주간 업무 보고서" if request.report_type == "weekly" else "최종 보고서"
        markdown_parts = [
            f"# {report_title}\n",
            f"**작성일:** {report_data['created_date']}",
            f"**작성자:** {author}\n"
        ]

        for result in report_data.get('results', []):
            if result.get('success', False):
                section_title = result.get('title') or "항목"
                answer = result.get('answer', '')
                markdown_parts.append(f"## {section_title}\n")
                markdown_parts.append(f"{answer}\n")

        report_markdown = "\n".join(markdown_parts)

        # 생성 시간 및 비용 계산
        generation_time = round(time.time() - start_time, 2)
        cost = calculate_generation_cost(report_data)

        # 메타데이터 구성
        metadata = {
            "report_type": request.report_type,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "author": author,
            "generation_time": generation_time,
            "model": report_data['llm']['model_id'],
            "cost": cost,
            "timestamp": datetime.now().isoformat()
        }

        # 응답 구성
        response_data = {
            "code": 1,
            "message": "성공하였습니다.",
            "result": {
                "data": {
                    "trace_id": trace_id,
                    "report": report_markdown,
                    "metadata": metadata
                }
            },
            "success": True
        }

        return response_data

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        import traceback
        error_detail = f"{str(e)}\n\n{traceback.format_exc()}"
        raise HTTPException(status_code=500, detail=error_detail)



if __name__ == "__main__":
    import uvicorn

    # 환경 변수에서 설정 읽기
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    reload = os.getenv("API_RELOAD", "false").lower() == "true"

    uvicorn.run(
        "app.api.main:app",
        host=host,
        port=port,
        reload=reload
    )
