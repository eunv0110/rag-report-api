#!/usr/bin/env python3
"""FastAPI 서버 실행 스크립트"""
import os
import uvicorn
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

if __name__ == "__main__":
    # 환경 변수에서 설정 읽기
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    reload = os.getenv("API_RELOAD", "false").lower() == "true"

    print("=" * 80)
    print("RAG Report Generator API Server")
    print("=" * 80)
    print(f"Host: {host}")
    print(f"Port: {port}")
    print(f"Reload: {reload}")
    print(f"API Docs: http://{host if host != '0.0.0.0' else 'localhost'}:{port}/docs")
    print(f"ReDoc: http://{host if host != '0.0.0.0' else 'localhost'}:{port}/redoc")
    print("=" * 80)
    print()

    uvicorn.run(
        "app.api.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )
