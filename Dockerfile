# Python 3.12 기반 이미지
FROM python:3.12-slim

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 패키지 업데이트 및 필수 패키지 설치
RUN apt-get update && apt-get install -y \
    pandoc \
    libreoffice \
    fonts-nanum \
    fonts-nanum-coding \
    fonts-nanum-extra \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Python 의존성 파일 복사
COPY requirements.txt .

# Python 패키지 설치
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY app/ ./app/
COPY main.py .
COPY pyproject.toml .

# 데이터 디렉토리 생성
RUN mkdir -p /app/data/reports /app/data/qdrant_data /app/data/notion_images

# 환경 변수 설정
ENV PYTHONUNBUFFERED=1
ENV PROJECT_ROOT=/app
ENV DATA_DIR=/app/data
ENV QDRANT_DATA_DIR=/app/data/qdrant_data
ENV PROMPTS_BASE_DIR=/app/app/prompts

# 포트 노출
EXPOSE 8000

# 헬스체크
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# 애플리케이션 실행
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
