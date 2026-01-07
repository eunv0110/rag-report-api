# RAG Report Generator

자동화된 보고서 생성 시스템 - RAG(Retrieval-Augmented Generation) 기반

## 🌟 주요 기능

- 📊 **자동 보고서 생성**: 주간/임원 보고서 자동 생성
- 📄 **다양한 출력 형식**: JSON, Word(DOCX), PDF 지원
- 🔍 **RAG 기반 검색**: Qdrant 벡터 데이터베이스를 활용한 문서 검색
- 🤖 **LLM 통합**: Azure AI, OpenRouter 등 다양한 LLM 지원
- 📝 **Notion 연동**: Notion 데이터베이스에서 자동으로 데이터 수집
- 📈 **모니터링**: Langfuse를 통한 LLM 호출 추적

## 🚀 빠른 시작

### 1. Docker를 사용한 배포 (권장)

```bash
# Docker 설치 (Ubuntu)
./install-docker.sh

# 환경 변수 설정
cp .env.example .env
nano .env  # 필수 값 입력

# 배포 실행
./deploy.sh
```

### 2. 로컬 개발 환경

```bash
# 가상환경 생성 및 활성화
python3 -m venv .venv
source .venv/bin/activate

# 의존성 설치
pip install -r requirements.txt

# 서버 실행
uvicorn app.api.main:app --reload
```

## 📡 API 사용법

### 보고서 생성 API

**엔드포인트**: `POST /generate-report`

**요청 예시 (Word 파일)**:
```json
{
  "report_type": "weekly",
  "question": "25년도 12월 첫째주 보고서 만들어줘",
  "output_format": "docx"
}
```

**요청 예시 (JSON)**:
```json
{
  "report_type": "weekly",
  "question": "이번 주 주요 성과와 진행 업무는?",
  "output_format": "json"
}
```

**curl 예시**:
```bash
# Word 파일 다운로드
curl -X POST http://localhost:8000/generate-report \
  -H "Content-Type: application/json" \
  -d '{"report_type": "weekly", "question": "이번 주 보고서", "output_format": "docx"}' \
  --output report.docx

# JSON 응답
curl -X POST http://localhost:8000/generate-report \
  -H "Content-Type: application/json" \
  -d '{"report_type": "weekly", "question": "이번 주 보고서", "output_format": "json"}'
```

### API 문서

서버 실행 후 다음 URL에서 확인:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 🛠️ 기술 스택

- **Backend**: FastAPI, Python 3.12
- **Vector DB**: Qdrant (Cloud 또는 Self-hosted)
- **LLM**: Azure AI, OpenRouter
- **Document Processing**: python-docx, pypandoc, LibreOffice
- **Observability**: Langfuse
- **Deployment**: Docker, Docker Compose

## 📚 문서

- [배포 가이드](DEPLOYMENT.md) - 상세한 배포 및 운영 가이드
- [API 문서](http://localhost:8000/docs) - Swagger UI (서버 실행 후)

## 🔧 환경 변수

필수 환경 변수:
```bash
# Azure AI
AZURE_AI_CREDENTIAL=your_credential
AZURE_AI_ENDPOINT=your_endpoint

# Notion
NOTION_TOKEN=your_token
DATA_SOURCE_ID=your_database_id

# Qdrant (Cloud 사용 시)
QDRANT_USE_SERVER=true
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_api_key
```

자세한 내용은 [.env.example](.env.example) 참고

## 📊 프로젝트 구조

```
rag-report-generator/
├── app/
│   ├── api/              # FastAPI 라우터 및 스키마
│   ├── chains/           # LangChain 체인
│   ├── models/           # 임베딩 및 LLM 모델
│   ├── prompts/          # 프롬프트 템플릿
│   ├── scripts/          # 보고서 생성 스크립트
│   └── utils/            # 유틸리티 함수
├── data/                 # 데이터 디렉토리
│   ├── reports/          # 생성된 보고서
│   ├── qdrant_data/      # Qdrant 로컬 데이터
│   └── notion_images/    # Notion 이미지
├── Dockerfile
├── docker-compose.yml
├── deploy.sh             # 배포 스크립트
├── install-docker.sh     # Docker 설치 스크립트
└── requirements.txt
```

## 🤝 기여

이슈 및 PR을 환영합니다!

## 📄 라이선스

MIT License

## 📧 문의

문제가 발생하면 GitHub Issues에 등록해주세요.
