# RAG Report Generator - 배포 가이드

## Docker를 이용한 배포 (권장)

### 사전 준비사항

1. Docker 및 Docker Compose 설치
2. `.env` 파일 설정 (`.env.example` 참고)

### 빠른 시작

#### 1. 환경 변수 설정

```bash
# .env.example을 복사하여 .env 파일 생성
cp .env.example .env

# .env 파일을 편집하여 실제 API 키 및 설정값 입력
nano .env
```

필수 설정 항목:
- `AZURE_AI_CREDENTIAL`: Azure AI 인증 정보
- `AZURE_AI_ENDPOINT`: Azure AI 엔드포인트
- `OPENROUTER_API_KEY`: OpenRouter API 키
- `NOTION_TOKEN`: Notion 통합 토큰
- `DATA_SOURCE_ID`: Notion 데이터베이스 ID
- `LANGFUSE_SECRET_KEY`, `LANGFUSE_PUBLIC_KEY`: Langfuse 관측성 키
- `UPSTAGE_API_KEY`: Upstage API 키

#### 2. Docker 이미지 빌드 및 실행

**기본 실행 (API만)**

```bash
# 이미지 빌드 및 컨테이너 실행
docker-compose up -d

# 로그 확인
docker-compose logs -f rag-api
```

**Nginx와 함께 실행**

```bash
# Nginx 프록시 포함하여 실행
docker-compose --profile with-nginx up -d

# 로그 확인
docker-compose logs -f
```

#### 3. 서비스 확인

```bash
# Health check
curl http://localhost:8000/health

# API 문서 확인
# 브라우저에서: http://localhost:8000/docs
```

### 배포 명령어 모음

```bash
# 컨테이너 시작
docker-compose up -d

# 컨테이너 중지
docker-compose down

# 컨테이너 재시작
docker-compose restart

# 로그 확인 (실시간)
docker-compose logs -f rag-api

# 이미지 재빌드 (코드 변경 시)
docker-compose build --no-cache
docker-compose up -d

# 컨테이너 상태 확인
docker-compose ps

# 컨테이너 내부 접속 (디버깅)
docker-compose exec rag-api bash
```

### 디렉토리 구조

```
rag-report-generator/
├── Dockerfile              # Docker 이미지 정의
├── docker-compose.yml      # Docker Compose 설정
├── .dockerignore          # Docker 빌드 시 제외할 파일
├── nginx.conf             # Nginx 설정 (선택사항)
├── .env                   # 환경 변수 (git ignore됨)
├── .env.example           # 환경 변수 템플릿
├── requirements.txt       # Python 의존성
├── app/                   # 애플리케이션 코드
├── data/                  # 데이터 디렉토리 (볼륨 마운트)
└── logs/                  # 로그 디렉토리 (볼륨 마운트)
```

### 프로덕션 배포 체크리스트

- [ ] `.env` 파일에 모든 필수 환경 변수 설정
- [ ] `nginx.conf`에서 `server_name`을 실제 도메인으로 변경
- [ ] HTTPS 설정 (SSL 인증서)
- [ ] 방화벽 설정 (필요한 포트만 오픈)
- [ ] 로그 로테이션 설정
- [ ] 모니터링 및 알림 설정
- [ ] 백업 전략 수립 (`data/` 디렉토리)
- [ ] 리소스 제한 확인 (CPU, Memory)

### 트러블슈팅

#### 컨테이너가 시작되지 않는 경우

```bash
# 로그 확인
docker-compose logs rag-api

# .env 파일 확인
cat .env

# 포트 충돌 확인
sudo netstat -tlnp | grep 8000
```

#### 데이터베이스 연결 실패

- `.env` 파일의 `NOTION_TOKEN` 및 `DATA_SOURCE_ID` 확인
- Notion Integration 권한 확인

#### 메모리 부족

`docker-compose.yml`의 리소스 제한 조정:

```yaml
deploy:
  resources:
    limits:
      memory: 16G  # 필요에 따라 조정
```

---

## 기존 서버 배포 (systemd)

기존 Linux 서버에 직접 배포하려면 `deploy.sh`를 사용하세요.

### 사전 준비사항

1. Python 3.11+ 설치
2. Nginx 설치 (선택사항)
3. Gunicorn 설치

### 배포 실행

```bash
# 실행 권한 부여
chmod +x deploy.sh

# 배포 스크립트 실행
./deploy.sh
```

### 서비스 관리

```bash
# 서비스 상태 확인
sudo systemctl status rag-api

# 서비스 시작/중지/재시작
sudo systemctl start rag-api
sudo systemctl stop rag-api
sudo systemctl restart rag-api

# 로그 확인
sudo journalctl -u rag-api -f
```

---

## 개발 환경 실행

로컬 개발 시:

```bash
# 가상환경 활성화
source .venv/bin/activate

# 환경 변수 로드
source .env

# 개발 서버 실행
python run_api.py
```

또는:

```bash
# Uvicorn 직접 실행 (auto-reload 포함)
uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000
```

---

## API 엔드포인트

### Health Check
```bash
GET /health
```

### 보고서 생성
```bash
POST /api/v1/generate-report
```

자세한 API 문서는 `/docs` 또는 `/redoc`에서 확인하세요.

---

## 보안 권장사항

1. `.env` 파일은 절대 Git에 커밋하지 마세요
2. 프로덕션에서는 HTTPS를 필수로 사용하세요
3. API 키는 정기적으로 로테이션하세요
4. 방화벽 규칙을 적절히 설정하세요
5. 컨테이너는 non-root 사용자로 실행됩니다 (보안 강화)

---

## 모니터링

Langfuse를 통한 관측성:
- Langfuse 대시보드: https://cloud.langfuse.com
- `.env`의 `LANGFUSE_*` 설정 필요

---

## 지원

문제가 발생하면:
1. 로그 확인: `docker-compose logs -f`
2. Health check: `curl http://localhost:8000/health`
3. API 문서: `http://localhost:8000/docs`
