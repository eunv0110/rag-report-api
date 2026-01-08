# RAG Report Generator - 배포 가이드

## 📋 사전 요구사항

- Docker 및 Docker Compose
- 최소 4GB RAM
- 최소 10GB 디스크 공간

## 🚀 빠른 시작

### 1. Docker 설치 (Ubuntu)

```bash
# Docker 설치
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# 현재 사용자를 docker 그룹에 추가
sudo usermod -aG docker $USER
newgrp docker

# Docker Compose 설치 확인
docker compose version
```

### 2. 환경 변수 설정

```bash
# .env.example을 복사하여 .env 생성
cp .env.example .env

# .env 파일 편집 (필수 값 설정)
nano .env
```

**필수 환경 변수:**
- `AZURE_AI_CREDENTIAL`: Azure AI 자격 증명
- `AZURE_AI_ENDPOINT`: Azure AI 엔드포인트
- `NOTION_TOKEN`: Notion 통합 토큰
- `DATA_SOURCE_ID`: Notion 데이터베이스 ID
- `LANGFUSE_SECRET_KEY`: Langfuse 비밀 키 (옵션)
- `UPSTAGE_API_KEY`: Upstage API 키 (임베딩 사용 시)

### 3. 애플리케이션 실행

```bash
# 모든 서비스 시작 (빌드 포함)
docker compose up -d --build

# 로그 확인
docker compose logs -f api

# 서비스 상태 확인
docker compose ps
```

### 4. 헬스체크

```bash
# API 헬스체크
curl http://localhost:8000/health

# Qdrant 헬스체크
curl http://localhost:6333/health
```

## 📡 API 엔드포인트

### 보고서 생성 API

**Endpoint:** `POST /generate-report`

**요청 예시:**

```json
{
  "report_type": "weekly",
  "question": "25년도 12월 첫째주 보고서 만들어줘",
  "output_format": "docx"
}
```

**응답 형식:**
- `output_format: "json"` - JSON 응답
- `output_format: "docx"` - Word 파일 다운로드
- `output_format: "pdf"` - PDF 파일 다운로드

**curl 예시:**

```bash
# JSON 응답
curl -X POST http://localhost:8000/generate-report \
  -H "Content-Type: application/json" \
  -d '{
    "report_type": "weekly",
    "question": "이번 주 보고서",
    "output_format": "json"
  }'

# Word 파일 다운로드
curl -X POST http://localhost:8000/generate-report \
  -H "Content-Type: application/json" \
  -d '{
    "report_type": "weekly",
    "question": "이번 주 보고서",
    "output_format": "docx"
  }' \
  --output report.docx
```

## 🔧 관리 명령어

### 서비스 관리

```bash
# 서비스 시작
docker compose up -d

# 서비스 중지
docker compose down

# 서비스 재시작
docker compose restart

# 특정 서비스만 재시작
docker compose restart api

# 로그 확인
docker compose logs -f api
docker compose logs -f qdrant

# 컨테이너 셸 접속
docker compose exec api bash
```

### 데이터 관리

```bash
# 데이터 백업
tar -czf backup_$(date +%Y%m%d).tar.gz data/

# Qdrant 데이터 초기화
docker compose down
rm -rf data/qdrant_data/*
docker compose up -d
```

### 업데이트

```bash
# 코드 업데이트 후 재배포
git pull
docker compose up -d --build

# 특정 서비스만 재빌드
docker compose up -d --build api
```

## 🌐 프로덕션 배포

### Nginx 리버스 프록시 설정

```nginx
server {
    listen 80;
    server_name your-domain.com;

    client_max_body_size 100M;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # 타임아웃 설정 (보고서 생성 시간 고려)
        proxy_connect_timeout 300s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;
    }
}
```

### SSL 인증서 설정 (Let's Encrypt)

```bash
# Certbot 설치
sudo apt install certbot python3-certbot-nginx

# SSL 인증서 발급
sudo certbot --nginx -d your-domain.com

# 자동 갱신 확인
sudo certbot renew --dry-run
```

### 환경 변수 보안

```bash
# .env 파일 권한 설정
chmod 600 .env

# Git에서 제외 확인
cat .gitignore | grep .env
```

## 📊 모니터링

### 로그 수집

```bash
# 최근 100줄 로그 확인
docker compose logs --tail=100 api

# 실시간 로그 스트리밍
docker compose logs -f api

# 에러 로그만 필터링
docker compose logs api | grep ERROR
```

### 리소스 사용량

```bash
# 컨테이너 리소스 모니터링
docker stats

# 디스크 사용량
docker system df
```

## 🐛 트러블슈팅

### 1. 포트 충돌

```bash
# 포트 사용 중인 프로세스 확인
sudo lsof -i :8000
sudo lsof -i :6333

# 프로세스 종료
sudo kill -9 <PID>
```

### 2. 메모리 부족

```bash
# docker-compose.yml에 메모리 제한 추가
services:
  api:
    deploy:
      resources:
        limits:
          memory: 2G
```

### 3. Qdrant 연결 실패

```bash
# Qdrant 컨테이너 상태 확인
docker compose ps qdrant

# Qdrant 로그 확인
docker compose logs qdrant

# Qdrant 재시작
docker compose restart qdrant
```

### 4. Word/PDF 생성 실패

```bash
# 컨테이너 내부에서 pandoc 확인
docker compose exec api pandoc --version

# LibreOffice 확인
docker compose exec api libreoffice --version

# 폰트 확인
docker compose exec api fc-list | grep Nanum
```

## 🔐 보안 권장사항

1. **환경 변수 관리**
   - `.env` 파일을 Git에 커밋하지 않기
   - 프로덕션에서는 비밀 관리 서비스 사용 (AWS Secrets Manager, HashiCorp Vault 등)

2. **API 보안**
   - API 키 인증 추가
   - Rate limiting 설정
   - CORS 정책 강화

3. **네트워크 보안**
   - 방화벽 설정
   - 불필요한 포트 닫기
   - SSL/TLS 사용

4. **정기 업데이트**
   - Docker 이미지 정기 업데이트
   - 보안 패치 적용

## 📚 추가 리소스

- [FastAPI 공식 문서](https://fastapi.tiangolo.com/)
- [Docker Compose 문서](https://docs.docker.com/compose/)
- [Qdrant 문서](https://qdrant.tech/documentation/)

