# RAG Report Generator API 배포 가이드

## 배포 개요

이 가이드는 Systemd + Gunicorn을 사용하여 RAG Report Generator API를 프로덕션 환경에 배포하는 방법을 설명합니다.

## 사전 요구사항

- Python 3.11+
- 가상환경 설치 완료
- sudo 권한

## 빠른 배포

```bash
cd /home/work/rag/Project/rag-report-generator
./deploy.sh
```

## 수동 배포 단계

### 1. 환경 설정

```bash
# 프로젝트 디렉토리로 이동
cd /home/work/rag/Project/rag-report-generator

# 가상환경 활성화
source .venv/bin/activate

# 의존성 설치
pip install -r requirements.txt
```

### 2. 환경변수 확인

`.env` 파일이 올바르게 설정되어 있는지 확인:

```bash
# 필수 환경변수
AZURE_AI_CREDENTIAL=your_credential
AZURE_AI_ENDPOINT=your_endpoint
OPENROUTER_API_KEY=your_key
NOTION_TOKEN=your_token
# ... 기타 설정
```

### 3. Systemd 서비스 등록

```bash
# 서비스 파일 복사
sudo cp rag-api.service /etc/systemd/system/

# Systemd 재로드
sudo systemctl daemon-reload

# 서비스 활성화 (부팅 시 자동 시작)
sudo systemctl enable rag-api

# 서비스 시작
sudo systemctl start rag-api
```

### 4. 서비스 확인

```bash
# 서비스 상태 확인
sudo systemctl status rag-api

# 로그 실시간 확인
sudo journalctl -u rag-api -f

# API 헬스체크
curl http://localhost:8000/health
```

## 서비스 관리 명령어

```bash
# 서비스 시작
sudo systemctl start rag-api

# 서비스 중지
sudo systemctl stop rag-api

# 서비스 재시작
sudo systemctl restart rag-api

# 서비스 상태 확인
sudo systemctl status rag-api

# 부팅 시 자동 시작 활성화
sudo systemctl enable rag-api

# 부팅 시 자동 시작 비활성화
sudo systemctl disable rag-api
```

## 로그 확인

```bash
# 전체 로그 확인
sudo journalctl -u rag-api

# 실시간 로그 확인 (tail -f와 유사)
sudo journalctl -u rag-api -f

# 최근 100줄 확인
sudo journalctl -u rag-api -n 100

# 오늘의 로그만 확인
sudo journalctl -u rag-api --since today

# Gunicorn 로그 파일
tail -f logs/gunicorn-access.log
tail -f logs/gunicorn-error.log
```

## 외부 접근 설정

### 방법 1: SSH 포트 포워딩 (개발/테스트용)

로컬 PC의 `~/.ssh/config`에 추가:

```
Host H100
    HostName nipa.nhncloud.com
    User work
    Port 10514
    IdentityFile ~/.ssh/id_container
    LocalForward 8000 localhost:8000
```

접속 후 로컬에서 `http://localhost:8000/docs` 접근

### 방법 2: 퍼블릭 IP 직접 노출

#### NHN Cloud 콘솔 설정

1. 인스턴스의 퍼블릭 IP 확인
2. 보안 그룹 → 인바운드 규칙 추가:
   - 프로토콜: TCP
   - 포트: 8000
   - 소스: 0.0.0.0/0 (또는 특정 IP)

#### 서버 방화벽 설정

```bash
# UFW 사용 시
sudo ufw allow 8000/tcp
sudo ufw reload

# firewalld 사용 시
sudo firewall-cmd --add-port=8000/tcp --permanent
sudo firewall-cmd --reload
```

### 방법 3: Nginx 리버스 프록시 (프로덕션 권장)

#### Nginx 설치

```bash
sudo apt update
sudo apt install nginx
```

#### Nginx 설정

`/etc/nginx/sites-available/rag-api` 생성:

```nginx
server {
    listen 80;
    server_name your-domain.com;  # 또는 서버 IP

    client_max_body_size 100M;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # 타임아웃 설정
        proxy_connect_timeout 300;
        proxy_send_timeout 300;
        proxy_read_timeout 300;
    }
}
```

#### Nginx 활성화

```bash
# 심볼릭 링크 생성
sudo ln -s /etc/nginx/sites-available/rag-api /etc/nginx/sites-enabled/

# 설정 테스트
sudo nginx -t

# Nginx 재시작
sudo systemctl restart nginx
```

#### HTTPS 설정 (Let's Encrypt)

```bash
# Certbot 설치
sudo apt install certbot python3-certbot-nginx

# SSL 인증서 발급 및 자동 설정
sudo certbot --nginx -d your-domain.com
```

## 보안 설정

### 1. CORS 설정 수정

프로덕션에서는 `app/api/main.py`의 CORS 설정을 특정 도메인으로 제한:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://your-frontend-domain.com",
        "https://your-app.com"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

### 2. 환경변수 보안

```bash
# .env 파일 권한 설정
chmod 600 .env

# 민감한 정보는 환경변수로만 관리
# Git에 절대 커밋하지 않기
```

### 3. API 키 보호

- API 요청에 인증 토큰 추가 고려
- Rate limiting 설정
- IP 화이트리스트 설정

## 성능 튜닝

### Gunicorn 워커 수 조정

`gunicorn.conf.py`:

```python
# CPU 코어 수에 따라 조정
workers = (2 * cpu_count) + 1

# 또는 환경변수로 설정
GUNICORN_WORKERS=4 ./deploy.sh
```

### 타임아웃 조정

RAG 처리 시간이 오래 걸리는 경우:

```python
timeout = 300  # 5분
```

## 모니터링

### 서비스 상태 모니터링

```bash
# 서비스 상태 자동 확인 스크립트
watch -n 5 'systemctl status rag-api --no-pager'
```

### 로그 로테이션 설정

`/etc/logrotate.d/rag-api`:

```
/home/work/rag/Project/rag-report-generator/logs/*.log {
    daily
    rotate 14
    compress
    delaycompress
    notifempty
    missingok
    create 0644 work work
    sharedscripts
    postrotate
        systemctl reload rag-api > /dev/null 2>&1 || true
    endscript
}
```

## 트러블슈팅

### 서비스가 시작되지 않는 경우

```bash
# 에러 로그 확인
sudo journalctl -u rag-api -n 50

# 설정 파일 경로 확인
sudo systemctl cat rag-api

# 권한 확인
ls -la /home/work/rag/Project/rag-report-generator/
```

### 포트 충돌

```bash
# 8000번 포트 사용 중인 프로세스 확인
sudo lsof -i :8000
sudo netstat -tlnp | grep 8000

# 프로세스 종료
sudo kill -9 <PID>
```

### 메모리 부족

```bash
# 시스템 리소스 확인
free -h
df -h

# 워커 수 줄이기
# gunicorn.conf.py에서 workers 수 감소
```

## 업데이트 배포

```bash
# 코드 업데이트 후
cd /home/work/rag/Project/rag-report-generator
git pull  # 또는 코드 업데이트 방법

# 의존성 업데이트
source .venv/bin/activate
pip install -r requirements.txt

# 서비스 재시작
sudo systemctl restart rag-api

# 상태 확인
sudo systemctl status rag-api
```

## 롤백

문제 발생 시 이전 버전으로 복원:

```bash
# Git 사용 시
git checkout <previous-commit>

# 서비스 재시작
sudo systemctl restart rag-api
```

## 완전 제거

```bash
# 서비스 중지 및 비활성화
sudo systemctl stop rag-api
sudo systemctl disable rag-api

# 서비스 파일 제거
sudo rm /etc/systemd/system/rag-api.service
sudo systemctl daemon-reload
```

## 문의 및 지원

문제가 발생하면 로그를 확인하고 이슈를 기록하세요.
