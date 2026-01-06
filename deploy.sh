#!/bin/bash

# RAG Report Generator API 배포 스크립트
set -e

PROJECT_DIR="/home/work/rag/Project/rag-report-generator"
SERVICE_NAME="rag-api"

echo "=========================================="
echo "RAG Report Generator API 배포 시작"
echo "=========================================="

# 1. 프로젝트 디렉토리로 이동
cd "$PROJECT_DIR"

# 2. 가상환경 활성화
echo "[1/7] 가상환경 활성화..."
source .venv/bin/activate

# 3. 의존성 설치/업데이트
echo "[2/7] 의존성 설치..."
pip install -r requirements.txt

# 4. 로그 디렉토리 생성
echo "[3/7] 로그 디렉토리 생성..."
mkdir -p logs

# 5. Systemd 서비스 파일 복사 (sudo 권한 필요)
echo "[4/7] Systemd 서비스 등록..."
sudo cp rag-api.service /etc/systemd/system/
sudo systemctl daemon-reload

# 6. 서비스 활성화 및 시작
echo "[5/7] 서비스 시작..."
sudo systemctl enable $SERVICE_NAME
sudo systemctl restart $SERVICE_NAME

# 7. 상태 확인
echo "[6/7] 서비스 상태 확인..."
sleep 3
sudo systemctl status $SERVICE_NAME --no-pager

# 8. 헬스체크
echo "[7/7] API 헬스체크..."
sleep 2
curl -s http://localhost:8000/health || echo "헬스체크 실패"

echo ""
echo "=========================================="
echo "배포 완료!"
echo "=========================================="
echo "서비스 관리 명령어:"
echo "  - 상태 확인: sudo systemctl status $SERVICE_NAME"
echo "  - 로그 확인: sudo journalctl -u $SERVICE_NAME -f"
echo "  - 재시작: sudo systemctl restart $SERVICE_NAME"
echo "  - 중지: sudo systemctl stop $SERVICE_NAME"
echo ""
echo "API 접속: http://localhost:8000/docs"
echo "=========================================="
