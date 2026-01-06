#!/bin/bash

echo "========================================="
echo "RAG Report Generator API 배포 시작"
echo "========================================="

# 1. systemd 서비스 파일 복사
echo "1. systemd 서비스 등록..."
sudo cp rag-api.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable rag-api
sudo systemctl restart rag-api
sudo systemctl status rag-api

# 2. Nginx 설정 (Nginx가 설치되어 있는 경우)
if command -v nginx &> /dev/null; then
    echo ""
    echo "2. Nginx 설정..."
    sudo cp nginx-rag-api.conf /etc/nginx/sites-available/rag-api
    sudo ln -sf /etc/nginx/sites-available/rag-api /etc/nginx/sites-enabled/
    sudo nginx -t
    sudo systemctl restart nginx
    echo "✅ Nginx 설정 완료"
else
    echo ""
    echo "⚠️  Nginx가 설치되어 있지 않습니다."
    echo "Nginx 없이 Gunicorn만 사용하려면 rag-api.service의 bind 주소를 0.0.0.0:8000으로 변경하세요."
fi

echo ""
echo "========================================="
echo "✅ 배포 완료!"
echo "========================================="
echo "서비스 상태 확인: sudo systemctl status rag-api"
echo "로그 확인: sudo journalctl -u rag-api -f"
echo "API 테스트: curl http://localhost/health"
echo "========================================="
