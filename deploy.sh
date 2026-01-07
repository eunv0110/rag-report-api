#!/bin/bash
# RAG Report Generator 배포 스크립트

set -e

echo "🚀 RAG Report Generator 배포 시작..."

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 함수: 에러 메시지 출력
error() {
    echo -e "${RED}❌ 오류: $1${NC}"
    exit 1
}

# 함수: 성공 메시지 출력
success() {
    echo -e "${GREEN}✅ $1${NC}"
}

# 함수: 경고 메시지 출력
warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# Docker 설치 확인
if ! command -v docker &> /dev/null; then
    error "Docker가 설치되어 있지 않습니다. 먼저 Docker를 설치해주세요."
fi

if ! command -v docker compose &> /dev/null; then
    error "Docker Compose가 설치되어 있지 않습니다. 먼저 Docker Compose를 설치해주세요."
fi

success "Docker 및 Docker Compose 확인 완료"

# .env 파일 확인
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        warning ".env 파일이 없습니다. .env.example을 복사합니다."
        cp .env.example .env
        echo ""
        echo "📝 .env 파일을 편집하여 필수 환경 변수를 설정해주세요:"
        echo "   - AZURE_AI_CREDENTIAL"
        echo "   - AZURE_AI_ENDPOINT"
        echo "   - NOTION_TOKEN"
        echo "   - DATA_SOURCE_ID"
        echo ""
        read -p "환경 변수 설정을 완료했습니까? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            error "환경 변수 설정 후 다시 실행해주세요."
        fi
    else
        error ".env 파일과 .env.example 파일이 모두 없습니다."
    fi
fi

success ".env 파일 확인 완료"

# 데이터 디렉토리 생성
echo "📁 데이터 디렉토리 생성 중..."
mkdir -p data/reports
mkdir -p data/qdrant_data
mkdir -p data/notion_images

success "데이터 디렉토리 생성 완료"

# 이전 컨테이너 중지 (있는 경우)
if [ "$(docker compose ps -q)" ]; then
    echo "🛑 기존 컨테이너 중지 중..."
    docker compose down
    success "기존 컨테이너 중지 완료"
fi

# Docker 이미지 빌드
echo "🔨 Docker 이미지 빌드 중..."
docker compose build --no-cache

success "Docker 이미지 빌드 완료"

# 서비스 시작
echo "🚀 서비스 시작 중..."
docker compose up -d

success "서비스 시작 완료"

# 서비스 상태 확인
echo ""
echo "⏳ 서비스 시작 대기 중 (30초)..."
sleep 30

echo ""
echo "🔍 서비스 상태 확인 중..."
docker compose ps

# Qdrant 헬스체크
echo ""
echo "🔍 Qdrant 헬스체크..."
if curl -s -f http://localhost:6333/health > /dev/null; then
    success "Qdrant 정상 작동"
else
    warning "Qdrant 헬스체크 실패. 로그를 확인하세요: docker compose logs qdrant"
fi

# API 헬스체크
echo ""
echo "🔍 API 헬스체크..."
if curl -s -f http://localhost:8000/health > /dev/null; then
    success "API 정상 작동"
    echo ""
    echo "📡 API 엔드포인트:"
    echo "   - API 문서: http://localhost:8000/docs"
    echo "   - 헬스체크: http://localhost:8000/health"
    echo "   - 보고서 생성: http://localhost:8000/generate-report"
else
    warning "API 헬스체크 실패. 로그를 확인하세요: docker compose logs api"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
success "배포 완료! 🎉"
echo ""
echo "📝 유용한 명령어:"
echo "   - 로그 확인: docker compose logs -f api"
echo "   - 서비스 중지: docker compose down"
echo "   - 서비스 재시작: docker compose restart"
echo "   - 서비스 상태: docker compose ps"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
