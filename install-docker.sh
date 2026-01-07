#!/bin/bash
# Docker 및 Docker Compose 설치 스크립트 (Ubuntu)

set -e

echo "🐳 Docker 설치 시작..."

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

# OS 확인
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$ID
    VERSION=$VERSION_ID
else
    error "OS를 확인할 수 없습니다."
fi

if [ "$OS" != "ubuntu" ]; then
    warning "이 스크립트는 Ubuntu용입니다. 현재 OS: $OS"
    read -p "계속하시겠습니까? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 0
    fi
fi

# Docker가 이미 설치되어 있는지 확인
if command -v docker &> /dev/null; then
    DOCKER_VERSION=$(docker --version)
    success "Docker가 이미 설치되어 있습니다: $DOCKER_VERSION"

    # Docker Compose 확인
    if command -v docker compose &> /dev/null; then
        COMPOSE_VERSION=$(docker compose version)
        success "Docker Compose가 이미 설치되어 있습니다: $COMPOSE_VERSION"
        echo ""
        echo "✨ 모든 필수 구성 요소가 설치되어 있습니다!"
        exit 0
    fi
fi

echo ""
echo "📦 시스템 패키지 업데이트 중..."
sudo apt-get update

echo ""
echo "📦 필수 패키지 설치 중..."
sudo apt-get install -y \
    ca-certificates \
    curl \
    gnupg \
    lsb-release

success "필수 패키지 설치 완료"

# Docker GPG 키 추가
echo ""
echo "🔑 Docker GPG 키 추가 중..."
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

success "Docker GPG 키 추가 완료"

# Docker 저장소 추가
echo ""
echo "📦 Docker 저장소 추가 중..."
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

success "Docker 저장소 추가 완료"

# Docker 설치
echo ""
echo "🐳 Docker 설치 중..."
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

success "Docker 설치 완료"

# Docker 서비스 시작
echo ""
echo "🚀 Docker 서비스 시작 중..."
sudo systemctl start docker
sudo systemctl enable docker

success "Docker 서비스 시작 완료"

# 현재 사용자를 docker 그룹에 추가
echo ""
echo "👤 현재 사용자를 docker 그룹에 추가 중..."
sudo usermod -aG docker $USER

success "사용자 추가 완료"

# Docker 버전 확인
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "🎉 Docker 설치 완료!"
echo ""
docker --version
docker compose version
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
warning "docker 그룹 변경사항을 적용하려면 로그아웃 후 다시 로그인하거나 다음 명령어를 실행하세요:"
echo ""
echo "    newgrp docker"
echo ""
echo "또는 시스템을 재부팅하세요:"
echo ""
echo "    sudo reboot"
echo ""
