#!/usr/bin/env bash
# CosyVoice 2 Docker Quick Start Script
#
# This script provides common operations for CosyVoice 2 Docker deployment.
# Usage: ./scripts/cosyvoice_docker_quick_start.sh [command]

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Functions
print_header() {
    echo -e "${GREEN}===================================================${NC}"
    echo -e "${GREEN}$1${NC}"
    echo -e "${GREEN}===================================================${NC}"
}

print_error() {
    echo -e "${RED}ERROR: $1${NC}" >&2
}

print_warning() {
    echo -e "${YELLOW}WARNING: $1${NC}"
}

check_prerequisites() {
    print_header "Checking Prerequisites"

    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker not found. Please install Docker first."
        exit 1
    fi
    echo "✓ Docker installed: $(docker --version)"

    # Check Docker Compose
    if ! docker compose version &> /dev/null; then
        print_error "Docker Compose not found. Please install Docker Compose v2."
        exit 1
    fi
    echo "✓ Docker Compose installed: $(docker compose version)"

    # Check NVIDIA Docker
    if ! docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi &> /dev/null; then
        print_warning "NVIDIA Docker runtime not working. GPU may not be accessible."
    else
        echo "✓ NVIDIA Docker runtime working"
    fi

    # Check voicepack
    if [ ! -d "voicepacks/cosyvoice/en-base" ]; then
        print_warning "Voicepack not found. Run: ./scripts/setup_cosyvoice_voicepack.sh"
    else
        echo "✓ Voicepack present: voicepacks/cosyvoice/en-base"
    fi
}

build_image() {
    print_header "Building CosyVoice Docker Image"
    docker build -f Dockerfile.tts-cosyvoice -t tts-cosyvoice:latest .
    echo -e "${GREEN}✓ Image built successfully${NC}"
}

start_services() {
    print_header "Starting CosyVoice Services"

    # Create .env.cosyvoice if not exists
    if [ ! -f .env.cosyvoice ]; then
        print_warning ".env.cosyvoice not found, creating from template..."
        cp .env.cosyvoice.example .env.cosyvoice
    fi

    docker compose --profile cosyvoice up -d
    echo -e "${GREEN}✓ Services started${NC}"

    echo ""
    print_header "Waiting for Services to Become Healthy"
    docker compose ps

    echo ""
    echo "Check logs: docker compose logs -f tts-cosyvoice"
    echo "Stop services: docker compose down"
}

stop_services() {
    print_header "Stopping CosyVoice Services"
    docker compose down
    echo -e "${GREEN}✓ Services stopped${NC}"
}

restart_services() {
    print_header "Restarting CosyVoice Services"
    docker compose restart tts-cosyvoice
    echo -e "${GREEN}✓ Service restarted${NC}"
}

show_logs() {
    print_header "CosyVoice Service Logs"
    docker compose logs -f --tail=50 tts-cosyvoice
}

run_tests() {
    print_header "Running Performance Tests"

    # Check if service is running
    if ! docker compose ps tts-cosyvoice | grep -q "Up"; then
        print_error "CosyVoice service is not running. Start it first: ./scripts/cosyvoice_docker_quick_start.sh start"
        exit 1
    fi

    # Wait for service to be healthy
    echo "Waiting for service to be healthy..."
    for i in {1..30}; do
        if docker compose exec -T tts-cosyvoice python3 -c "import grpc; grpc.insecure_channel('localhost:7002').close()" 2>/dev/null; then
            echo "✓ Service is healthy"
            break
        fi
        if [ "$i" -eq 30 ]; then
            print_error "Service did not become healthy in time"
            exit 1
        fi
        sleep 2
    done

    # Run performance tests
    echo ""
    uv run pytest tests/performance/test_cosyvoice_performance.py -v --gpu

    echo -e "\n${GREEN}✓ Tests completed${NC}"
}

shell() {
    print_header "Opening Shell in CosyVoice Container"
    docker compose exec tts-cosyvoice bash
}

gpu_stats() {
    print_header "GPU Statistics"
    docker compose exec tts-cosyvoice nvidia-smi
}

show_status() {
    print_header "CosyVoice Service Status"
    docker compose ps tts-cosyvoice

    echo ""
    print_header "Container Statistics"
    docker stats --no-stream tts-cosyvoice

    echo ""
    print_header "GPU Utilization"
    if docker compose exec -T tts-cosyvoice nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader 2>/dev/null; then
        :
    else
        print_warning "GPU stats not available (container may not be running)"
    fi
}

clean() {
    print_header "Cleaning Up Docker Resources"
    echo "This will remove containers, networks, and volumes."
    read -p "Continue? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        docker compose down -v
        docker system prune -f
        echo -e "${GREEN}✓ Cleanup complete${NC}"
    else
        echo "Cancelled"
    fi
}

show_help() {
    cat << EOF
CosyVoice 2 Docker Quick Start

Usage: $0 [command]

Commands:
  check       Check prerequisites (Docker, GPU, voicepack)
  build       Build CosyVoice Docker image
  start       Start CosyVoice services (Redis + tts-cosyvoice)
  stop        Stop all services
  restart     Restart CosyVoice service
  logs        Show and follow service logs
  test        Run performance validation tests
  shell       Open bash shell in container
  gpu         Show GPU statistics
  status      Show service status and resource usage
  clean       Clean up Docker resources (containers, volumes)
  help        Show this help message

Examples:
  $0 check                  # Verify setup
  $0 build                  # Build Docker image
  $0 start                  # Start services
  $0 logs                   # Watch logs
  $0 test                   # Run performance tests
  $0 status                 # Check service health

For detailed documentation, see: docs/DOCKER_DEPLOYMENT_COSYVOICE.md
EOF
}

# Main script
case "${1:-help}" in
    check)
        check_prerequisites
        ;;
    build)
        build_image
        ;;
    start)
        start_services
        ;;
    stop)
        stop_services
        ;;
    restart)
        restart_services
        ;;
    logs)
        show_logs
        ;;
    test)
        run_tests
        ;;
    shell)
        shell
        ;;
    gpu)
        gpu_stats
        ;;
    status)
        show_status
        ;;
    clean)
        clean
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        print_error "Unknown command: $1"
        echo ""
        show_help
        exit 1
        ;;
esac
