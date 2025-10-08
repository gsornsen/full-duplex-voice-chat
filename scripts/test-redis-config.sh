#!/usr/bin/env bash
# Test Redis configuration for Docker Compose setup
# Verifies that services use correct internal Redis URLs

set -euo pipefail

echo "=========================================="
echo "Redis Configuration Test"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

success() {
    echo -e "${GREEN}✓${NC} $1"
}

error() {
    echo -e "${RED}✗${NC} $1"
}

warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

info() {
    echo "  $1"
}

ERRORS=0

echo "1. Checking docker-compose.yml configuration..."
echo ""

# Check Redis service has no external port mapping
if grep -A 5 "^  redis:" docker-compose.yml | grep -q "ports:"; then
    error "Redis service has external port mapping (should be internal only)"
    info "Found: $(grep -A 5 '^  redis:' docker-compose.yml | grep 'ports:' || true)"
    ERRORS=$((ERRORS + 1))
else
    success "Redis service uses internal network only"
fi

# Check orchestrator uses correct Redis URL
ORCH_REDIS_URL=$(grep -A 20 "^  orchestrator:" docker-compose.yml | grep "REDIS_URL=" | sed 's/.*REDIS_URL=//' | tr -d ' ')
if [ "$ORCH_REDIS_URL" = "redis://redis:6379" ]; then
    success "Orchestrator uses correct Redis URL: $ORCH_REDIS_URL"
else
    error "Orchestrator has incorrect Redis URL: $ORCH_REDIS_URL"
    info "Expected: redis://redis:6379"
    ERRORS=$((ERRORS + 1))
fi

# Check TTS worker uses correct Redis URL
TTS_REDIS_URL=$(grep -A 20 "^  tts0:" docker-compose.yml | grep "REDIS_URL=" | sed 's/.*REDIS_URL=//' | tr -d ' ')
if [ "$TTS_REDIS_URL" = "redis://redis:6379" ]; then
    success "TTS worker uses correct Redis URL: $TTS_REDIS_URL"
else
    error "TTS worker has incorrect Redis URL: $TTS_REDIS_URL"
    info "Expected: redis://redis:6379"
    ERRORS=$((ERRORS + 1))
fi

# Check Redis health check uses correct port
if grep -A 3 "redis:" docker-compose.yml | grep -q "redis-cli.*6379"; then
    success "Redis health check uses correct internal port (6379)"
else
    warning "Redis health check may not specify port 6379 explicitly"
fi

echo ""
echo "2. Checking for hardcoded port 6380 references..."
echo ""

# Search for any references to 6380 in config files
if grep -r "6380" configs/ 2>/dev/null | grep -v "^Binary"; then
    error "Found references to port 6380 in configs/"
    info "All internal references should use port 6379"
    ERRORS=$((ERRORS + 1))
else
    success "No hardcoded port 6380 references in configs/"
fi

# Search for 6380 in Python source
if grep -r "6380" src/ --include="*.py" 2>/dev/null | grep -v "^Binary"; then
    error "Found references to port 6380 in Python source"
    info "All internal references should use port 6379 or environment variables"
    ERRORS=$((ERRORS + 1))
else
    success "No hardcoded port 6380 references in Python source"
fi

echo ""
echo "3. Checking .env.example configuration..."
echo ""

if grep -q "REDIS_URL=redis://localhost:6379" .env.example 2>/dev/null; then
    success ".env.example uses correct Redis URL for local development"
else
    warning ".env.example may have incorrect Redis URL"
    info "Expected: REDIS_URL=redis://localhost:6379"
fi

echo ""
echo "4. Checking for host Redis/Valkey conflicts..."
echo ""

# Check if port 6379 is in use on host
if command -v lsof &> /dev/null; then
    if sudo lsof -i :6379 -sTCP:LISTEN &> /dev/null 2>&1; then
        warning "Port 6379 is in use on host (likely existing Redis/Valkey)"
        info "This is OK - Docker Redis uses internal network and won't conflict"
    else
        info "Port 6379 is available on host"
    fi
elif command -v ss &> /dev/null; then
    if ss -tln | grep -q ":6379"; then
        warning "Port 6379 is in use on host (likely existing Redis/Valkey)"
        info "This is OK - Docker Redis uses internal network and won't conflict"
    else
        info "Port 6379 is available on host"
    fi
else
    info "Cannot check port status (lsof/ss not available)"
fi

echo ""
echo "=========================================="
if [ $ERRORS -eq 0 ]; then
    echo -e "${GREEN}All checks passed!${NC}"
    echo ""
    echo "Configuration summary:"
    echo "  - Redis runs on Docker internal network (no host port exposure)"
    echo "  - Services use redis://redis:6379 for internal communication"
    echo "  - No conflicts with host Redis/Valkey instances"
    echo ""
    echo "To test the configuration:"
    echo "  1. docker compose up -d"
    echo "  2. docker compose exec orchestrator redis-cli -h redis -p 6379 ping"
    echo "  3. Expected output: PONG"
else
    echo -e "${RED}Found $ERRORS error(s)${NC}"
    echo ""
    echo "Please review the errors above and fix the configuration."
    exit 1
fi
echo "=========================================="
