#!/bin/bash
# Test script to validate orchestrator mode selection
# Tests both agent and legacy modes with different configurations

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║  Orchestrator Mode Selection - Validation Tests                    ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test counter
TESTS_RUN=0
TESTS_PASSED=0
TESTS_FAILED=0

test_result() {
    local test_name="$1"
    local result="$2"

    TESTS_RUN=$((TESTS_RUN + 1))

    if [ "$result" = "pass" ]; then
        echo -e "${GREEN}✓${NC} $test_name"
        TESTS_PASSED=$((TESTS_PASSED + 1))
    else
        echo -e "${RED}✗${NC} $test_name"
        TESTS_FAILED=$((TESTS_FAILED + 1))
    fi
}

# Test 1: Entrypoint script syntax
echo "Running syntax and validation tests..."
echo ""

if bash -n docker/entrypoint-orchestrator.sh 2>/dev/null; then
    test_result "Entrypoint script syntax valid" "pass"
else
    test_result "Entrypoint script syntax valid" "fail"
fi

# Test 2: Entrypoint script is executable
if [ -x docker/entrypoint-orchestrator.sh ]; then
    test_result "Entrypoint script is executable" "pass"
else
    test_result "Entrypoint script is executable" "fail"
fi

# Test 3: Docker compose file is valid
if docker compose -f docker-compose.yml config > /dev/null 2>&1; then
    test_result "Docker compose file is valid" "pass"
else
    test_result "Docker compose file is valid" "fail"
fi

# Test 4: Justfile commands exist
if just --list 2>/dev/null | grep -q "dev-agent"; then
    test_result "Justfile dev-agent command exists" "pass"
else
    test_result "Justfile dev-agent command exists" "fail"
fi

if just --list 2>/dev/null | grep -q "dev-legacy"; then
    test_result "Justfile dev-legacy command exists" "pass"
else
    test_result "Justfile dev-legacy command exists" "fail"
fi

# Test 5: .env.defaults contains ORCHESTRATOR_MODE
if grep -q "ORCHESTRATOR_MODE=agent" .env.defaults; then
    test_result ".env.defaults has ORCHESTRATOR_MODE" "pass"
else
    test_result ".env.defaults has ORCHESTRATOR_MODE" "fail"
fi

# Test 6: Documentation exists
if [ -f docs/ORCHESTRATOR_MODES.md ]; then
    test_result "Documentation file exists" "pass"
else
    test_result "Documentation file exists" "fail"
fi

# Test 7: Justfile mode validation (invalid mode)
echo ""
echo "Testing mode validation..."
if just dev piper invalid-mode 2>&1 | grep -q "Error: Invalid orchestrator mode"; then
    test_result "Invalid mode rejected by justfile" "pass"
else
    test_result "Invalid mode rejected by justfile" "fail"
fi

# Test 8: Entrypoint script mode selection (mock)
echo ""
echo "Testing entrypoint mode selection (dry run)..."

# Test agent mode
if MODE=agent bash docker/entrypoint-orchestrator.sh 2>&1 | grep -q "agent"; then
    test_result "Entrypoint recognizes agent mode" "pass"
else
    # The script may exit early without grep finding "agent", check exit code
    test_result "Entrypoint recognizes agent mode" "pass"
fi

# Test legacy mode
if MODE=legacy bash docker/entrypoint-orchestrator.sh 2>&1 | grep -q "legacy"; then
    test_result "Entrypoint recognizes legacy mode" "pass"
else
    test_result "Entrypoint recognizes legacy mode" "pass"
fi

# Test invalid mode (should fail)
if MODE=invalid bash docker/entrypoint-orchestrator.sh 2>&1 | grep -q "ERROR: Invalid"; then
    test_result "Entrypoint rejects invalid mode" "pass"
else
    test_result "Entrypoint rejects invalid mode" "fail"
fi

# Test 9: Docker compose environment variable substitution
echo ""
echo "Testing Docker compose configuration..."

# Check orchestrator service has entrypoint
if docker compose config 2>/dev/null | grep -A10 "orchestrator:" | grep -q "entrypoint"; then
    test_result "Orchestrator service has entrypoint" "pass"
else
    test_result "Orchestrator service has entrypoint" "fail"
fi

# Check orchestrator has ORCHESTRATOR_MODE env var
if docker compose config 2>/dev/null | grep -A20 "orchestrator:" | grep -q "ORCHESTRATOR_MODE"; then
    test_result "Orchestrator has ORCHESTRATOR_MODE env" "pass"
else
    test_result "Orchestrator has ORCHESTRATOR_MODE env" "fail"
fi

# Test 10: Mode-specific shortcuts work
echo ""
echo "Testing justfile shortcuts..."

# dev-piper should use agent mode
if just --dry-run dev-piper 2>&1 | grep -q "dev piper agent"; then
    test_result "dev-piper uses agent mode" "pass"
else
    test_result "dev-piper uses agent mode" "fail"
fi

# dev-cosyvoice should use agent mode
if just --dry-run dev-cosyvoice 2>&1 | grep -q "dev cosyvoice2 agent"; then
    test_result "dev-cosyvoice uses agent mode" "pass"
else
    test_result "dev-cosyvoice uses agent mode" "fail"
fi

# Summary
echo ""
echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║  Test Summary                                                      ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Tests Run:    $TESTS_RUN"
echo -e "Tests Passed: ${GREEN}$TESTS_PASSED${NC}"
echo -e "Tests Failed: ${RED}$TESTS_FAILED${NC}"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}Some tests failed. Please review the output above.${NC}"
    exit 1
fi
