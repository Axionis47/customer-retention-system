#!/bin/bash
# Validation script to verify the complete setup

set -e

echo "🔍 Validating Churn-Saver RLHF+PPO Setup..."
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Track results
PASSED=0
FAILED=0

check() {
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓${NC} $1"
        ((PASSED++))
    else
        echo -e "${RED}✗${NC} $1"
        ((FAILED++))
    fi
}

# 1. Check directory structure
echo "📁 Checking directory structure..."
for dir in env agents rlhf serve eval models ops tests; do
    [ -d "$dir" ]
    check "Directory $dir exists"
done
echo ""

# 2. Check key files
echo "📄 Checking key files..."
for file in pyproject.toml Makefile README.md DEPLOYMENT.md PROJECT_SUMMARY.md; do
    [ -f "$file" ]
    check "File $file exists"
done
echo ""

# 3. Check Python modules
echo "🐍 Checking Python modules..."
for module in env/retention_env.py agents/ppo_policy.py agents/lagrangian.py serve/app.py; do
    [ -f "$module" ]
    check "Module $module exists"
done
echo ""

# 4. Check test files
echo "🧪 Checking test files..."
TEST_COUNT=$(find tests -name "test_*.py" | wc -l | tr -d ' ')
[ "$TEST_COUNT" -ge 10 ]
check "Found $TEST_COUNT test files (expected ≥10)"
echo ""

# 5. Check config files
echo "⚙️  Checking config files..."
for config in ops/configs/env.yaml ops/configs/ppo.yaml ops/configs/serve.yaml; do
    [ -f "$config" ]
    check "Config $config exists"
done
echo ""

# 6. Check Docker files
echo "🐳 Checking Docker files..."
for dockerfile in ops/docker/Dockerfile.app ops/docker/Dockerfile.trainer; do
    [ -f "$dockerfile" ]
    check "Dockerfile $dockerfile exists"
done
echo ""

# 7. Check Terraform files
echo "🏗️  Checking Terraform files..."
for tf in ops/terraform/main.tf ops/terraform/provider.tf ops/terraform/variables.tf; do
    [ -f "$tf" ]
    check "Terraform file $tf exists"
done
echo ""

# 8. Check Cloud Build
echo "☁️  Checking Cloud Build..."
[ -f "ops/cloudbuild.yaml" ]
check "Cloud Build config exists"
echo ""

# 9. Check baseline policies
echo "📊 Checking baseline policies..."
for baseline in agents/baselines/propensity_threshold.py agents/baselines/uplift_tree.py agents/baselines/ts_bandit.py; do
    [ -f "$baseline" ]
    check "Baseline $baseline exists"
done
echo ""

# 10. Check RLHF components
echo "🤖 Checking RLHF components..."
for rlhf in rlhf/sft_train.py rlhf/rm_train.py rlhf/ppo_text.py rlhf/safety/shield.py; do
    [ -f "$rlhf" ]
    check "RLHF component $rlhf exists"
done
echo ""

# 11. Check evaluation suite
echo "📈 Checking evaluation suite..."
for eval in eval/business_metrics.py eval/stress_tests.py eval/arena_ab.py eval/plots.py; do
    [ -f "$eval" ]
    check "Evaluation $eval exists"
done
echo ""

# 12. Check test fixtures
echo "🔧 Checking test fixtures..."
for fixture in tests/fixtures/tiny_telco.csv tests/fixtures/tiny_bank.csv tests/fixtures/tiny_pairs.jsonl; do
    [ -f "$fixture" ]
    check "Fixture $fixture exists"
done
echo ""

# 13. Python syntax check (if Python is available)
if command -v python3 &> /dev/null; then
    echo "🔍 Checking Python syntax..."
    python3 -m py_compile env/retention_env.py 2>/dev/null
    check "env/retention_env.py syntax valid"
    
    python3 -m py_compile agents/ppo_policy.py 2>/dev/null
    check "agents/ppo_policy.py syntax valid"
    
    python3 -m py_compile serve/app.py 2>/dev/null
    check "serve/app.py syntax valid"
    echo ""
fi

# 14. Check for required markers in pyproject.toml
echo "📋 Checking pyproject.toml configuration..."
grep -q "unit: Unit tests" pyproject.toml
check "Unit test marker configured"

grep -q "integration: Integration tests" pyproject.toml
check "Integration test marker configured"

grep -q "contract: Contract tests" pyproject.toml
check "Contract test marker configured"

grep -q "e2e: End-to-end tests" pyproject.toml
check "E2E test marker configured"
echo ""

# Summary
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 Validation Summary"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${GREEN}Passed: $PASSED${NC}"
echo -e "${RED}Failed: $FAILED${NC}"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✅ All validation checks passed!${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Run 'make setup' to install dependencies"
    echo "  2. Run 'make test' to execute all tests"
    echo "  3. See DEPLOYMENT.md for deployment instructions"
    exit 0
else
    echo -e "${RED}❌ Some validation checks failed${NC}"
    echo "Please review the errors above"
    exit 1
fi

