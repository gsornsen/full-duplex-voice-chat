#!/bin/bash
# Quick fix for test imports using sed
set -e

echo "Fixing test imports..."

# Fix all test files
find tests/ -name "*.py" -type f -exec sed -i \
    -e 's/from src\.rpc\./from rpc./g' \
    -e 's/import src\.rpc\./import rpc./g' \
    -e 's/from src\.orchestrator\./from orchestrator./g' \
    -e 's/import src\.orchestrator\./import orchestrator./g' \
    -e 's/from src\.tts\./from tts./g' \
    -e 's/import src\.tts\./import tts./g' \
    -e 's/from src\.asr\./from orchestrator.asr./g' \
    -e 's/import src\.asr\./import orchestrator.asr./g' \
    -e 's/from src\.common\./from shared./g' \
    -e 's/import src\.common\./import shared./g' \
    -e 's/from src\.plugins\./from plugins./g' \
    -e 's/import src\.plugins\./import plugins./g' \
    {} +

echo "✓ Test imports fixed!"
