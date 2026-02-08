#!/usr/bin/env bash

set -e

SOURCE="${1:-planogram_addon}"
OUTPUT="${2:-${SOURCE}.zip}"

[ -e "$SOURCE" ] || { echo "❌ Source not found: $SOURCE"; exit 1; }

# 1️⃣ 优先使用 zip（macOS / Linux / Git Bash）
if command -v zip >/dev/null 2>&1; then
    if [ -d "$SOURCE" ]; then
        zip -r "$OUTPUT" "$SOURCE"
    else
        zip "$OUTPUT" "$SOURCE"
    fi
    echo "✅ Created: $OUTPUT"
    exit 0
fi

# 2️⃣ fallback：PowerShell（Windows）
if command -v powershell >/dev/null 2>&1; then
    powershell -Command "Compress-Archive -Force '$SOURCE' '$OUTPUT'"
    echo "✅ Created: $OUTPUT"
    exit 0
fi

echo "❌ No zip or PowerShell found"
exit 1
