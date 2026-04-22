#!/bin/bash
# 01-startup.sh
echo "=========================================="
echo "   🚀 STARTING DEVELOPMENT SESSION"
echo "=========================================="

# 1. Pull code from GitHub
echo "[1/4] Pulling latest code from GitHub..."
if git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
    git pull origin $(git branch --show-current)
else
    echo "Warning: Not a git repository. Skipping git pull."
fi

# 2. Initialize OpenSpec (refresh slash commands & skills)
echo "[2/4] Initializing OpenSpec..."
if command -v openspec &> /dev/null; then
    openspec init
else
    echo "Warning: openspec not found in PATH. Skipping openspec init."
fi

# 3. Read the latest handover document
echo "[3/4] Checking for handover documents..."
# Numbering rule: 01-handover.md, 02-handover.md, etc.
LATEST_HANDOVER=$(ls [0-9][0-9]-handover.md 2>/dev/null | sort -r | head -n 1)

if [ -f "$LATEST_HANDOVER" ]; then
    echo "📄 Found Handover: $LATEST_HANDOVER"
    echo "------------------------------------------"
    cat "$LATEST_HANDOVER"
    echo "------------------------------------------"
    
    # 4. Suggest next actions
    echo "[4/4] 💡 Suggested Next Actions:"
    # Extract lines under "## Next Actions"
    grep -A 10 "## Next Actions" "$LATEST_HANDOVER" | grep -v "## Next Actions"
else
    echo "📄 No handover document found starting with 01-, 02-, etc."
    echo "[4/4] 💡 Suggested Next Actions:"
    echo "  - Create your first handover document: 01-handover.md"
    echo "  - Or start a new task using: /opsx:propose"
fi

echo "=========================================="
echo "   READY TO DEVELOP!"
echo "=========================================="
