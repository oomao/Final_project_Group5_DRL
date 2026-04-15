#!/bin/bash
# 02-ending.sh
echo "=========================================="
echo "   🏁 WRAPPING UP DEVELOPMENT SESSION"
echo "=========================================="

# 1. Update tasks.md
echo "[1/5] Checking tasks.md..."
if [ ! -f "tasks.md" ]; then
    echo "Creating tasks.md..."
    echo "# Project Tasks" > tasks.md
    echo "- [x] Initialize project structure" >> tasks.md
fi
echo "✅ tasks.md is ready."

# 2. Archive the change if everything is complete
echo "[2/5] Checking OpenSpec status..."
if command -v openspec &> /dev/null; then
    # We could check if there are active changes to archive
    echo "Hint: If your current change is done, run: openspec archive"
else
    echo "OpenSpec command not found. Skipping archive."
fi

# 3. Write handover document for next development
echo "[3/5] Creating handover document..."
# Get the last handover number
LAST_FILE=$(ls [0-9][0-9]-handover.md 2>/dev/null | sort -r | head -n 1)
if [ -z "$LAST_FILE" ]; then
    NEXT_NUM="01"
else
    LAST_NUM=$(echo "$LAST_FILE" | cut -c1-2)
    # Remove leading zero for calculation, then add leading zero back
    NEXT_NUM=$(printf "%02d" $((10#$LAST_NUM + 1)))
fi

NEW_HANDOVER="${NEXT_NUM}-handover.md"

cat <<EOF > "$NEW_HANDOVER"
# Handover Document ($NEXT_NUM) - $(date +'%Y-%m-%d %H:%M')

## Summary of Changes
- Implemented startup and ending scripts.
- Added numbering rule (01-) for handovers and scripts.
- Updated package.json with dev:start and dev:ending.

## Current Status
- Scripts 01-startup.sh and 02-ending.sh are created.
- package.json is configured.

## Next Actions
- [ ] Test the startup script by running 'npm run dev:start'
- [ ] Continue organizing the portfolio projects.
EOF

echo "📄 Created: $NEW_HANDOVER"

# 4. Push code to GitHub
echo "[4/5] Pushing changes to GitHub..."
if git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
    git add .
    git commit -m "Wrap up session: Added $NEW_HANDOVER and updated scripts"
    git push origin $(git branch --show-current)
    echo "✅ Changes pushed."
else
    echo "Warning: Not a git repository. Skipping git push."
fi

echo "=========================================="
echo "   SESSION COMPLETE. SEE YOU NEXT TIME!"
echo "=========================================="
