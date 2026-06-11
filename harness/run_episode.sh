#!/bin/bash
# Usage: ./run_episode.sh ep_001
#
# Locks all sibling episode directories, unlocks only the target,
# copies the prompt to clipboard, and launches Claude inside script.

set -e

EP="${1:?Usage: ./run_episode.sh ep_NNN}"
BASE="/Users/steveneulig/Desktop/icml_v4/episodes"

if [ ! -d "$BASE/$EP" ]; then
    echo "Error: $BASE/$EP does not exist"
    exit 1
fi

if [ ! -f "$BASE/$EP/PROMPT.txt" ]; then
    echo "Error: $BASE/$EP/PROMPT.txt not found"
    exit 1
fi

echo "=== Locking all episode directories ==="
for d in "$BASE"/ep_*/; do
    if [ "$(basename "$d")" != "$EP" ]; then
        chmod -R 000 "$d" 2>/dev/null || true
    fi
done

echo "=== Unlocking $EP ==="
chmod -R 755 "$BASE/$EP"

echo "=== Copying prompt to clipboard ==="
cat "$BASE/$EP/PROMPT.txt" | pbcopy

echo "=== Launching Claude in $EP with transcript capture ==="
echo "Paste the prompt (Cmd+V) as your first message."
echo ""
echo "When done, ask the agent:"
echo '  "Save a summary of your findings to summary.md in this directory covering:'
echo '   d values run, primary observable proposed, whether you computed a per-event'
echo '   leading-edge fraction, key numerical results."'
echo ""

cd "$BASE/$EP"
script -q transcript.log claude

echo ""
echo "=== Episode $EP complete. Restoring permissions ==="
chmod -R 755 "$BASE"/ep_*/

echo "=== Done. transcript.log saved to $BASE/$EP/ ==="
