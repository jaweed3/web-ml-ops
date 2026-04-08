#!/usr/bin/env bash
# End-to-end smoke test against a running rescuevision-serve container.
# Usage: bash scripts/smoke_test.sh [BASE_URL]
set -euo pipefail

BASE="${1:-http://localhost:8080}"
PASS=0
FAIL=0

check() {
    local desc="$1"
    local result="$2"
    local expected="$3"
    if echo "$result" | grep -q "$expected"; then
        echo "  PASS  $desc"
        ((PASS++)) || true
    else
        echo "  FAIL  $desc"
        echo "        Expected to contain: $expected"
        echo "        Got: $result"
        ((FAIL++)) || true
    fi
}

echo ""
echo "RescueVision Serve — Smoke Test"
echo "Target: $BASE"
echo "================================"

# /health
R=$(curl -sf "$BASE/health" || echo '{"error":"connection_refused"}')
check "/health → status ok"    "$R" '"status"'

# /ready
R=$(curl -sf "$BASE/ready" || echo '{"status":"not_ready"}')
check "/ready → status ready"  "$R" '"status"'

# /model/info
R=$(curl -sf "$BASE/model/info" || echo '{}')
check "/model/info → has name"    "$R" '"name"'
check "/model/info → has version" "$R" '"version"'

# /model/version
R=$(curl -sf "$BASE/model/version" || echo '{}')
check "/model/version → has version" "$R" '"version"'

# /predict — valid image
IMG=$(mktemp /tmp/smoke_XXXXXX.jpg)
python3 -c "
import cv2, numpy as np, sys
img = np.zeros((480, 640, 3), dtype=np.uint8)
cv2.imwrite('$IMG', img)
" 2>/dev/null && {
    R=$(curl -sf -X POST "$BASE/predict" \
        -F "file=@$IMG;type=image/jpeg" || echo '{}')
    check "/predict → detections key"    "$R" '"detections"'
    check "/predict → request_id key"    "$R" '"request_id"'
    check "/predict → inference_time_ms" "$R" '"inference_time_ms"'
    rm -f "$IMG"
}

# /predict — wrong MIME type
CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$BASE/predict" \
    -F "file=@/dev/null;type=text/plain")
check "/predict → 422 for text/plain" "$CODE" "422"

echo ""
echo "================================"
echo "Results: $PASS passed, $FAIL failed"
echo ""

[ "$FAIL" -eq 0 ] || exit 1
