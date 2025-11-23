#!/bin/bash
set -e

echo "🔍 Checking Python site-packages paths..."
PYTHON_SITE=$(python3 - << 'EOF'
import site
for p in site.getsitepackages():
    print(p)
EOF
)

USER_SITE=$(python3 - << 'EOF'
import site
print(site.getusersitepackages())
EOF
)

echo "📌 System site-packages:"
echo "$PYTHON_SITE"
echo "📌 User site-packages:"
echo "$USER_SITE"

echo ""
echo "🧹 Starting cleanup of residual XLA / _XLAC libraries..."

TARGETS=(
    "$USER_SITE/_XLAC.so"
    "$USER_SITE/_XLAC"*
    "$USER_SITE/xla"*
    "$USER_SITE/torch_xla"*
    "$USER_SITE/_XLAC_cuda_functions"*
)

for t in "${TARGETS[@]}"; do
    if ls $t 1>/dev/null 2>&1; then
        echo "🗑️ Deleting: $t"
        rm -rf $t
    fi
done

echo ""
echo "🧽 Uninstalling potentially residual pip package torch_xla..."
pip uninstall -y torch_xla 2>/dev/null || true
pip uninstall -y torch-xla 2>/dev/null || true

echo ""
echo "🔎 Checking again to confirm..."

if ls $USER_SITE | grep -i XLAC 1>/dev/null 2>&1; then
    echo "⚠️ Residual files still exist, please manually check $USER_SITE"
else
    echo "✅ Cleanup complete: All _XLAC / torch_xla residual files have been deleted"
fi

echo ""
echo "🎉 Done! Your Python environment has now completely removed conflicting XLA libraries."
