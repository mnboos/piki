#!/usr/bin/env bash
set -euo pipefail

BASE_URL="https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/Ultralytics_YOLO_OE_1.2.8"
MODEL_DIR="$(cd "$(dirname "$0")/.." && pwd)/model"
VARIANTS=(n s m l x)
FORCE=false

if [ "${1:-}" = "--force" ]; then
    FORCE=true
fi

mkdir -p "$MODEL_DIR"

for variant in "${VARIANTS[@]}"; do
    filename="yolo26${variant}_seg_bayese_640x640_nv12.bin"
    dest="$MODEL_DIR/$filename"

    if [ -f "$dest" ] && [ "$FORCE" != true ]; then
        echo "[skip] $filename (already exists, use --force to re-download)"
        continue
    fi

    echo "[downloading] $filename ..."
    curl -fSL --progress-bar -o "$dest" "${BASE_URL}/${filename}"
    echo "[done] $filename"
done

echo ""
echo "All models downloaded to $MODEL_DIR/"
ls -lh "$MODEL_DIR"/yolo26*_seg_bayese_640x640_nv12.bin
