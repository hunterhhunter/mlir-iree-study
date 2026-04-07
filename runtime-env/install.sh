#!/bin/bash
# 환경 설치 스크립트
# onnxruntime-gpu vs onnxruntime (CPU) 충돌 문제를 해결하기 위해
# constraints.txt와 후처리 제거를 조합합니다.

set -e

if [ -n "$CONDA_PREFIX" ]; then
    PYTHON="${CONDA_PREFIX}/bin/python"
elif [ -n "$VIRTUAL_ENV" ]; then
    PYTHON="${VIRTUAL_ENV}/bin/python"
else
    echo "[Error] 활성화된 환경(conda/venv)이 없습니다."
    exit 1
fi
UV="uv pip"

echo "[1/3] 패키지 설치 (constraints 적용)..."
${UV} install -r requirements.txt -c constraints.txt --python "${PYTHON}"

echo "[2/3] ultralytics 등이 설치한 CPU 전용 onnxruntime 제거 후 GPU 버전 복구..."
# onnxruntime (CPU)과 onnxruntime-gpu는 Python 파일을 공유하므로
# CPU 버전이 있으면 GPU 버전을 force-reinstall로 덮어씁니다.
if "${PYTHON}" -c "import importlib.metadata; importlib.metadata.version('onnxruntime')" &>/dev/null 2>&1; then
    echo "  -> onnxruntime (CPU) 발견. GPU 버전으로 force-reinstall..."
    ${UV} install onnxruntime-gpu==1.24.4 --force-reinstall --no-deps --python "${PYTHON}"
else
    echo "  -> onnxruntime (CPU) 없음. 패스."
fi

echo "[3/3] onnxruntime-gpu 재확인..."
${UV} show onnxruntime-gpu 2>/dev/null || echo "installed"
"${PYTHON}" -c "
import onnxruntime as ort
providers = ort.get_available_providers()
print('  사용 가능한 Provider:', providers)
if 'CUDAExecutionProvider' in providers:
    print('  [OK] CUDAExecutionProvider 활성화됨')
else:
    print('  [WARN] CUDAExecutionProvider 없음 — CUDA 환경을 확인하세요')
"

echo "설치 완료."
