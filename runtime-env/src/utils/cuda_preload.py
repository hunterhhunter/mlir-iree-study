"""
CUDA 라이브러리 사전 로드 유틸리티

nvidia-*-cu?? pip 패키지로 설치된 CUDA 라이브러리를 onnxruntime import 전에
ctypes.CDLL(RTLD_GLOBAL)로 사전 로드합니다.

문제 상황:
  onnxruntime-gpu는 CUDAExecutionProvider를 활성화할 때 libcublasLt.so 등을
  dlopen()으로 찾습니다. 이 라이브러리들이 Python site-packages 안에만 있고
  ld.so의 기본 탐색 경로(LD_LIBRARY_PATH, ldconfig)에 없으면 로드가 실패해
  CUDAExecutionProvider가 비활성화됩니다.

  preload_cuda_libs()를 `import onnxruntime` 이전에 호출하면 해결됩니다.

지원 CUDA 버전: 11.x, 12.x
  (CUDA 13.x 드라이버는 cu12 패키지와 하위 호환됩니다.)
"""

import ctypes
import glob
import importlib
import importlib.metadata
import os
import re
import site
import subprocess
from typing import Optional


# ---------------------------------------------------------------------------
# CUDA 버전별 라이브러리 목록
#
# 형식: (nvidia_python_module, .so_파일명)
# nvidia_python_module: importlib.import_module()로 접근 가능한 nvidia 하위 패키지
# .so_파일명: 해당 패키지 lib 디렉터리 내 실제 파일 이름
#
# 주의: .so 버전 번호는 CUDA major 버전과 다를 수 있습니다.
#   예) cu12 패키지의 libcufft.so.11, libcurand.so.10 은 라이브러리 자체 버전 번호입니다.
# ---------------------------------------------------------------------------
_CUDA_LIB_TABLE: dict[int, list[tuple[str, str]]] = {
    11: [
        ("nvidia.cuda_runtime.lib", "libcudart.so.11.0"),
        ("nvidia.cublas.lib",       "libcublas.so.11"),
        ("nvidia.cublas.lib",       "libcublasLt.so.11"),
        ("nvidia.cufft.lib",        "libcufft.so.10"),
        ("nvidia.curand.lib",       "libcurand.so.10"),
        ("nvidia.cusolver.lib",     "libcusolver.so.11"),
        ("nvidia.cusparse.lib",     "libcusparse.so.11"),
    ],
    12: [
        ("nvidia.cuda_runtime.lib", "libcudart.so.12"),
        ("nvidia.cublas.lib",       "libcublas.so.12"),
        ("nvidia.cublas.lib",       "libcublasLt.so.12"),
        ("nvidia.cufft.lib",        "libcufft.so.11"),
        ("nvidia.curand.lib",       "libcurand.so.10"),
        ("nvidia.cusolver.lib",     "libcusolver.so.11"),
        ("nvidia.cusparse.lib",     "libcusparse.so.12"),
        ("nvidia.nvjitlink.lib",    "libnvJitLink.so.12"),
    ],
}

# cuDNN: CUDA 버전과 독립적으로 관리되므로 별도 목록 (버전 우선순위 순)
_CUDNN_SO_NAMES = ["libcudnn.so.9", "libcudnn.so.8"]


# ---------------------------------------------------------------------------
# 공개 API
# ---------------------------------------------------------------------------

def preload_cuda_libs() -> None:
    """
    nvidia-*-cu?? pip 패키지에서 CUDA 라이브러리를 RTLD_GLOBAL로 사전 로드합니다.

    - 설치된 CUDA 버전을 자동 감지합니다 (cu12 → cu11 순 fallback).
    - CUDA 환경이 없거나 패키지가 없으면 조용히 스킵합니다.
    - 반드시 `import onnxruntime` 이전에 호출해야 합니다.
    """
    cu_ver = _detect_cuda_version()

    if cu_ver in _CUDA_LIB_TABLE:
        versions_to_try = [cu_ver]
    else:
        # 감지 실패 시 지원 목록 전체를 내림차순으로 시도
        versions_to_try = sorted(_CUDA_LIB_TABLE.keys(), reverse=True)

    for ver in versions_to_try:
        for mod_name, lib_name in _CUDA_LIB_TABLE[ver]:
            _try_load_from_module(mod_name, lib_name)

    _load_cudnn()


# ---------------------------------------------------------------------------
# 내부 헬퍼
# ---------------------------------------------------------------------------

def _detect_cuda_version() -> Optional[int]:
    """
    설치된 nvidia Python 패키지 또는 시스템 nvcc에서 CUDA major 버전을 감지합니다.

    탐지 순서:
      1. nvidia-cuda-runtime-cu?? 패키지 (cu12 → cu11 → cu13 순으로 탐색)
      2. 시스템 nvcc --version
      3. None (탐지 실패 — preload_cuda_libs()가 전체 버전을 순서대로 시도)
    """
    for cu_ver in (12, 11, 13):
        try:
            importlib.metadata.version(f"nvidia-cuda-runtime-cu{cu_ver}")
            return cu_ver
        except importlib.metadata.PackageNotFoundError:
            pass

    # fallback: nvcc
    try:
        result = subprocess.run(
            ["nvcc", "--version"],
            capture_output=True, text=True, timeout=3,
        )
        m = re.search(r"release\s+(\d+)\.", result.stdout)
        if m:
            return int(m.group(1))
    except Exception:
        pass

    return None


def _try_load_from_module(mod_name: str, lib_name: str) -> bool:
    """
    nvidia Python 패키지(mod_name)의 lib 디렉터리에서 lib_name을 RTLD_GLOBAL로 로드합니다.

    Returns:
        True  — 로드 성공
        False — 모듈 없음 / 파일 없음 / 로드 실패 (모두 조용히 스킵)
    """
    try:
        mod = importlib.import_module(mod_name)
        lib_dir = os.path.dirname(mod.__file__) if mod.__file__ else None
        if not lib_dir:
            return False
        lib_path = os.path.join(lib_dir, lib_name)
        if not os.path.exists(lib_path):
            return False
        ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
        return True
    except Exception:
        return False


def _load_cudnn() -> None:
    """
    cuDNN .so를 site-packages의 nvidia/cudnn/lib 아래에서 glob으로 찾아 로드합니다.

    nvidia.cudnn.lib.__file__이 None인 경우가 있어 glob 방식을 사용합니다.
    첫 번째 성공한 버전에서 중단합니다.
    """
    for sp in site.getsitepackages():
        for so_name in _CUDNN_SO_NAMES:
            for lib_path in glob.glob(
                os.path.join(sp, "nvidia", "cudnn", "lib", so_name)
            ):
                try:
                    ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
                    return
                except Exception:
                    pass
