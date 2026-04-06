"""
ONNX Runtime CUDA 실행 통합 테스트

테스트 범위:
  1. CUDAExecutionProvider 가용 여부 검증
  2. CUDA device로 OnnxRuntime 초기화
  3. ResNet50 모델 로드 및 CUDA 추론 실행
  4. CPU/CUDA 추론 결과 일치 검증 (숫자 오차 허용)
  5. VRAM 해제 (unload) 검증

실행 방법:
  python -m pytest tests/test_onnx_cuda.py -v
"""

import os
import sys
import numpy as np
import pytest
from pathlib import Path

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from core.model_spec import Model_Spec, Task
from core.compiled_model import CompiledModel
from runtimes import OnnxRuntime
from utils.cuda_preload import preload_cuda_libs, _detect_cuda_version

# ---------------------------------------------------------------------------
# 테스트 픽스처
# ---------------------------------------------------------------------------

RESNET50_ONNX = os.path.join(project_root, "models/resnet-50/resnet50.onnx")
INPUT_NAME = "pixel_values"   # Hugging Face export 기본 이름


def _detect_input_name(onnx_path: str) -> str:
    """ONNX 그래프에서 실제 입력 텐서 이름을 추출합니다."""
    import onnx
    model = onnx.load(onnx_path)
    return model.graph.input[0].name


def _make_spec(onnx_path: str) -> Model_Spec:
    input_name = _detect_input_name(onnx_path)
    return Model_Spec(
        name="resnet50",
        task=Task.IMAGE_CLASSIFICATION,
        input_shapes={input_name: (1, 3, 224, 224)},
        input_dtype={input_name: "float32"},
        output_shapes={"logits": (1, 1000)},
        model_paths={"onnx": onnx_path},
    )


def _make_compiled(spec: Model_Spec, onnx_path: str) -> CompiledModel:
    return CompiledModel(
        spec=spec,
        backend_name="onnxruntime",
        artifact_path=Path(onnx_path),
    )


def _random_input(spec: Model_Spec) -> dict:
    """스펙에 정의된 입력 형태로 랜덤 numpy 배열을 생성합니다."""
    return {
        name: np.random.randn(*shape).astype(np.float32)
        for name, shape in spec.input_shapes.items()
    }


# ---------------------------------------------------------------------------
# 조건부 skip 마커
# ---------------------------------------------------------------------------

def _is_cuda_available() -> bool:
    """_preload_cuda_libs() 수행 후 CUDAExecutionProvider 가용 여부를 반환합니다."""
    try:
        import onnxruntime as ort
        return "CUDAExecutionProvider" in ort.get_available_providers()
    except Exception:
        return False


requires_cuda = pytest.mark.skipif(
    not _is_cuda_available(),
    reason="CUDAExecutionProvider를 사용할 수 없는 환경입니다.",
)

requires_model = pytest.mark.skipif(
    not os.path.exists(RESNET50_ONNX),
    reason=f"모델 파일이 없습니다: {RESNET50_ONNX}",
)


# ---------------------------------------------------------------------------
# 테스트 케이스
# ---------------------------------------------------------------------------

class TestCudaPreloadUtil:
    """src.utils.cuda_preload 유틸리티 단위 테스트"""

    def test_preload_does_not_raise(self):
        """preload_cuda_libs()는 CUDA 환경 유무와 무관하게 예외를 던지지 않아야 합니다."""
        preload_cuda_libs()

    def test_detect_cuda_version_returns_int_or_none(self):
        """_detect_cuda_version()은 int 또는 None을 반환해야 합니다."""
        ver = _detect_cuda_version()
        assert ver is None or isinstance(ver, int), f"예상치 못한 반환 타입: {type(ver)}"

    def test_detect_cuda_version_matches_installed_packages(self):
        """nvidia-cuda-runtime-cu?? 패키지가 있으면 해당 버전이 감지되어야 합니다."""
        import importlib.metadata
        detected = _detect_cuda_version()
        if detected is None:
            pytest.skip("CUDA 패키지가 설치되어 있지 않습니다.")
        pkg_name = f"nvidia-cuda-runtime-cu{detected}"
        try:
            importlib.metadata.version(pkg_name)
        except importlib.metadata.PackageNotFoundError:
            pytest.fail(f"감지된 버전 cu{detected}에 해당하는 패키지 '{pkg_name}'가 없습니다.")


class TestCudaProvider:
    """CUDAExecutionProvider 가용성 검증"""

    def test_cuda_libs_preload_does_not_raise(self):
        """preload_cuda_libs()는 CUDA 환경 유무와 무관하게 예외를 던지지 않아야 합니다."""
        preload_cuda_libs()  # should not raise

    def test_cuda_provider_available(self):
        """CUDAExecutionProvider가 ort.get_available_providers()에 포함되어야 합니다."""
        import onnxruntime as ort
        available = ort.get_available_providers()
        print(f"\n  가용 Provider 목록: {available}")
        assert "CUDAExecutionProvider" in available, (
            f"CUDAExecutionProvider가 없습니다. 현재 목록: {available}\n"
            "onnxruntime-gpu 설치 및 CUDA 환경을 확인하세요."
        )


class TestOnnxRuntimeCudaInit:
    """OnnxRuntime CUDA 초기화 테스트"""

    @requires_cuda
    def test_cuda_runtime_creation(self):
        """device='cuda'로 OnnxRuntime을 생성하면 예외 없이 초기화되어야 합니다."""
        rt = OnnxRuntime(device="cuda")
        assert rt.device == "cuda"
        assert "CUDAExecutionProvider" in rt.providers

    @requires_cuda
    def test_device_spec_reflects_cuda(self):
        """get_device_spec()이 cuda 디바이스와 provider를 올바르게 반환해야 합니다."""
        rt = OnnxRuntime(device="cuda")
        spec = rt.get_device_spec()
        assert spec["device"] == "cuda"
        assert "CUDAExecutionProvider" in spec["active_providers"]

    def test_cpu_runtime_unaffected(self):
        """CUDA 라이브러리 사전 로드 이후에도 CPU 런타임은 정상 동작해야 합니다."""
        rt = OnnxRuntime(device="cpu")
        assert rt.device == "cpu"
        assert rt.providers == ["CPUExecutionProvider"]

    @requires_cuda
    def test_invalid_device_raises(self):
        """지원하지 않는 device 이름은 ValueError를 발생시켜야 합니다."""
        with pytest.raises((ValueError, RuntimeError)):
            OnnxRuntime(device="tpu")


class TestOnnxRuntimeCudaInference:
    """CUDA 디바이스에서 실제 모델 추론 테스트"""

    @requires_cuda
    @requires_model
    def test_load_model_on_cuda(self):
        """ResNet50 ONNX 모델이 CUDA 위에서 정상 로드되어야 합니다."""
        spec = _make_spec(RESNET50_ONNX)
        compiled = _make_compiled(spec, RESNET50_ONNX)

        rt = OnnxRuntime(device="cuda")
        rt.load(compiled)

        assert rt.session is not None
        assert len(rt.input_names) > 0
        assert len(rt.output_names) > 0
        rt.unload()

    @requires_cuda
    @requires_model
    def test_inference_returns_correct_shape(self):
        """CUDA 추론 결과가 (1, 1000) 형태의 numpy 배열이어야 합니다."""
        spec = _make_spec(RESNET50_ONNX)
        compiled = _make_compiled(spec, RESNET50_ONNX)
        inputs = _random_input(spec)

        rt = OnnxRuntime(device="cuda")
        rt.load(compiled)
        outputs = rt.run(inputs)
        rt.unload()

        assert isinstance(outputs, dict)
        output_tensor = next(iter(outputs.values()))
        assert isinstance(output_tensor, np.ndarray)
        assert output_tensor.ndim == 2
        assert output_tensor.shape[1] == 1000, (
            f"출력 클래스 수가 1000이어야 하는데 {output_tensor.shape[1]}입니다."
        )

    @requires_cuda
    @requires_model
    def test_cuda_cpu_outputs_match(self):
        """동일한 입력에 대해 CUDA와 CPU 추론 결과가 근사적으로 일치해야 합니다."""
        spec = _make_spec(RESNET50_ONNX)
        compiled = _make_compiled(spec, RESNET50_ONNX)

        np.random.seed(42)
        inputs = _random_input(spec)

        rt_cpu = OnnxRuntime(device="cpu")
        rt_cpu.load(compiled)
        cpu_out = rt_cpu.run(inputs)
        rt_cpu.unload()

        rt_cuda = OnnxRuntime(device="cuda")
        rt_cuda.load(compiled)
        cuda_out = rt_cuda.run(inputs)
        rt_cuda.unload()

        # GPU와 CPU는 float32 연산 순서 차이로 최대 ~1% 수치 오차가 발생할 수 있습니다.
        # top-1 예측 결과의 일치 여부는 별도 assert로 검증합니다.
        for key in cpu_out:
            np.testing.assert_allclose(
                cpu_out[key], cuda_out[key],
                rtol=1e-2, atol=1e-2,
                err_msg=f"CPU/CUDA 출력 '{key}' 불일치 (허용 오차 초과)"
            )

        # top-1 예측 클래스는 동일해야 합니다
        for key in cpu_out:
            cpu_top1 = int(np.argmax(cpu_out[key], axis=1)[0])
            cuda_top1 = int(np.argmax(cuda_out[key], axis=1)[0])
            assert cpu_top1 == cuda_top1, (
                f"top-1 예측 불일치: CPU={cpu_top1}, CUDA={cuda_top1}"
            )

    @requires_cuda
    @requires_model
    def test_warmup_runs_without_error(self):
        """warmup() 호출이 예외 없이 완료되어야 합니다."""
        spec = _make_spec(RESNET50_ONNX)
        compiled = _make_compiled(spec, RESNET50_ONNX)
        inputs = _random_input(spec)

        rt = OnnxRuntime(device="cuda")
        rt.load(compiled)
        rt.warmup(inputs, num_runs=2)
        rt.unload()

    @requires_cuda
    @requires_model
    def test_unload_releases_session(self):
        """unload() 후 session이 None이 되고 run() 호출 시 RuntimeError가 발생해야 합니다."""
        spec = _make_spec(RESNET50_ONNX)
        compiled = _make_compiled(spec, RESNET50_ONNX)
        inputs = _random_input(spec)

        rt = OnnxRuntime(device="cuda")
        rt.load(compiled)
        rt.unload()

        assert rt.session is None
        with pytest.raises(RuntimeError, match="load()"):
            rt.run(inputs)

    @requires_cuda
    @requires_model
    def test_argmax_top1_is_valid_class_index(self):
        """CUDA 추론 결과의 top-1 예측 인덱스가 0~999 범위 내여야 합니다."""
        spec = _make_spec(RESNET50_ONNX)
        compiled = _make_compiled(spec, RESNET50_ONNX)
        inputs = _random_input(spec)

        rt = OnnxRuntime(device="cuda")
        rt.load(compiled)
        outputs = rt.run(inputs)
        rt.unload()

        logits = next(iter(outputs.values()))  # (1, 1000)
        top1 = int(np.argmax(logits, axis=1)[0])
        assert 0 <= top1 < 1000, f"top-1 인덱스 {top1}가 유효 범위(0~999)를 벗어났습니다."
