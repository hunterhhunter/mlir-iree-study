"""
vLLM Runtime 백엔드

PagedAttention 기반의 vLLM 엔진을 BenchmarkRunner 인터페이스로 래핑합니다.
ONNX 기반 OnnxRuntime과 달리, 이 런타임은 HuggingFace 모델 가중치 디렉토리를
CompiledModel.artifact_path로 받아 vLLM LLM 엔진을 초기화합니다.

오프라인 배치 추론(LLM.generate) 방식을 사용하므로 per-token 타이밍은
근사값으로 제공됩니다. 정확한 TTFT가 필요하면 vLLM AsyncLLMEngine을 사용하세요.
"""

import time
from typing import Any, Dict, List, Optional, Union

import numpy as np

from core.compiled_model import CompiledModel
from .base import Runtime
from core.generation_result import GenerationResult


class VllmRuntime(Runtime):
    """
    vLLM 오프라인 배치 추론 엔진 래퍼.

    CompiledModel.artifact_path는 HuggingFace 모델 가중치 디렉토리를 가리켜야 합니다.
    (예: Path("models/llama-3.1-8b/"))

    런타임 옵션 (create_runtime(...) 또는 생성자 kwargs):
        tensor_parallel_size (int):  텐서 병렬 GPU 수 (기본값: 1)
        gpu_memory_utilization (float): GPU 메모리 사용률 (기본값: 0.90)
        max_model_len (int | None):  최대 컨텍스트 길이 (기본값: None = 모델 기본값)
        dtype (str):                 가중치 dtype ('auto', 'float16', 'bfloat16' 등)
    """

    def __init__(self, **runtime_options):
        self.device = runtime_options.get("device", "cuda")
        self.tensor_parallel_size = runtime_options.get("tensor_parallel_size", 1)
        self.gpu_memory_utilization = runtime_options.get("gpu_memory_utilization", 0.90)
        self.max_model_len = runtime_options.get("max_model_len", None)
        self.dtype = runtime_options.get("dtype", "auto")

        self._llm = None
        self._model_path: str = ""
        self.compiled_model: CompiledModel | None = None

    # ------------------------------------------------------------------
    # Runtime 인터페이스 구현
    # ------------------------------------------------------------------

    def load(self, compiled_model: CompiledModel) -> None:
        """
        HuggingFace 모델 디렉토리에서 vLLM LLM 엔진을 초기화합니다.

        vLLM은 import 시점에 CUDA 컨텍스트를 구성하므로,
        load()가 처음 호출될 때 CUDA graph capture가 수행됩니다.
        """
        try:
            from vllm import LLM
        except ImportError:
            raise ImportError(
                "vllm 패키지가 설치되어 있지 않습니다. "
                "pip install vllm 으로 설치하세요."
            )

        self.compiled_model = compiled_model
        self._model_path = str(compiled_model.artifact_path)

        print(f"[VllmRuntime] Loading model from: {self._model_path}")
        print(f"[VllmRuntime] tensor_parallel={self.tensor_parallel_size}, "
              f"gpu_memory={self.gpu_memory_utilization}, dtype={self.dtype}")

        llm_kwargs: Dict[str, Any] = dict(
            model=self._model_path,
            tensor_parallel_size=self.tensor_parallel_size,
            max_model_len=self.max_model_len,
            dtype=self.dtype,
        )
        if self.device == "cpu":
            # CPU 모드: CUDA graph 캡처 불가, gpu_memory_utilization 비적용
            llm_kwargs["device"] = "cpu"
        else:
            llm_kwargs["gpu_memory_utilization"] = self.gpu_memory_utilization

        self._llm = LLM(**llm_kwargs)
        print("[VllmRuntime] vLLM engine ready.")

    def run(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        단일 forward pass 인터페이스.

        vLLM은 자기회귀 생성에 최적화되어 있어 단일 forward pass를 직접 지원하지 않습니다.
        max_new_tokens=1로 generate()를 호출하여 next-token 예측 결과를 반환합니다.
        NLP_GENERATION 태스크는 generate()를 통해 평가하는 것을 권장합니다.
        """
        gen_result = self.generate(inputs, max_new_tokens=1)
        return {"generated_ids": gen_result.generated_ids}

    def supports_generate(self) -> bool:
        return True

    def generate(
        self,
        inputs: Dict[str, np.ndarray],
        max_new_tokens: int = 256,
        stop_token_ids: Union[List[int], None] = None,
    ) -> GenerationResult:
        """
        vLLM 오프라인 배치 추론을 통한 자기회귀 생성 (Greedy Decoding).

        타이밍 측정:
            vLLM 오프라인 모드(LLM.generate)는 per-token 타이밍을 노출하지 않습니다.
            total_ms  : 전체 generate() 호출 wall-time
            ttft_ms   : total_ms를 토큰 수로 나눈 근사값 (실제 TTFT보다 작을 수 있음)
            tpot_ms   : ttft_ms와 동일한 근사값

        Args:
            inputs:         'input_ids' (1, padded_len), 'attention_mask' (1, padded_len) 포함 dict
            max_new_tokens: 최대 생성 토큰 수
            stop_token_ids: 조기 종료 토큰 ID 리스트 (예: [eos_id])

        Returns:
            GenerationResult: generated_ids, ttft_ms, tpot_ms, total_ms, num_tokens
        """
        try:
            from vllm import SamplingParams
        except ImportError:
            raise ImportError("vllm 패키지가 설치되어 있지 않습니다.")

        if self._llm is None:
            raise RuntimeError("vLLM engine is not loaded. Call load() first.")

        # stop_token_ids 정규화: None이 아니면 그대로 사용
        if stop_token_ids is not None and not isinstance(stop_token_ids, list):
            stop_token_ids = [stop_token_ids]

        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=0.0,            # Greedy decoding
            stop_token_ids=stop_token_ids,
        )

        # input_ids 전처리: (1, seq_len) → [token_id, ...]
        input_ids_np = inputs["input_ids"]
        attention_mask_np = inputs.get("attention_mask", None)

        if input_ids_np.ndim == 2:
            input_ids_np = input_ids_np[0]  # (seq_len,)

        # attention_mask로 패딩 제거하여 실제 프롬프트만 추출
        if attention_mask_np is not None:
            if attention_mask_np.ndim == 2:
                attention_mask_np = attention_mask_np[0]
            prompt_len = int(attention_mask_np.sum())
            prompt_token_ids = input_ids_np[:prompt_len].tolist()
        else:
            prompt_token_ids = input_ids_np.tolist()

        # 추론 실행 및 전체 시간 측정
        # vLLM v0.5+ API: prompt_token_ids는 딕셔너리 입력으로 전달
        t_start = time.perf_counter()
        outputs = self._llm.generate(
            {"prompt_token_ids": prompt_token_ids},
            sampling_params=sampling_params,
        )
        t_end = time.perf_counter()
        total_ms = (t_end - t_start) * 1000.0

        # 생성된 토큰 추출
        generated_token_ids: List[int] = list(outputs[0].outputs[0].token_ids)
        num_tokens = len(generated_token_ids)

        # 오프라인 모드: per-token 타이밍 불가 → 균등 분배 근사
        avg_ms = total_ms / max(num_tokens, 1)
        ttft_ms = avg_ms
        tpot_ms = avg_ms

        return GenerationResult(
            generated_ids=np.array(generated_token_ids, dtype=np.int64),
            ttft_ms=ttft_ms,
            tpot_ms=tpot_ms,
            total_ms=total_ms,
            num_tokens=num_tokens,
        )

    def warmup(self, inputs: Dict[str, np.ndarray], num_runs: int = 1) -> None:
        """
        vLLM은 load() 시 CUDA graph capture로 자동 웜업을 수행합니다.
        추가적인 명시적 웜업이 필요한 경우에만 사용합니다.
        """
        print("[VllmRuntime] vLLM auto-warmed up via CUDA graph capture during load().")
        if num_runs > 0:
            print(f"[VllmRuntime] Running {num_runs} additional manual warmup pass(es)...")
            for _ in range(num_runs):
                self.generate(inputs, max_new_tokens=4)

    def unload(self) -> None:
        """vLLM 엔진 및 GPU 메모리 해제."""
        if self._llm is not None:
            del self._llm
            self._llm = None
        self._model_path = ""
        self.compiled_model = None

        try:
            import torch
            torch.cuda.empty_cache()
        except ImportError:
            pass

    def get_device_spec(self) -> Dict[str, Any]:
        return {
            "backend": "vllm",
            "device": self.device,
            "tensor_parallel_size": self.tensor_parallel_size,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "dtype": self.dtype,
            "model_path": self._model_path,
        }

    def is_compatible(self, compiled_model: CompiledModel) -> bool:
        """
        vLLM은 HuggingFace 모델 디렉토리 (config.json 포함) 또는
        backend_name에 'vllm'이 포함된 CompiledModel과 호환됩니다.
        """
        path = compiled_model.artifact_path
        backend_match = "vllm" in compiled_model.backend_name.lower()
        is_hf_dir = path.is_dir() and (path / "config.json").exists()
        return backend_match or is_hf_dir
