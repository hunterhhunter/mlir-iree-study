import numpy as np
from typing import Dict, Any

from ..utils.cuda_preload import preload_cuda_libs

# onnxruntime import 전에 CUDA 라이브러리를 사전 로드합니다.
# 자세한 내용은 src/utils/cuda_preload.py 참조.
preload_cuda_libs()

import onnxruntime as ort

import time
from .base import Runtime, GenerationResult
from ..core.compiled_model import CompiledModel

class OnnxRuntime(Runtime):
    """
    ONNX Runtime 기반의 실행 엔진 래퍼.
    """
    def __init__(self, **runtime_options):
        """
        [1. Hardware Provisioning & Context Initialization]
        """
        # 실행 디바이스 환경 변수 받기 (기본값 cpu)
        self.device = runtime_options.get("device", "cpu")
        
        # ONNX Runtime의 Execution Provider 설정
        _SUPPORTED_DEVICES = {"cpu", "cuda"}
        if self.device not in _SUPPORTED_DEVICES:
            raise ValueError(
                f"지원하지 않는 device입니다: '{self.device}'. "
                f"지원 목록: {sorted(_SUPPORTED_DEVICES)}"
            )

        if self.device == "cuda":
            available = ort.get_available_providers()
            if "CUDAExecutionProvider" not in available:
                raise RuntimeError(
                    "CUDAExecutionProvider를 사용할 수 없습니다. "
                    "onnxruntime-gpu 설치 여부와 CUDA 환경을 확인하세요. "
                    f"현재 가용 Provider: {available}"
                )
            self.providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            self.providers = ['CPUExecutionProvider']
            
        # 런타임의 상태 변수들 초기화
        self.session = None
        self.input_names = []
        self.output_names = []
        self.compiled_model = None

    def load(self, compiled_model: CompiledModel) -> None:
        """
        [2. Artifact Deserialization & Memory Mapping]
        .onnx 파일을 computation graph로 로드.
        """
        if not self.is_compatible(compiled_model):
            raise ValueError(f"Incompatible backend: {compiled_model.backend_name}")
            
        self.compiled_model = compiled_model
        
        # ONNX Runtime은 내부적으로 mmap 최적화 및 직렬화 해제를 자체 지원.
        self.session = ort.InferenceSession(
            str(self.compiled_model.artifact_path), 
            providers=self.providers
        )
        
        # 모델이 요구하는 입출력 텐서의 이름표를 추출.
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]

    def run(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        [4. Kernel Dispatch & Forward Pass (Inference)]
        순수 Numpy 배열을 던져주고 결과도 Numpy로 추출.
        """
        if self.session is None:
            raise RuntimeError("ONNX Runtime session is not loaded. Call load() first.")
            
        # 입력된 텐서들 중 모델이 실제로 필요로 하는 이름표만 매핑해서 넣음
        ort_inputs = {name: inputs[name] for name in self.input_names if name in inputs}
        
        # 실제 하드웨어 커널에 연산 지시
        results = self.session.run(self.output_names, ort_inputs)
        
        # 결과 리스트를 이름표와 함께 묶어 Dict 타입의 Numpy로 반환
        return {out_name: np.array(res) for out_name, res in zip(self.output_names, results)}

    def warmup(self, inputs: Dict[str, np.ndarray], num_runs: int = 1) -> None:
        """
        [3. JIT Triggering & Cache Warming]
        실제 측정 전, Cold-start 지연 시간을 제거.
        """
        print(f"[ONNX Runtime] Warming up {num_runs} times on {self.device}...")
        # LLM 패딩 trim: NLP_GENERATION 태스크에서만 실제 토큰 길이로 슬라이싱
        # BERT 등 고정 seq_len 모델에 적용하면 shape mismatch가 발생하므로 반드시 분기
        from ..core.model_spec import Task
        warmup_inputs = dict(inputs)
        is_llm = (
            self.compiled_model is not None
            and self.compiled_model.spec.task == Task.NLP_GENERATION
        )
        if is_llm and "attention_mask" in inputs and "input_ids" in inputs:
            real_len = int(inputs["attention_mask"].sum())
            total_len = inputs["input_ids"].shape[-1]
            if real_len < total_len:
                warmup_inputs = {
                    k: v[:, :real_len] if v.ndim == 2 else v
                    for k, v in inputs.items()
                }
        for _ in range(num_runs):
            self.run(warmup_inputs)

    def unload(self) -> None:
        """
        [5. Resource Deallocation & Teardown]
        메모리 누수 및 다른 모델 테스트 시 발생할 수 있는 VRAM OOM 에러를 방지.
        """
        self.session = None
        self.input_names = []
        self.output_names = []
        self.compiled_model = None

    def get_device_spec(self) -> Dict[str, Any]:
        """현재 런타임이 구동 중인 하드웨어 명세를 반환."""
        return {
            "backend": "onnxruntime", 
            "device": self.device, 
            "active_providers": self.providers
        }

    def supports_generate(self) -> bool:
        return True

    def generate(self, inputs: Dict[str, np.ndarray], max_new_tokens: int = 256,
                 stop_token_ids=None) -> GenerationResult:
        """
        Greedy autoregressive 생성 루프.
        각 스텝에서 전체 시퀀스를 ONNX 모델에 넣고 마지막 위치의 logits에서
        argmax로 다음 토큰을 결정합니다 (KV 캐시 없는 단순 구현).

        stop_token_ids: int 또는 List[int] — 해당 토큰 생성 시 즉시 중단.
                        EOS, 줄바꿈(\n) 등을 포함해 과잉 생성을 방지합니다.
        timing:
            ttft_ms: 첫 번째 토큰 생성에 걸린 시간 (첫 forward pass)
            tpot_ms: 이후 토큰들의 평균 생성 시간
            total_ms: 전체 생성 시간
        """
        # stop_token_ids를 set으로 정규화
        if stop_token_ids is None:
            _stop_ids = set()
        elif isinstance(stop_token_ids, int):
            _stop_ids = {stop_token_ids}
        else:
            _stop_ids = set(stop_token_ids)
        input_ids = inputs["input_ids"].copy()          # (1, prompt_len)
        attention_mask = inputs["attention_mask"].copy() # (1, prompt_len)

        # 패딩 제거: attention_mask == 1인 실제 토큰만 슬라이싱합니다.
        # padding="max_length"로 4096까지 패딩된 경우 logits가 (1,4096,128256)×4B = 2.1GB
        # 실제 토큰만 넘기면 예: (1,82,128256)×4B = 42MB 로 줄어듭니다.
        real_len = int(attention_mask.sum())
        if real_len < input_ids.shape[1]:
            input_ids = input_ids[:, :real_len]
            attention_mask = attention_mask[:, :real_len]

        generated_ids = []
        token_times = []
        ttft_ms = 0.0
        total_start = time.perf_counter()

        for step in range(max_new_tokens):
            # attention_mask 누적합으로 position_ids 재계산
            position_ids = np.maximum(
                np.cumsum(attention_mask, axis=-1) - 1, 0
            ).astype(np.int64)

            t0 = time.perf_counter()
            outputs = self.run({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            })
            elapsed = (time.perf_counter() - t0) * 1000.0
            token_times.append(elapsed)
            if step == 0:
                ttft_ms = elapsed

            # 마지막 위치의 logits에서 greedy decoding
            logits = outputs[self.output_names[0]]  # (1, seq_len, vocab_size)
            next_token = int(np.argmax(logits[0, -1, :]))

            if next_token in _stop_ids:
                break

            generated_ids.append(next_token)

            # 다음 스텝을 위해 시퀀스 확장
            input_ids = np.concatenate(
                [input_ids, np.array([[next_token]], dtype=np.int64)], axis=1
            )
            attention_mask = np.concatenate(
                [attention_mask, np.ones((1, 1), dtype=np.int64)], axis=1
            )

        total_ms = (time.perf_counter() - total_start) * 1000.0
        tpot_ms = float(np.mean(token_times[1:])) if len(token_times) > 1 else 0.0

        return GenerationResult(
            generated_ids=np.array(generated_ids, dtype=np.int64),
            ttft_ms=ttft_ms,
            tpot_ms=tpot_ms,
            total_ms=total_ms,
            num_tokens=len(generated_ids),
        )

    def is_compatible(self, compiled_model: CompiledModel) -> bool:
        """이 런타임이 실행할 수 있는 '.onnx' 확장자 모델이 맞는지 검사함."""
        backend_match = compiled_model.backend_name.startswith("onnx")
        extension_match = str(compiled_model.artifact_path).endswith(".onnx")
        return backend_match or extension_match
