import time
import numpy as np
from typing import Dict, Any

from dataloader.base import DataLoader
from runtimes.base import Runtime
from evaluators.base import Evaluator
from .model_spec import Task

class BenchmarkRunner:
    """
    DataLoader(데이터 공급) -> Runtime(추론 실행) -> Evaluator(결과 평가)
    전체 파이프라인을 일관되게 관리하는 오케스트레이터 클래스입니다.

    스트리밍 평가(Streaming Evaluation) 패턴을 채택합니다.
    배치마다 Evaluator.add_batch()를 호출하여 무거운 출력 텐서를 즉시 처리·폐기하고,
    루프 종료 후 Evaluator.compute()로 최종 메트릭을 산출합니다.
    이로써 수백만 샘플을 처리해도 RAM 사용량이 선형으로 폭발하지 않습니다.
    """
    def __init__(self, dataloader: DataLoader, runtime: Runtime, evaluator: Evaluator,
                 max_new_tokens: int = 256):
        self.dataloader = dataloader
        self.runtime = runtime
        self.evaluator = evaluator
        self._max_new_tokens = max_new_tokens

        # DataLoader의 공식 메타데이터 계약(Contract)을 통해 Fast-Path 여부 확인
        metadata = self.dataloader.get_metadata()
        self.is_static_batched = metadata.get("is_static_batched", False)
        # LLM 생성 중단 토큰 (없으면 None → max_new_tokens까지 생성)
        self._stop_token_ids = metadata.get("stop_token_ids", None)

    def _collate_batch(self, batch_list: Any) -> Dict[str, Any]:
        """
        List of single sample dicts -> Batched dict (np.stack)
        """
        if self.is_static_batched:
            # Fast-path: DataLoader 메타데이터 명세서 계약에 의해 이미 그룹화된 통짜 구조물이 넘어온 경우
            return batch_list

        collated = {}
        for key in batch_list[0].keys():
            if key == "input":
                # 입력 피처(Feature Tensor)만 Runtime 전달을 위해 Numpy Stack 수행
                first_input = batch_list[0][key]
                if isinstance(first_input, dict):
                    # Multi-input 딕셔너리 내부를 각각 병합
                    collated[key] = {
                        k: np.stack([item[key][k] for item in batch_list], axis=0)
                        for k in first_input.keys()
                    }
                else:
                    collated[key] = np.stack([item[key] for item in batch_list], axis=0)
            else:
                # 정답지(label) 및 메타 데이터는 차원 검사 없이 순수 List로 원형 보존 (SOLID 규칙)
                collated[key] = [item[key] for item in batch_list]
        return collated

    def _prepare_runtime_input(self, collated_input: Any, fallback_name: str) -> Dict[str, Any]:
        """로더가 던진 input이 딕셔너리(Multi-input)면 통째로 반환, 아니면 단일 노드(Single-input) 이름으로 래핑합니다."""
        if isinstance(collated_input, dict):
            return collated_input
        return {fallback_name: collated_input}

    def run(self, warmup_runs: int = 1, batch_size: int = 1, max_steps: int = None) -> Dict[str, Any]:
        """
        주입된 컴포넌트들을 연결하여 벤치마크 테스트 전체 루프를 수행합니다.

        스트리밍 평가 패턴:
          배치마다 evaluator.add_batch() 호출 → 출력 텐서 즉시 GC 반환
          루프 완료 후 evaluator.compute()로 최종 메트릭 산출

        Args:
            warmup_runs (int): 본 측정 전 Runtime 엔진을 예열하기 위한 횟수
            batch_size (int): 한 번에 묶어서 추론을 보낼 갯수
            max_steps (int): 옵션 - 지정된 횟수만큼만 루프를 돌고 탈출(테스트/리미터용)

        Returns:
            Dict[str, Any]: 최종 성능 종합 메트릭 리포트 (Evaluator.compute() 반환값)
        """
        print("[BenchmarkRunner] 🚀 Starts benchmarking...")

        # 모델 스펙에서 입력 노드 이름을 찾아 추출합니다. (단일 입력 가정)
        input_name = "input"
        if hasattr(self.runtime, 'compiled_model') and self.runtime.compiled_model is not None:
             input_name = list(self.runtime.compiled_model.spec.input_shapes.keys())[0]

        # 1. Warm-up
        if warmup_runs > 0:
            print(f"[BenchmarkRunner] 🌡️ Warming up {warmup_runs} times...")
            warmup_batch = self.dataloader.load_batch(batch_size)
            if warmup_batch:
                collated = self._collate_batch(warmup_batch)
                runtime_input = self._prepare_runtime_input(collated["input"], input_name)
                self.runtime.warmup(runtime_input, num_runs=warmup_runs)

        # 본 실행을 위해 DataLoader 순회를 첫 번째 샘플로 되돌립니다.
        self.dataloader.current_idx = 0

        # LLM 여부 감지: NLP_GENERATION 태스크 + generate() 실제 지원 런타임이면 생성 경로 사용
        _spec = getattr(getattr(self.runtime, 'compiled_model', None), 'spec', None)
        is_llm = (
            _spec is not None
            and _spec.task == Task.NLP_GENERATION
            and self.runtime.supports_generate()
        )
        if is_llm:
            print(f"[BenchmarkRunner] 🤖 LLM 감지 (NLP_GENERATION) — generate() 경로 사용 (max_new_tokens={self._max_new_tokens})")

        # 2. Streaming Inference Loop
        # ── 핵심 ─────────────────────────────────────────────────────────────────
        # 이전: all_outputs_list에 모든 배치 출력을 RAM에 쌓은 뒤 한 번에 평가 (OOM 위험)
        # 현재: 배치마다 evaluator.add_batch()로 경량 통계만 누산 → 출력 텐서 즉시 GC 반환
        # ─────────────────────────────────────────────────────────────────────────
        print("[BenchmarkRunner] ⚡ Running inference loop (streaming evaluation)...")
        batch_idx = 1
        while True:
            # 강제 종료 리미터 발동
            if max_steps is not None and batch_idx > max_steps:
                print(f"[BenchmarkRunner] 🛑 사용자가 요청한 리미터에 도달했습니다! ({max_steps} steps) - 즉각 탈출하여 결과를 채점합니다.")
                break

            batch = self.dataloader.load_batch(batch_size)
            if not batch:
                break

            collated = self._collate_batch(batch)
            runtime_input = self._prepare_runtime_input(collated["input"], input_name)

            # 단일 Batch 시간 정밀 측정
            if is_llm:
                gen_result = self.runtime.generate(
                    runtime_input, max_new_tokens=self._max_new_tokens,
                    stop_token_ids=self._stop_token_ids,
                )
                outputs = {"generated_ids": gen_result.generated_ids}
                # TTFT / TPOT / total_ms를 dict로 묶어 evaluator에 전달
                latency_ms = {
                    "total_ms": gen_result.total_ms,
                    "ttft_ms":  gen_result.ttft_ms,
                    "tpot_ms":  gen_result.tpot_ms,
                }
            else:
                start_time = time.perf_counter()
                outputs = self.runtime.run(runtime_input)
                end_time = time.perf_counter()
                latency_ms = (end_time - start_time) * 1000.0

            # 스트리밍 평가: Evaluator가 outputs에서 경량 통계만 추출 후 텐서 즉시 폐기
            self.evaluator.add_batch(outputs, collated["label"], latency_ms)

            if batch_idx % 10 == 0:
                if isinstance(collated["input"], dict):
                    first_key = next(iter(collated["input"]))
                    actual_batch_size = len(collated["input"][first_key])
                else:
                    actual_batch_size = len(collated["input"])
                if isinstance(latency_ms, dict):
                    latency_display = f"total={latency_ms.get('total_ms', 0):.2f} ms, ttft={latency_ms.get('ttft_ms', 0):.2f} ms, tpot={latency_ms.get('tpot_ms', 0):.2f} ms"
                else:
                    latency_display = f"{latency_ms:.2f} ms"
                print(f"  - Completed batch {batch_idx} ({actual_batch_size} samples), Latency: {latency_display}")
            batch_idx += 1

        # 3. 최종 메트릭 산출 (경량 누산 통계 → 최종 점수 계산)
        print("[BenchmarkRunner] 🏆 Computing final metrics...")
        metrics = self.evaluator.compute()
        return metrics
