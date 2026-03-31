import time
import numpy as np
from typing import Dict, Any, List

from .inference_result import InferenceResult
from ..dataloader.base import DataLoader
from ..runtimes.base import Runtime
from ..evaluators.base import Evaluator

class BenchmarkRunner:
    """
    DataLoader(데이터 공급) -> Runtime(추론 실행) -> Evaluator(결과 평가)
    전체 파이프라인을 일관되게 관리하는 오케스트레이터 클래스입니다.
    """
    def __init__(self, dataloader: DataLoader, runtime: Runtime, evaluator: Evaluator):
        self.dataloader = dataloader
        self.runtime = runtime
        self.evaluator = evaluator
        
        # DataLoader의 공식 메타데이터 계약(Contract)을 통해 Fast-Path 여부 확인
        metadata = self.dataloader.get_metadata()
        self.is_static_batched = metadata.get("is_static_batched", False)

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



    def _aggregate_results(self, all_outputs_list: List[Dict[str, Any]], all_labels_list: List[Any], timing_records: List[float]) -> InferenceResult:
        """
        루프 종료 후 수집된 추론 산출물을 최종 공통 DTO 규격으로 고속 병합합니다.
        정답지(Labels) 조립의 경우 차원에 대한 지식 없이 Evaluator에게 조립 권한을 위임합니다.
        """
        aggregated_outputs = {}
        if all_outputs_list:
            for out_key in all_outputs_list[0].keys():
                aggregated_outputs[out_key] = np.concatenate([out[out_key] for out in all_outputs_list], axis=0)

        # 안전하고 빠른 고속 라벨 배송 로직 (결합은 Evaluator가 자신의 도메인 규칙에 맞게 처리)
        return InferenceResult(
            outputs=aggregated_outputs,
            timing_records=timing_records,
            labels=all_labels_list
        )
    def run(self, warmup_runs: int = 1, batch_size: int = 1, max_steps: int = None) -> Dict[str, Any]:
        """
        주입된 컴포넌트들을 연결하여 벤치마크 테스트 전체 루프를 수행합니다.
        
        Args:
            warmup_runs (int): 본 측정 전 Runtime 엔진을 예열하기 위한 횟수
            batch_size (int): 한 번에 묶어서 추론을 보낼 갯수
            max_steps (int): 옵션 - 지정된 횟수만큼만 루프를 돌고 탈출(테스트/리미터용)
            
        Returns:
            Dict[str, Any]: 최종 성능 종합 메트릭 리포트 (Evaluator 반환값)
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

        # 2. Main Inference Loop
        timing_records = []
        all_outputs_list = []
        all_labels_list = []

        print("[BenchmarkRunner] ⚡ Running inference loop...")
        batch_idx = 1
        while True:
            # 강제 종료 리미터 발동
            if max_steps is not None and batch_idx > max_steps:
                print(f"[BenchmarkRunner] 🛑 사용자가 요청한 리미터에 도달했습니다! ({max_steps} steps) - 즉각 탈출하여 결과를 채점합니다.")
                break
                
            # 지정된 크기만큼 데이터 확보 (메모리 OOM 방지)
            batch = self.dataloader.load_batch(batch_size)
            if not batch:
                break
                
            collated = self._collate_batch(batch)
            
            runtime_input = self._prepare_runtime_input(collated["input"], input_name)
            
            # DataLoader가 던져준 원형(List, Tensor, Dict 등) 그대로 안전하게 적재
            all_labels_list.append(collated["label"])
            
            # 단일 Batch 시간 정밀 측정 시작
            start_time = time.perf_counter()
            outputs = self.runtime.run(runtime_input)
            end_time = time.perf_counter()
            
            # ms 단위 변환 저장
            latency_ms = (end_time - start_time) * 1000.0
            timing_records.append(latency_ms)
            all_outputs_list.append(outputs)
            
            if batch_idx % 10 == 0:
                # Multi-input 딕셔너리일 경우 텐서의 0번째 축(배치) 길이를 구해 정확한 샘플 수 계측
                if isinstance(collated["input"], dict):
                    first_key = next(iter(collated["input"]))
                    actual_batch_size = len(collated["input"][first_key])
                else:
                    actual_batch_size = len(collated["input"])
                    
                print(f"  - Completed batch {batch_idx} ({actual_batch_size} samples), Latency: {latency_ms:.2f} ms")
            batch_idx += 1
            
        print("[BenchmarkRunner] 📊 Aggregating results...")
        result_dto = self._aggregate_results(all_outputs_list, all_labels_list, timing_records)
        
        print("[BenchmarkRunner] 🏆 Evaluating metrics...")
        metrics = self.evaluator.evaluate(result_dto)
        return metrics
