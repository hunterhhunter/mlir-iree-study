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

    def _collate_batch(self, batch_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        List of single sample dicts -> Batched dict (np.stack)
        """
        collated = {}
        for key in batch_list[0].keys():
            if key == "input":
                collated[key] = np.stack([item[key] for item in batch_list], axis=0)
            elif key == "label":
                # 파이썬 안티 패턴(try-except 제어 흐름) 제거
                # 모든 정답지의 형태(Shape)가 일치하면 Numpy 압축, 하나라도 다르면 리스트 유지
                shapes = set(np.array(item[key]).shape for item in batch_list)
                if len(shapes) == 1:
                    collated[key] = np.array([item[key] for item in batch_list])
                else:
                    collated[key] = [item[key] for item in batch_list]
            else:
                collated[key] = [item[key] for item in batch_list]
        return collated

    def run(self, warmup_runs: int = 1, batch_size: int = 1) -> Dict[str, Any]:
        """
        주입된 컴포넌트들을 연결하여 벤치마크 테스트 전체 루프를 수행합니다.
        
        Args:
            warmup_runs (int): 본 측정 전 Runtime 엔진을 예열하기 위한 횟수
            batch_size (int): 한 번에 묶어서 추론을 보낼 갯수
            
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
                
                # DataLoader의 기본 키인 "input"을 모델의 실제 입력값 이름표로 매핑
                runtime_input = {input_name: collated["input"]}
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
            # 지정된 크기만큼 데이터 확보 (메모리 OOM 방지)
            batch = self.dataloader.load_batch(batch_size)
            if not batch:
                break
                
            collated = self._collate_batch(batch)
            runtime_input = {input_name: collated["input"]}
            all_labels_list.extend(collated["label"])
            
            # 단일 Batch 시간 정밀 측정 시작
            start_time = time.perf_counter()
            outputs = self.runtime.run(runtime_input)
            end_time = time.perf_counter()
            
            # ms 단위 변환 저장
            latency_ms = (end_time - start_time) * 1000.0
            timing_records.append(latency_ms)
            all_outputs_list.append(outputs)
            
            if batch_idx % 10 == 0:
                print(f"  - Completed batch {batch_idx} ({len(collated['input'])} samples), Latency: {latency_ms:.2f} ms")
            batch_idx += 1
            
        print("[BenchmarkRunner] 📊 Aggregating results...")
        
        # List 형식을 Evaluator가 선호하는 통짜 Numpy Array 딕셔너리로 병합
        aggregated_outputs = {}
        if all_outputs_list:
            for out_key in all_outputs_list[0].keys():
                aggregated_outputs[out_key] = np.concatenate([out[out_key] for out in all_outputs_list], axis=0)
                
        # 리스트 요소들의 모양이 모두 같으면 배열로, 다르면 리스트로 유지
        shapes = set(np.array(lbl).shape for lbl in all_labels_list)
        if len(shapes) == 1:
            aggregated_labels = np.array(all_labels_list)
        else:
            aggregated_labels = all_labels_list
        
        # 공통 DTO 규격으로 래핑
        result_dto = InferenceResult(
            outputs=aggregated_outputs,
            timing_records=timing_records,
            labels=aggregated_labels
        )
        
        print("[BenchmarkRunner] 🏆 Evaluating metrics...")
        metrics = self.evaluator.evaluate(result_dto)
        return metrics
