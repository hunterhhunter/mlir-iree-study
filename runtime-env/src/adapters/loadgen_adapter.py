"""
LoadGen Adapter (스켈레톤)

mlperf_loadgen의 C++ LoadGen 엔진과 자체 프레임워크 컴포넌트(Runtime, DataLoader)를
연결하는 어댑터 계층입니다.

설치 방법:
    pip install mlperf-loadgen
    # 또는 소스 빌드:
    # git clone https://github.com/mlcommons/inference.git
    # cd inference/loadgen && pip install .

이 파일은 mlperf_loadgen이 설치되어 있지 않아도 임포트 에러 없이 존재할 수 있도록
조건부 import 구조를 사용합니다.
"""

import logging
from typing import Optional

log = logging.getLogger(__name__)

# mlperf_loadgen의 존재 여부를 런타임에 확인 (없어도 나머지 프레임워크는 정상 동작)
try:
    import mlperf_loadgen as lg
    _LOADGEN_AVAILABLE = True
except ImportError:
    lg = None
    _LOADGEN_AVAILABLE = False
    log.warning(
        "[LoadGenAdapter] mlperf_loadgen not found. "
        "Install it to enable official MLPerf measurement mode. "
        "BenchmarkRunner (basic mode) works without it."
    )


class LoadGenAdapter:
    """
    자체 Runtime/DataLoader와 MLPerf LoadGen C++ 엔진을 연결하는 어댑터.

    사용법 (mlperf_loadgen 설치 후):
        adapter = LoadGenAdapter(runtime=my_runtime, dataloader=my_loader)
        adapter.run(scenario="SingleStream")

    TODO (Phase 4 구현 시 채워야 할 항목):
        - issue_queries()  : SUT 콜백 — LoadGen이 쿼리를 던지면 Runtime.run()으로 위임
        - flush_queries()  : 남아 있는 비동기 쿼리 처리
        - load_query_samples() : QSL 콜백 — DataLoader.load_by_index()로 샘플 메모리 로드
        - unload_query_samples() : 메모리 해제
    """

    def __init__(self, runtime, dataloader):
        """
        Args:
            runtime    : Runtime ABC 구현체 (OnnxRuntime, IREERuntime 등)
            dataloader : DataLoader ABC 구현체 (load_by_index() 지원 필수)
        """
        if not _LOADGEN_AVAILABLE:
            raise RuntimeError(
                "mlperf_loadgen is required for LoadGenAdapter. "
                "Run: pip install mlperf-loadgen"
            )
        self.runtime    = runtime
        self.dataloader = dataloader

        # --- LoadGen SUT/QSL 객체 (Phase 4에서 초기화) ---
        self._sut: Optional[object] = None
        self._qsl: Optional[object] = None

    # ------------------------------------------------------------------
    # QSL (Query Sample Library) 콜백
    # ------------------------------------------------------------------

    def load_query_samples(self, sample_list):
        """
        LoadGen이 측정 전 호출. 지정된 인덱스의 샘플을 미리 메모리에 올립니다.
        DataLoader.load_by_index()를 활용하여 .npy 캐시를 통한 빠른 로딩이 가능합니다.

        TODO: sample_list 순회하여 self.dataloader.load_by_index(s) 결과를
              self._sample_cache[s]에 저장
        """
        raise NotImplementedError("Phase 4에서 구현 예정")

    def unload_query_samples(self, sample_list):
        """
        LoadGen이 측정 후 호출. 메모리 해제.

        TODO: self._sample_cache 비우기
        """
        raise NotImplementedError("Phase 4에서 구현 예정")

    # ------------------------------------------------------------------
    # SUT (System Under Test) 콜백
    # ------------------------------------------------------------------

    def issue_queries(self, query_samples):
        """
        LoadGen이 쿼리를 발행할 때 호출 (SingleStream: 1개씩, Offline: 전체 한번에).
        Runtime.run()으로 추론을 위임하고 결과를 LoadGen에 반환합니다.

        TODO:
            for qs in query_samples:
                data   = self._sample_cache[qs.index]["input"]
                output = self.runtime.run({input_name: data[np.newaxis, ...]})
                ...
                lg.QuerySamplesComplete([lg.QuerySampleResponse(qs.id, ...)])
        """
        raise NotImplementedError("Phase 4에서 구현 예정")

    def flush_queries(self):
        """비동기 모드에서 미완료 쿼리 처리. SingleStream에서는 no-op."""
        pass

    # ------------------------------------------------------------------
    # 메인 실행 진입점
    # ------------------------------------------------------------------

    def run(self, scenario: str = "SingleStream", count: int = 500):
        """
        지정된 MLPerf 시나리오로 측정을 수행합니다.

        Args:
            scenario : "SingleStream" | "Offline" | "Server" | "MultiStream"
            count    : 측정에 사용할 샘플 수

        TODO: Phase 4에서 lg.TestSettings, lg.StartTest() 호출로 구현
        """
        raise NotImplementedError("Phase 4에서 구현 예정")
