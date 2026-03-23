import os
import sys
import subprocess
from pathlib import Path
import onnx
import iree.compiler.tools as ireec

from .base import Compiler
from ..core.model_spec import Model_Spec
from ..core.compiled_model import CompiledModel

class IREECompiler(Compiler):
    def __init__(self, **compile_options):
        super().__init__(**compile_options)
        # 타겟 백엔드 설정 (기본값: llvm-cpu)
        self.target_backend = self.compile_options.get("target_backend", "llvm-cpu")
        self.iree_cuda_target = self.compile_options.get("iree_cuda_target", "sm_80")
        
        # 현재 Python 가상환경 내의 bin 폴더 경로 추론
        # sys.executable 은 보통 .venv/bin/python 을 가리킵니다.
        self.venv_bin = os.path.dirname(sys.executable)
        self.import_onnx_bin = os.path.join(self.venv_bin, "iree-import-onnx")
        
        if not os.path.exists(self.import_onnx_bin):
            print(f"[WARNING] IREE import tool not found at {self.import_onnx_bin}. Please ensure iree-tools-tf is installed.")

    def get_artifact_name(self, model_spec: Model_Spec) -> str:
        backend_str = "cuda" if "cuda" in self.target_backend else "cpu"
        return f"{model_spec.name}_iree_{backend_str}.vmfb"

    def _convert_to_opset_17(self, source_onnx_path: str, v17_onnx_path: str) -> str:
        """
        IREE MLIR Importer는 최신 Opset을 선호하므로 Opset 17로 버전 변환을 수행합니다.
        """
        print(f"[IREE Compiler] Converting ONNX model to Opset 17...")
        original_model = onnx.load(source_onnx_path)
        try:
            converted_model = onnx.version_converter.convert_version(original_model, 17)
            onnx.save(converted_model, v17_onnx_path)
            print(f"[IREE Compiler] Saved Opset 17 ONNX to {v17_onnx_path}")
        except Exception as e:
            print(f"[IREE Compiler] Opset conversion failed: {e}. Proceeding with original model.")
            # 실패 시 부득이 원본 경로를 그대로 반환 (하위 호환성 대비)
            return source_onnx_path
            
        return v17_onnx_path

    def compile(self, model_spec: Model_Spec, output_dir: str) -> CompiledModel:
        """
        ONNX 소스 모델을 MLIR로 낮추고(Lowering) 최종적으로 IREE VMFB 바이너리로 컴파일합니다.
        """
        # 저장할 디렉토리가 없으면 생성
        os.makedirs(output_dir, exist_ok=True)
        
        final_artifact_path = os.path.join(output_dir, self.get_artifact_name(model_spec))
        
        # 0. 캐시 점검
        if self.is_cached(model_spec, output_dir):
            print(f"[IREE Compiler] Cached VMFB found at {final_artifact_path}. Skipping compilation.")
            return CompiledModel(spec=model_spec, backend_name=self.target_backend, artifact_path=Path(final_artifact_path))

        # 모델 스펙에서 ONNX 경로 가져오기 (Model_Spec이 복수 경로 dict 형태로 변경된 것에 대응)
        source_model_path = getattr(model_spec, 'model_paths', {}).get('onnx')
        if not source_model_path or not os.path.exists(source_model_path):
            raise FileNotFoundError(f"원본 모델 파일(ONNX)을 찾을 수 없거나 Model_Spec에 누락되었습니다: {source_model_path}")

        base_name = model_spec.name
        v17_onnx_path = os.path.join(output_dir, f"{base_name}_v17.onnx")
        mlir_path = os.path.join(output_dir, f"{base_name}.mlir")

        # 1. ONNX 버전 변환 (Opset 17)
        active_onnx_path = self._convert_to_opset_17(source_model_path, v17_onnx_path)

        # 2. MLIR Import (Subprocess)
        print(f"[IREE Compiler] Importing ONNX {active_onnx_path} to MLIR...")
        import_cmd = [self.import_onnx_bin, active_onnx_path, "-o", mlir_path]
        try:
            subprocess.run(import_cmd, check=True, capture_output=True, text=True)
            print(f"[IREE Compiler] Generated MLIR at {mlir_path}")
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] IREE MLIR Import Failed!\n{e.stderr}")
            raise e

        # 3. IREE Compile (Native Python API)
        print(f"[IREE Compiler] Compiling MLIR to {self.target_backend} VMFB...")
        extra_args = []
        if self.target_backend == "cuda":
            extra_args.append(f"--iree-cuda-target={self.iree_cuda_target}")
        else:
            # CPU 백엔드 시 호스트 시스템 타겟 최적화
            extra_args.append("--iree-llvmcpu-target-cpu=host")
            
        try:
            ireec.compile_file(
                mlir_path,
                output_file=final_artifact_path,
                target_backends=[self.target_backend],
                extra_args=extra_args
            )
            print(f"[IREE Compiler] Compilation Completed: {final_artifact_path}")
        except Exception as e:
            print(f"[ERROR] IREE Compilation Failed: {e}")
            raise e

        # 완료 후 불필요한 중간 산출물(선택적) 제거를 고려해 볼 수 있으나 디버깅을 위해 유지합니다.
        
        return CompiledModel(spec=model_spec, backend_name=self.target_backend, artifact_path=Path(final_artifact_path))
