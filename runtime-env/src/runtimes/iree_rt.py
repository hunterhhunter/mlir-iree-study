import os
import subprocess
import onnx
import numpy as np
import time
import mmap
import iree.compiler.tools as ireec
import iree.runtime as ireert

class IREERuntime:
    """
    IREE (MLIR) 기반의 컴파일 및 실행 엔진 래퍼.
    """
    def __init__(self, model_path, device="cpu"):
        self.model_path = model_path
        self.device_type = device # 'cpu' or 'cuda'
        self.iree_device = "local-task" if device == "cpu" else "cuda"
        
        # 경로 설정
        base_name = os.path.splitext(os.path.basename(model_path))[0]
        self.onnx_v17_path = f"models/{base_name}_v17.onnx"
        self.mlir_path = f"models/{base_name}.mlir"
        self.vmfb_path = f"models/{base_name}_{device}.vmfb"
        
        # 가상환경 내 바이너리 경로
        self.venv_bin = os.path.join(os.getcwd(), ".venv/bin")
        self.import_onnx_bin = os.path.join(self.venv_bin, "iree-import-onnx")

        # 런타임 캐시 (연속 추론용)
        self.config = None
        self.context = None
        self.bound_func = None

    def _preprocess_onnx(self):
        """ONNX 모델을 Opset 17로 변환합니다."""
        if os.path.exists(self.onnx_v17_path):
            return self.onnx_v17_path
        print(f"[IREE] Converting {self.model_path} to Opset 17...")
        model = onnx.load(self.model_path)
        v17_model = onnx.version_converter.convert_version(model, 17)
        onnx.save(v17_model, self.onnx_v17_path)
        return self.onnx_v17_path

    def compile(self, force=False):
        """ONNX -> MLIR -> VMFB 컴파일 파이프라인을 실행합니다."""
        if not force and os.path.exists(self.vmfb_path):
            print(f"[IREE] VMFB already exists at {self.vmfb_path}. Skipping compile.")
            return self.vmfb_path

        # 1. ONNX 전처리
        self._preprocess_onnx()
        
        # 2. MLIR Import
        print(f"[IREE] Importing ONNX to MLIR...")
        subprocess.run([self.import_onnx_bin, self.onnx_v17_path, "-o", self.mlir_path], check=True)
        
        # 3. IREE Compile
        print(f"[IREE] Compiling MLIR to VMFB for {self.device_type}...")
        target_backend = "cuda" if self.device_type == "cuda" else "llvm-cpu"
        
        extra_args = []
        if self.device_type == "cuda":
            extra_args.append("--iree-cuda-target=sm_86")
        else:
            extra_args.append("--iree-llvmcpu-target-cpu=host")
        
        ireec.compile_file(
            self.mlir_path,
            output_file=self.vmfb_path,
            target_backends=[target_backend],
            extra_args=extra_args
        )
        print(f"[IREE] Compilation completed: {self.vmfb_path}")
        return self.vmfb_path

    def initialize_runtime(self, function_name="torch-jit-export"):
        """VMFB를 로드하고 추론을 위한 컨텍스트를 한 번만 초기화합니다."""
        if self.context:
            return

        print(f"[IREE] Initializing HAL Device: {self.iree_device}...")
        self.config = ireert.Config(self.iree_device)
        
        # Python mmap을 사용하여 파일을 메모리에 직접 매핑 (가장 호환성이 높고 빠름)
        with open(self.vmfb_path, "rb") as f:
            # mmap.ACCESS_READ를 사용하여 읽기 전용으로 매핑
            mapped_file = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            # from_flatbuffer 또는 from_buffer는 대부분의 버전에서 지원됨
            vm_module = ireert.VmModule.from_flatbuffer(self.config.vm_instance, mapped_file)
        
        self.context = ireert.SystemContext(config=self.config)
        self.context.add_vm_module(vm_module)
        
        # 모듈 내 함수 바인딩
        bound_module = self.context.modules.module
        try:
            self.bound_func = getattr(bound_module, function_name)
        except AttributeError:
            print(f"[IREE] '{function_name}' not found, falling back to 'main'")
            self.bound_func = getattr(bound_module, "main")

    def execute(self, input_data):
        """이미 초기화된 컨텍스트를 사용하여 빠르게 추론을 수행합니다."""
        if not self.bound_func:
            self.initialize_runtime()
            
        # IREE 추론 (단일 프로세스 루프 내에서 매우 빠름)
        result = self.bound_func(input_data)
        
        # DeviceArray를 NumPy 배열로 변환
        return np.array(result)

    def profile(self, input_str="1x3x224x224xf32=0.5"):
        print(f"[IREE] Benchmarking on {self.iree_device}...")
        benchmark_bin = os.path.join(self.venv_bin, "iree-benchmark-module")
        
        cmd = [
            benchmark_bin,
            f"--device={self.iree_device}",
            f"--module={self.vmfb_path}",
            f"--input={input_str}",
            "--benchmark_repetitions=5"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)