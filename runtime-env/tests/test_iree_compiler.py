import sys
import os
import unittest
import pytest
from pathlib import Path

# 프로젝트 최상단 경로 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.model_spec import Model_Spec, Task
from compilers import get_compiler, Compiler
from compilers.iree_compiler import IREECompiler

class TestIREECompilerIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("\n=== [SETUP] IREE Compiler Test Environment ===")
        # 사용자가 다운로드할(혹은 제공할) ONNX 모델의 타겟 경로
        cls.models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models"))
        os.makedirs(cls.models_dir, exist_ok=True)
        
        cls.onnx_path = os.path.join(cls.models_dir, "resnet-50/resnet50.onnx")
        cls.output_dir = os.path.join(cls.models_dir, "compiled")
        
        # 1. 모델 스펙 정의
        cls.spec = Model_Spec(
            name="resnet50",
            model_paths={"onnx": cls.onnx_path},
            task=Task.IMAGE_CLASSIFICATION,
            input_shapes={"input": (1, 3, 224, 224)},
            input_dtype={"input": "float32"},
            output_shapes={"output": (1, 1000)}
        )

    def test_01_factory_instantiation(self):
        """Step 1: 팩토리 패턴을 통한 컴파일러 정상 생성 확인"""
        print("\n[TEST 1] Factory Instantiation")
        compiler = get_compiler("iree", target_backend="llvm-cpu")
        
        self.assertIsInstance(compiler, Compiler)
        self.assertIsInstance(compiler, IREECompiler)
        self.assertEqual(compiler.target_backend, "llvm-cpu")
        print(" -> [OK] Compiler successfully instantiated via factory.")

    def test_02_get_artifact_name(self):
        """Step 2: 타겟별 산출물 일관성 네이밍 룰 검증"""
        print("\n[TEST 2] Artifact Naming Convention")
        compiler_cpu = get_compiler("iree", target_backend="llvm-cpu")
        compiler_gpu = get_compiler("iree", target_backend="cuda", iree_cuda_target="sm_86")
        
        cpu_name = compiler_cpu.get_artifact_name(self.spec)
        gpu_name = compiler_gpu.get_artifact_name(self.spec)
        
        self.assertEqual(cpu_name, "resnet50_iree_cpu.vmfb")
        self.assertEqual(gpu_name, "resnet50_iree_cuda.vmfb")
        print(f" -> [OK] Artifact names correctly resolved: {cpu_name}, {gpu_name}")

    @pytest.mark.skip(reason="Known IREE upstream issue: failed to legalize 'onnx.ReduceMean' for ResNet50")
    def test_03_compilation_pipeline(self):
        """
        Step 3: 실제 컴파일 파이프라인(ONNX -> MLIR -> VMFB) 통합 검증
        (* 단, 실제 ONNX 파일이 없을 경우 스킵하도록 처리)
        """
        print("\n[TEST 3] Full IREE Compilation Pipeline")
        if not os.path.exists(self.onnx_path):
            self.skipTest(f"Model file not found at {self.onnx_path}. Please download it first.")
            
        compiler = get_compiler("iree", target_backend="llvm-cpu")
        
        # 컴파일 수행 (시간이 걸릴 수 있음)
        print(" -> Compiling... This may take a few minutes.")
        final_vmfb_path = compiler.compile(self.spec, output_dir=self.output_dir)
        
        # 검증 1: 반환된 존재 파일 경로가 정상인가?
        self.assertTrue(os.path.exists(final_vmfb_path), "VMFB output file does not exist.")
        self.assertIn("resnet50_iree_cpu.vmfb", final_vmfb_path)
        print(f" -> [OK] Compilation successful! Saved at: {final_vmfb_path}")
        
        # 검증 2: 캐시 히트 동작 확인 (바로 리턴되어야 함)
        print(" -> Testing cache hit...")
        cached_vmfb_path = compiler.compile(self.spec, output_dir=self.output_dir)
        self.assertEqual(final_vmfb_path, cached_vmfb_path)
        print(" -> [OK] Cache hit functional.")

if __name__ == "__main__":
    unittest.main(verbosity=2)
