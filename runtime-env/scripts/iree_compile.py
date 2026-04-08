import os
import subprocess
import onnx
import argparse

def run_cmd(cmd, description):
    """
    외부 셸 명령어를 실행하고 결과를 캡처하는 헬퍼 함수.
    
    Args:
        cmd (str): 실행할 셸 명령어 문자열.
        description (str): 현재 단계에 대한 설명 (로그 출력용).
        
    Returns:
        str: 표준 출력(stdout) 결과 문자열.
        
    Note:
        - subprocess.run을 사용하여 동기식으로 명령어를 실행.
        - capture_output=True를 통해 stdout/stderr를 메모리에 버퍼링.
        - returncode가 0이 아닐 경우(비정상 종료) stderr를 출력하고 프로세스를 중단함.
    """
    print(f"[EXECUTING] {description}")
    print(f"Command: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[ERROR] {result.stderr}")
        exit(1)
    return result.stdout

def main():
    # ... (argparse 설정 생략)
    parser = argparse.ArgumentParser(description="IREE End-to-End Pipeline for ONNX")
    parser.add_argument("--onnx_path", type=str, required=True, help="Input ONNX model path")
    parser.add_argument("--target", type=str, choices=["cpu", "cuda"], default="cpu", help="Target backend (cpu or cuda)")
    parser.add_argument("--input", type=str, help="Input data for the model (e.g., '1x3x224x224xf32=1.0')")
    parser.add_argument("--function", type=str, default="main", help="Model function name (default: main)")
    args = parser.parse_args()

    # IREE 빌드 도구 경로 설정 (Venv 환경 내 바이너리 참조)
    venv_bin = "/home/slugg/iree/build-demo/.venv/bin"
    python_bin = os.path.join(venv_bin, "python")
    import_onnx_bin = os.path.join(venv_bin, "iree-import-onnx") # ONNX -> Torch-MLIR/IREE MLIR 변환 도구
    compile_bin = os.path.join(venv_bin, "iree-compile")        # MLIR -> VMFB (FlatBuffer) 컴파일러
    run_module_bin = os.path.join(venv_bin, "iree-run-module")  # IREE VM 기반 런타임 실행기

    base_name = os.path.splitext(os.path.basename(args.onnx_path))[0]
    onnx_v17_path = f"models/{base_name}_v17.onnx"
    mlir_path = f"models/{base_name}.mlir"
    vmfb_path = f"models/{base_name}_{args.target}.vmfb"

    # 1. ONNX 버전 변환 (Opset 17 고정)
    # IREE의 ONNX 임포터는 최신 Opset 버전에 최적화되어 있으므로, 
    # 호환성을 위해 onnx.version_converter를 사용하여 변환을 수행함.
    print(f"\n--- [Step 1: ONNX Opset 17 Conversion] ---")
    original_model = onnx.load(args.onnx_path)
    converted_model = onnx.version_converter.convert_version(original_model, 17)
    onnx.save(converted_model, onnx_v17_path)
    print(f"Saved: {onnx_v17_path}")

    # 2. MLIR 임포트 (Input Dialect 생성)
    # ONNX 모델의 계산 그래프를 IREE가 이해할 수 있는 MLIR(linalg, flow, stream 등) Dialect로 변환.
    print(f"\n--- [Step 2: MLIR Import] ---")
    import_cmd = f"{import_onnx_bin} {onnx_v17_path} -o {mlir_path}"
    run_cmd(import_cmd, "Importing ONNX to MLIR")

    # 3. IREE 컴파일 (Lowering Pipeline)
    # MLIR 입력 파일을 하드웨어 독립적인(Flow) 레이어에서 하드웨어 종속적인(HAL) 레이어로 로워링함.
    # 최종적으로 IREE VM에서 실행 가능한 바이너리 포맷인 .vmfb(FlatBuffer)를 생성.
    print(f"\n--- [Step 3: IREE Compilation for {args.target}] ---")
    if args.target == "cpu":
        # LLVM-CPU 백엔드 사용: x86/ARM 등의 CPU 명령어로 컴파일.
        compile_cmd = f"{compile_bin} {mlir_path} --iree-hal-target-backends=llvm-cpu -o {vmfb_path}"
    else:
        # CUDA 백엔드 사용: NVIDIA GPU용 PTX 및 LLVM IR 생성.
        # --iree-cuda-target=sm_80 옵션은 NVIDIA Ampere 아키텍처(A100 등)를 타겟팅함.
        compile_cmd = (f"{compile_bin} {mlir_path} "
                       f"--iree-hal-target-backends=cuda "
                       f"--iree-cuda-target=sm_80 "  
                       f"-o {vmfb_path}")
    
    run_cmd(compile_cmd, f"Compiling MLIR to {args.target} VMFB")



    # 4. 런타임 실행 (Inference 수행)
    # 컴파일된 VMFB 모듈을 IREE 런타임에 로드하고 입력 데이터를 주입하여 추론을 실행함.
    print(f"\n--- [Step 4: IREE Runtime Execution] ---")
    device = "local-task" if args.target == "cpu" else "cuda"

    # TODO: CUDA target일 때 LD_LIBRARY_PATH를 추가해줘야하는데 아래 명령어를 커맨드로 실행하는 방법
    # if device == "cuda":
    #     run_cmd(f'export LD_LIBRARY_PATH="/usr/lib/wsl/lib:${LD_LIBRARY_PATH}"')
    
    # 입력 데이터가 없는 경우 사용자에게 입력을 요구함 (예: 1x3x224x224xf32=0.5)
    if not args.input:
        print("\n[PROMPT] Inference input data is required.")
        print("Format example: '1x3x224x224xf32=0.5' or '@file.bin'")
        args.input = input("Enter input data string: ").strip()

    if args.input:
        # iree-run-module은 경량화된 런타임 드라이버로, 함수 호출 및 텐서 입출력을 관리함.
        run_test_cmd = (f"{run_module_bin} --device={device} "
                        f"--module={vmfb_path} "
                        f"--function={args.function} "
                        f"--input=\"{args.input}\"")
        
        output = run_cmd(run_test_cmd, "Running Inference")
        print("\n--- [Inference Result] ---")
        print(output)
    else:
        print("\nExecution skipped because no input data was provided.")

if __name__ == "__main__":
    main()
