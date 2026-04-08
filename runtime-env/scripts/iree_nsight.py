import os
import subprocess
import onnx
import argparse
import sys

def run_cmd(cmd, description, capture=True):
    """
    Nsight 프로파일링 명령어를 실행하는 헬퍼 함수.
    
    Args:
        capture (bool): True이면 출력을 변수에 저장, False이면 실시간으로 터미널에 출력함.
                        (NCU처럼 진행 상황 확인이 필요한 경우 False 권장)
    """
    print(f"\n[EXECUTING] {description}")
    print(f"Command: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=capture, text=True)
    if result.returncode != 0:
        print(f"[ERROR] {result.stderr if capture else 'Command failed'}")
        return False, (result.stderr if capture else "")
    return True, (result.stdout if capture else "Success")

def main():
    parser = argparse.ArgumentParser(description="IREE NVIDIA Nsight Profiling Automation")
    parser.add_argument("--onnx_path", type=str, required=True, help="Input ONNX model path")
    parser.add_argument("--input", type=str, required=True, help="Input data (e.g., '1x3x224x224xf32=1.0')")
    parser.add_argument("--mode", type=str, choices=["nsys", "ncu", "both"], default="both", help="Profiling mode")
    parser.add_argument("--target_arch", type=str, default="sm_80", help="CUDA Target Architecture (e.g., sm_80, sm_86)")
    parser.add_argument("--function", type=str, default="main", help="Model function name")
    args = parser.parse_args()

    # IREE 빌드 환경 경로
    venv_bin = "/home/slugg/iree/build-demo/.venv/bin"
    import_onnx_bin = os.path.join(venv_bin, "iree-import-onnx")
    compile_bin = os.path.join(venv_bin, "iree-compile")
    run_module_bin = os.path.join(venv_bin, "iree-run-module")
    
    # Nsight 시스템 도구 (환경 변수 PATH에 등록되어 있어야 함)
    nsys_bin = "nsys"
    ncu_bin = "ncu"

    base_name = os.path.splitext(os.path.basename(args.onnx_path))[0]
    mlir_path = f"models/{base_name}.mlir"
    vmfb_path = f"models/{base_name}_cuda.vmfb"
    nsys_report = f"logs/{base_name}_nsys"
    ncu_report = f"logs/{base_name}_ncu"

    print(f"--- [Step 1: IREE CUDA Lowering Pipeline] ---")
    
    # ONNX -> MLIR 변환
    success, _ = run_cmd(f"{import_onnx_bin} {args.onnx_path} -o {mlir_path}", "Importing ONNX to MLIR")
    if not success: return

    # CUDA VMFB 컴파일: 타겟 아키텍처(sm_xx)에 맞춘 코드 생성
    # IREE는 이 단계에서 Linalg Dialect의 Tiling 및 Fusion 최적화를 수행하여 커널을 생성함.
    compile_cmd = (f"{compile_bin} {mlir_path} "
                   f"--iree-hal-target-backends=cuda "
                   f"--iree-cuda-target={args.target_arch} "
                   f"-o {vmfb_path}")
    success, _ = run_cmd(compile_cmd, "Compiling to CUDA VMFB")
    if not success: return

    # 프로파일링 대상이 될 기본 실행 명령어
    base_run_cmd = (f"{run_module_bin} --device=cuda "
                    f"--module={vmfb_path} --function={args.function} "
                    f"--input=\"{args.input}\"")

    # Step 2: Nsight Systems (nsys) 프로파일링
    # 타임라인 분석을 통해 CPU-GPU 간의 Dispatch 간격 및 동기화 오버헤드를 확인.
    if args.mode in ["nsys", "both"]:
        print(f"\n--- [Step 2: Nsight Systems - Timeline Analysis] ---")
        # --trace=cuda,nvtx,osrt: CUDA API, NVTX 마커, OS 런타임을 추적하여 IREE 스케줄링 시각화.
        nsys_cmd = (f"{nsys_bin} profile --trace=cuda,nvtx,osrt --force-overwrite=true "
                    f"--output={nsys_report} {base_run_cmd}")
        success, output = run_cmd(nsys_cmd, "Generating Nsys Report", capture=False)
        if success:
            print(f"[SUCCESS] Nsys report saved to: {nsys_report}.nsys-rep")

    # Step 3: Nsight Compute (ncu) 프로파일링
    # 개별 CUDA 커널의 리소스 사용량 및 하드웨어 성능 한계(Roofline)를 정밀 분석.
    if args.mode in ["ncu", "both"]:
        print(f"\n--- [Step 3: Nsight Compute - Kernel Deep Dive] ---")
        # 주요 메트릭: SM 점유율(Occupancy), 메모리 대역폭(Memory Throughput) 등 캡처.
        ncu_cmd = (f"{ncu_bin} --metrics sm__throughput.avg,gpu__compute_memory_throughput.avg "
                   f"--target-processes all --force-overwrite "
                   f"--export {ncu_report} {base_run_cmd}")
        # NCU의 출력은 매우 방대하므로 실시간 터미널 출력을 허용함.
        success, output = run_cmd(ncu_cmd, "Generating NCU Report", capture=False)
        if success:
            print(f"[SUCCESS] NCU report saved to: {ncu_report}.ncu-rep")

    print("\n" + "="*60)
    print("IREE NSIGHT PROFILING COMPLETE")
    print("="*60)
    print(f"Analyzed Model: {args.onnx_path}")
    print(f"Target Arch: {args.target_arch}")
    if args.mode in ["nsys", "both"]: print(f"Nsys Report: {nsys_report}.nsys-rep")
    if args.mode in ["ncu", "both"]: print(f"NCU Report: {ncu_report}.ncu-rep")
    print("="*60)
    print("\n[ANALYSIS TIP]")
    print("1. nsys-ui를 사용하여 HAL Dispatch 사이의 유휴 시간(Gap)을 확인하십시오.")
    print("2. ncu-ui를 사용하여 커널의 Register Pressure 및 SM Occupancy를 점검하십시오.")

if __name__ == "__main__":
    main()
