import os
import argparse
import subprocess

def run_cmd(cmd: list, description: str):
    """
    셸 명령어를 실행하고 실시간 출력을 유지하는 헬퍼 함수
    """
    print(f"\n[*] {description}")
    print(f"[*] Executing: {' '.join(cmd)}\n")
    try:
        # check=True: 0이 아닌 종료 코드가 나오면 에러 발생시킴
        subprocess.run(cmd, check=True)
        print("\n[+] Operation Completed Successfully!")
    except subprocess.CalledProcessError as e:
        print(f"\n[!] Export failed with exit code: {e.returncode}")
        exit(1)

def main():
    parser = argparse.ArgumentParser(description="Optimum CLI Python Wrapper for ONNX Export")
    parser.add_argument("--model", type=str, required=True, 
                        help="Hugging Face repo_id (e.g. 'microsoft/resnet-50') or local directory path.")
    parser.add_argument("--task", type=str, required=True, 
                        help="Task type for inference (e.g. 'image-classification', 'text-generation')")
    parser.add_argument("--output", type=str, default="None",
                        help="Output directory for the ONNX file. If 'None', defaults to models/<model_name>")
    parser.add_argument("--dtype", type=str, default=None,
                        help="데이터 타입 (예: fp16, fp32, int8). 미지정 시 모델 기본값 사용")
    parser.add_argument("--no-post-process", action="store_true",
                        help="export 후 검증(validation) 단계 생략. RAM 부족 시 사용")
    
    args = parser.parse_args()

    # 기본 폴더 경로 자동 생성 (사용자가 --output 을 미지정했을 경우)
    if args.output == "None":
        safe_name = args.model.replace("/", "_").rstrip("_")
        args.output = os.path.join("models", safe_name)

    # Output 디렉토리 보장
    os.makedirs(args.output, exist_ok=True)

    # optimum-cli 커맨드라인 조립
    # --model : 원본 소스
    # --task  : 작업 종류 강제 명시 (로컬 폴더에서 읽을 때 에러 방지)
    # 마지막 인자: 저장될 대상 폴더
    cmd = ["optimum-cli", "export", "onnx", "--model", args.model, "--task", args.task]
    if args.dtype:
        cmd += ["--dtype", args.dtype]
    if args.no_post_process:
        cmd.append("--no-post-process")
    cmd.append(args.output)

    run_cmd(cmd, f"Exporting {args.model} to ONNX format...")

if __name__ == "__main__":
    main()
