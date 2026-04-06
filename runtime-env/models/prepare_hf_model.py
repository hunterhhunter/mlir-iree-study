import os
import argparse
from huggingface_hub import snapshot_download

def download_model_repository(repo_id: str, file_format: str = None, output_root: str = "models"):
    """
    Hugging Face Hub에서 레포지토리의 모든 파일을 다운로드하여 저장합니다.
    """
    safe_repo_name = repo_id.replace("/", "_")
    target_dir = os.path.join(output_root, safe_repo_name)
    
    print(f"[*] Starting full snapshot download for: {repo_id}")
    print(f"[*] Target directory: {target_dir}")
    
    try:
        # 특정 포맷 필터링 설정 (기본값은 모든 파일)
        allow_patterns = None
        if file_format:
            # 기본 설정 파일들과 요청한 포맷의 모델 파일을 함께 포함
            allow_patterns = ["*.json", "*.txt", f"*.{file_format.lower()}"]
            print(f"[*] Filtering patterns: {allow_patterns}")

        # 레포지토리 전체 스냅샷 다운로드
        downloaded_path = snapshot_download(
            repo_id=repo_id,
            local_dir=target_dir,
            local_dir_use_symlinks=False,  # 심볼릭 링크 문제 방지
            allow_patterns=allow_patterns,
            ignore_patterns=["*.msgpack", "*.h5", "*.ot"] # 불필요한 프레임워크 바이너리 제외 (선택사항)
        )

        print(f"\n[+] Success! Entire repository saved to: {downloaded_path}")

    except Exception as e:
        print(f"\n[!] Critical Error during snapshot download: {e}")
        import sys
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Hugging Face Full Model Downloader")
    parser.add_argument("--name", type=str, required=True, help="Hugging Face repository ID")
    parser.add_argument("--format", type=str, default=None, help="Specific model format to include (e.g., 'onnx')")
    parser.add_argument("--output", type=str, default="models", help="Output root directory")

    args = parser.parse_args()
    
    download_model_repository(args.name, args.format, args.output)

if __name__ == "__main__":
    main()
