import os
import subprocess
import shutil
import tempfile
import urllib.request
import sys

def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    models_dir = os.path.join(project_root, "models", "yolov5m")
    os.makedirs(models_dir, exist_ok=True)
    target_onnx = os.path.join(models_dir, "yolov5m.onnx")

    print("[*] 오리지널 YOLOv5m (Legacy) 모델 다운로드 및 Export를 시작합니다...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # 1. YOLOv5 공식(레거시) 레포지토리 클론 (폴더가 비어있어야 함)
        print("[*] YOLOv5 레포지토리를 임시 폴더에 Clone 합니다...")
        subprocess.check_call(["git", "clone", "https://github.com/ultralytics/yolov5", temp_dir])

        # 2. 모델 가중치 분리(강제) 다운로드 (내부 다운로더 멈춤 방지 및 진행률 표시)
        pt_url = "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5m.pt"
        pt_path = os.path.join(temp_dir, "yolov5m.pt")
        print(f"[*] 모델 가중치 사전 다운로드 중: {pt_url}")
        
        try:
            req = urllib.request.Request(pt_url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
            with urllib.request.urlopen(req) as response, open(pt_path, 'wb') as out_file:
                total_length = response.getheader('content-length')
                if total_length is None:
                    shutil.copyfileobj(response, out_file)
                else:
                    total_length = int(total_length)
                    fetched = 0
                    while True:
                        chunk = response.read(8192)
                        if not chunk:
                            break
                        out_file.write(chunk)
                        fetched += len(chunk)
                        percent = int(fetched * 100 / total_length)
                        print(f"\rDownloading... {percent}% ({fetched/(1024*1024):.1f}MB / {total_length/(1024*1024):.1f}MB)", end="")
                        sys.stdout.flush()
            print("\n[+] 가중치 다운로드 완료!")
        except Exception as e:
            print(f"\n[!] 가중치 다운로드 실패: {e}")
            return
        
        # 3. export.py 실행 (미리 다운 받은 yolov5m.pt로 ONNX 변환)
        print("[*] export.py를 통해 Legacy YOLOv5m 포맷(25200, 85)으로 파싱합니다...")
        export_script = os.path.join(temp_dir, "export.py")
        subprocess.check_call([
            "python", export_script, 
            "--weights", "yolov5m.pt", 
            "--include", "onnx", 
            "--opset", "17",
            "--dynamic"
        ], cwd=temp_dir)
        
        # 4. 결과물 복사
        exported_onnx = os.path.join(temp_dir, "yolov5m.onnx")
        if os.path.exists(exported_onnx):
            shutil.copy(exported_onnx, target_onnx)
            print(f"[+] 오리지널 YOLOv5m ONNX 모델이 준비되었습니다: {target_onnx}")
        else:
            print("[!] Export 실패: 변환된 onnx 파일을 찾을 수 없습니다.")

if __name__ == "__main__":
    main()
