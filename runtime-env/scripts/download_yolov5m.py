import os
import subprocess
import shutil

def main():
    # 1. ultralytics 패키지 설치 확인 및 설치
    try:
        import ultralytics
    except ImportError:
        print("[*] 'ultralytics' 패키지가 없습니다. 설치를 진행합니다...")
        subprocess.check_call(["pip", "install", "ultralytics"])
        
    from ultralytics import YOLO
    
    # 루트 디렉토리 및 모델이 저장될 목표 디렉토리 절대경로 산출
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    models_dir = os.path.join(project_root, "models", "yolov5m")
    os.makedirs(models_dir, exist_ok=True)
    
    print(f"[*] 다운로드 및 ONNX Export를 시작합니다 (저장 폴더: {models_dir})...")
    
    # PyTorch 모델 가중치가 scripts 폴더 같은 엉뚱한 곳에 받아지지 않도록 작업 폴더 변경
    original_cwd = os.getcwd()
    os.chdir(models_dir)
    
    try:
        # 2. 모델 로드 (자동으로 yolov5m.pt 가 models_dir 내부에 다운로드 됨)
        # v8 패키지에서 v5 모델을 사용할 경우 확장자로 자동 식별합니다.
        model = YOLO("yolov5m.pt") 
        
        # 3. ONNX 포맷으로 추출 (export)
        print("[*] 모델을 ONNX 규격으로 내보내는 중...")
        exported_path = model.export(format="onnx", opset=12, imgsz=[640,640], simplify=True)
        print(f"[+] Export 결과물: {exported_path}")
        
        # Export 된 파일을 안전하게 찾아서 변경된 target_path 와 매치시킵니다.
        target_path = os.path.join(models_dir, "yolov5m.onnx")
        
        if os.path.abspath(exported_path) != os.path.abspath(target_path):
            shutil.move(exported_path, target_path)
            
        print(f"[+] YOLOv5m ONNX 모델이 준비되었습니다: {target_path}")
        
    except Exception as e:
        print(f"[!] ONNX Export 중 에러가 발생했습니다: {e}")
    finally:
        os.chdir(original_cwd)

if __name__ == "__main__":
    main()
