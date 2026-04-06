import os
import shutil

def main():
    print("="*60)
    print(" Ultralytics 패키지 기반 YOLOv5m ONNX Export 스크립트 ")
    print("="*60)
    
    # 1. 저장할 폴더 설정
    output_dir = os.path.join(os.path.dirname(__file__), "yolov5m")
    os.makedirs(output_dir, exist_ok=True)
    final_onnx_path = os.path.join(output_dir, "yolov5m.onnx")
    
    if os.path.exists(final_onnx_path):
        print(f"[*] 이미 파이프라인 대상 모델이 존재합니다: {final_onnx_path}")
        return
        
    print("[*] Ultralytics 라이브러리를 통해 모델 레이어를 불러옵니다...")
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[!] ultralytics 패키지가 없습니다. 터미널에서 아래 명령어로 설치해주세요:")
        print("    uv pip install ultralytics")
        return

    # YOLO 클래스를 거치면 허깅페이스/공식 저장소에서 .pt 파일을 자동으로 캐시 다운로드 합니다.
    # yolov5m.pt 또는 v8 호환 구조인 yolov5mu.pt 를 사용할 수 있습니다.
    model = YOLO("yolov5m.pt")
    
    print(f"[*] 모델 다운로드 및 로드 성공! ONNX 포맷으로 변환(Export)합니다...")
    
    # Ultralytics 엔진에 내장된 강력한 export 기능 사용 (NMS 여부, Dynamic batch 등 자동 최적화)
    try:
        # 현재 실행중인 디렉토리에 yolov5m.onnx 이름으로 추출됩니다.
        export_path = model.export(
            format="onnx", 
            imgsz=640,
            dynamic=True  # Batch 크기 등 Dynamic 축 허용
        )
        print(f"[+] Export 추출 성공: {export_path}")
        
        # 보기 좋게 기존 구조화된 폴더(models/yolov5m/) 로 파일 이동
        if os.path.exists(export_path):
            shutil.move(export_path, final_onnx_path)
            print(f"[+] 모델 파일 정리 완료: {final_onnx_path}")
            
        # 잔여 .pt 파일 정리 (옵션)
        pt_path = "yolov5m.pt"
        if os.path.exists(pt_path):
            shutil.move(pt_path, os.path.join(output_dir, "yolov5m.pt"))
        
    except Exception as e:
        print(f"[!] ONNX Export 중 오류 발생: {e}")

if __name__ == "__main__":
    main()
