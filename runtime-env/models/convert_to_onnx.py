import os
import argparse
import json
import torch
import onnx
import numpy as np
from transformers import AutoModelForImageClassification, AutoModelForObjectDetection, AutoConfig

def convert_safetensors_to_onnx(model_dir: str, task: str = "classification"):
    """
    로컬 디렉토리의 가중치를 읽어 ONNX 포맷으로 익스포트합니다.
    """
    print(f"[*] Starting ONNX conversion for model in: {model_dir}")
    
    onnx_path = os.path.join(model_dir, "model.onnx")
    
    # 1. YOLO(Ultralytics) 스타일 모델인지 확인
    config_path = os.path.join(model_dir, "config.json")
    is_yolo = False
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            try:
                cfg_data = json.load(f)
                if isinstance(cfg_data, dict) and "model_type" not in cfg_data and ("names" in cfg_data or cfg_data.get("task") == "detect"):
                    is_yolo = True
            except json.JSONDecodeError:
                pass

    if is_yolo:
        print("[*] YOLO-style model detected. Processing via ultralytics...")
        try:
            from ultralytics import YOLO
            from safetensors.torch import load_file
            
            st_path = os.path.join(model_dir, "model.safetensors")
            with open(config_path, "r") as f:
                cfg = json.load(f)
            model_yaml = cfg.get("model", "yolov10s.yaml")
            
            # 1. 모델 아키텍처 초기화
            model = YOLO(model_yaml, task="detect")
            
            # 2. 가중치 주입 (Prefix 교정 포함)
            if os.path.exists(st_path):
                print(f"[*] Loading weights from: {st_path}")
                state_dict = load_file(st_path)
                
                # 키 이름 교정 로직 (model.model.0 -> model.0)
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith("model.model."):
                        new_state_dict[k.replace("model.model.", "model.")] = v
                    else:
                        new_state_dict[k] = v
                
                # 가중치 주입
                load_result = model.model.load_state_dict(new_state_dict, strict=False)
                print(f"[*] Load Result - Missing: {len(load_result.missing_keys)}, Unexpected: {len(load_result.unexpected_keys)}")
                
                sample_val = next(model.model.parameters()).data.mean().item()
                print(f"[*] Loaded Weights Mean: {sample_val:.6f}")
            
            # 3. ONNX 익스포트 (Bypass Mode: TopK 제거)
            print(f"[*] Exporting to ONNX (Raw Output Mode)...")
            
            # YOLOv10의 내부 속성을 수정하여 end2end 후처리 헤드를 비활성화
            if hasattr(model.model, 'end2end'):
                model.model.end2end = False
            
            # 익스포트 실행
            model.export(format="onnx", opset=17, dynamic=False, imgsz=640)
            
            # 파일 정리
            generated_onnx = model_yaml.replace(".yaml", ".onnx")
            if not os.path.exists(generated_onnx):
                # ultralytics 버전에 따라 현재 디렉토리에 생성될 수 있음
                base_yaml = os.path.basename(model_yaml).replace(".yaml", ".onnx")
                if os.path.exists(base_yaml):
                    generated_onnx = base_yaml

            if os.path.exists(generated_onnx):
                if os.path.exists(onnx_path): os.remove(onnx_path)
                os.rename(generated_onnx, onnx_path)
                print(f"[+] YOLO conversion complete: {onnx_path}")
            else:
                print(f"[!] Warning: Could not find generated ONNX file. Check current directory.")
                
            return
        except Exception as e:
            print(f"\n[!] YOLO export failed: {e}")
            import traceback
            traceback.print_exc()
            return

    # 2. 표준 Transformers 모델 로드
    try:
        config = AutoConfig.from_pretrained(model_dir)
        if task == "classification":
            model = AutoModelForImageClassification.from_pretrained(model_dir)
            input_name = "pixel_values"
            output_name = "logits"
        elif task == "detection":
            model = AutoModelForObjectDetection.from_pretrained(model_dir)
            input_name = "pixel_values"
            output_name = "output"
        else:
            raise ValueError(f"Unsupported task: {task}")
            
        model.eval()
        print(f"[+] Transformers model loaded: {config.model_type}")

        img_size = getattr(config, "image_size", 224)
        dummy_input = torch.randn(1, 3, img_size, img_size)
        
        print(f"[*] Exporting to ONNX: {onnx_path}...")
        torch.onnx.export(
            model, dummy_input, onnx_path,
            export_params=True, opset_version=17,
            do_constant_folding=True,
            input_names=[input_name], output_names=[output_name],
            dynamic_axes={input_name: {0: "batch_size"}, output_name: {0: "batch_size"}}
        )
        print(f"[+] Conversion complete: {onnx_path}")
    except Exception as e:
        print(f"\n[!] Transformers conversion failed: {e}")

def main():
    parser = argparse.ArgumentParser(description="Unified Model to ONNX Converter")
    parser.add_argument("--dir", type=str, required=True, help="Model directory path")
    parser.add_argument("--task", type=str, choices=["classification", "detection"], default="classification", help="Task type")

    args = parser.parse_args()
    convert_safetensors_to_onnx(args.dir, args.task)

if __name__ == "__main__":
    main()
