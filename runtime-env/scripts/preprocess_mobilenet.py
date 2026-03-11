import numpy as np
from PIL import Image
import argparse
import os

def preprocess_image(image_path, output_bin):
    """
    MobileNetV2용 표준 ImageNet 전처리 파이프라인.
    1. 224x224 리사이즈 (Centering 생략, 단순 리사이즈)
    2. [0, 1] 범위 정규화
    3. ImageNet Mean/Std 정규화 (0.485, 0.456, 0.406 / 0.229, 0.224, 0.225)
    4. HWC -> NCHW 변환
    5. Raw Binary(float32) 저장
    """
    # 이미지 로드 및 RGB 변환
    img = Image.open(image_path).convert('RGB')
    
    # 224x224 리사이즈 (Bilinear)
    img = img.resize((224, 224), Image.BILINEAR)
    
    # Numpy 배열 변환 및 [0, 1] 정규화
    img_data = np.array(img).astype(np.float32) / 255.0
    
    # ImageNet 정규화
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_data = (img_data - mean) / std
    
    # HWC (224, 224, 3) -> NCHW (1, 3, 224, 224)
    img_data = np.transpose(img_data, (2, 0, 1))
    img_data = np.expand_dims(img_data, axis=0)
    
    # 바이너리 파일로 저장 (IREE @file.bin 입력용)
    img_data.tofile(output_bin)
    print(f"[SUCCESS] Preprocessed image saved to: {output_bin}")
    print(f"Shape: {img_data.shape}, Dtype: {img_data.dtype}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Input image path")
    parser.add_argument("--output", type=str, default="input_mobilenet.bin", help="Output binary file path")
    args = parser.parse_args()
    preprocess_image(args.image, args.output)
