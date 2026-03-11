import os
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

def main():
    # 설정값
    DATASET_NAME = "imagenet-1k"
    SPLIT = "validation"
    NUM_SAMPLES = 1000
    OUTPUT_DIR = "datasets/imagenet_1k/val"
    LABEL_FILE = "datasets/imagenet_1k/val_labels.txt"

    # 디렉토리 생성
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"[*] Loading '{DATASET_NAME}' ({SPLIT} split) from Hugging Face...")
    
    try:
        # 스트리밍 모드로 로드하여 필요한 만큼만 가져옴
        dataset = load_dataset(DATASET_NAME, split=SPLIT, streaming=True, trust_remote_code=True)
        
        labels_list = []
        
        print(f"[*] Downloading and saving {NUM_SAMPLES} samples to '{OUTPUT_DIR}'...")
        
        for i, item in enumerate(tqdm(dataset, total=NUM_SAMPLES)):
            if i >= NUM_SAMPLES:
                break
            
            image = item['image']
            label = item['label']
            
            # 파일명 형식: image_0000.jpg
            file_name = f"image_{i:04d}.jpg"
            file_path = os.path.join(OUTPUT_DIR, file_name)
            
            # 이미지 저장 (표준 RGB 변환)
            if image.mode != "RGB":
                image = image.convert("RGB")
            image.save(file_path, "JPEG", quality=95)
            
            # 레이블 정보 기록 (파일명 정답_인덱스)
            labels_list.append(f"{file_name} {label}")

        # 레이블 파일 저장
        with open(LABEL_FILE, "w") as f:
            f.write("\n".join(labels_list))

        print(f"\n[+] Success! {NUM_SAMPLES} images saved in '{OUTPUT_DIR}'")
        print(f"[+] Label mapping saved in '{LABEL_FILE}'")

    except Exception as e:
        print(f"\n[!] Error: {e}")
        print("[!] Tip: Ensure you are logged in via 'huggingface-cli login' or have set HF_TOKEN.")

if __name__ == "__main__":
    main()
