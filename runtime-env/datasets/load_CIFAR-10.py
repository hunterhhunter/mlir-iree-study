import os
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

def main():
    # 설정값
    DATASET_NAME = "cifar10"
    SPLIT = "test"
    NUM_SAMPLES = 3000
    OUTPUT_DIR = "datasets/cifar10/test"
    LABEL_FILE = "datasets/cifar10/test_labels.txt"

    # 디렉토리 생성
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"[*] Loading '{DATASET_NAME}' ({SPLIT} split) from Hugging Face...")
    
    try:
        # 스트리밍 모드로 로드하여 필요한 만큼만 가져옴
        # CIFAR-10은 전체 크기가 작으므로, 스트리밍 모드(ZIP 파싱)에서 발생하는 무한 대기 버그를 피하기 위해
        # 일반 모드(streaming=False)로 전체 캐싱 후 처리합니다.
        dataset = load_dataset(DATASET_NAME, split=SPLIT)
        
        labels_list = []
        
        print(f"[*] Downloading and saving {NUM_SAMPLES} samples to '{OUTPUT_DIR}'...")
        
        for i, item in enumerate(tqdm(dataset, total=NUM_SAMPLES)):
            if i >= NUM_SAMPLES:
                break
            
            # 공식 cifar10은 'img' 키를 사용하고, imagenet은 'image'를 사용하므로 범용 대응
            image = item.get('image', item.get('img'))
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
