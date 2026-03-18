import os
import argparse
import json
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Simple Dataset Downloader (Hugging Face)")
    parser.add_argument("--name", type=str, required=True, help="Hugging Face dataset name (e.g., 'cifar10')")
    parser.add_argument("--split", type=str, default="validation", help="Dataset split (default: 'validation')")
    parser.add_argument("--samples", type=int, default=100, help="Number of samples to collect")
    parser.add_argument("--output", type=str, default="dataset", help="Output root directory")

    args = parser.parse_args()

    # 데이터셋 저장 경로 설정
    dataset_tag = args.name.replace("/", "_")
    output_dir = os.path.join(args.output, dataset_tag)
    image_dir = os.path.join(output_dir, "validation")
    label_file = os.path.join(output_dir, "validation_labels.json")
    class_map_file = os.path.join(output_dir, "classes.json")

    os.makedirs(image_dir, exist_ok=True)

    print(f"[*] Loading dataset '{args.name}' (split: {args.split}) in streaming mode...")
    try:
        ds = load_dataset(args.name, split=args.split, streaming=True)
    except Exception as e:
        print(f"[!] Error loading dataset: {e}")
        return

    # 클래스 정보(Label names) 저장 시도
    try:
        features = ds.features if hasattr(ds, 'features') else None
        if features:
            for key, feat in features.items():
                if hasattr(feat, 'names'):
                    class_map = {i: name for i, name in enumerate(feat.names)}
                    with open(class_map_file, "w", encoding="utf-8") as f:
                        json.dump(class_map, f, indent=4, ensure_ascii=False)
                    print(f"[+] Class map saved to {class_map_file}")
                    break
    except Exception as e:
        print(f"[*] Could not save class map: {e}")

    annotations = {}
    print(f"[*] Downloading {args.samples} samples...")

    for i, item in enumerate(tqdm(ds, total=args.samples)):
        if i >= args.samples:
            break

        # 이미지 필드 자동 탐색 (PIL.Image 객체 찾기)
        image_key = None
        for key, value in item.items():
            if isinstance(value, Image.Image):
                image_key = key
                break
        
        if image_key is None:
            # 이미지가 객체가 아닐 경우 'image' 또는 'img' 키 확인
            for k in ['image', 'img']:
                if k in item:
                    image_key = k
                    break
        
        if image_key is None:
            print(f"\n[-] No image found in sample {i}, skipping.")
            continue

        img = item[image_key]
        if not isinstance(img, Image.Image): # 경로인 경우 등 처리
            try:
                img = Image.open(img)
            except:
                continue

        if img.mode != "RGB":
            img = img.convert("RGB")

        file_name = f"sample_{i:05d}.jpg"
        file_path = os.path.join(image_dir, file_name)
        img.save(file_path, "JPEG", quality=95)

        # 이미지를 제외한 나머지 모든 데이터를 어노테이션으로 저장
        record = {k: v for k, v in item.items() if k != image_key}
        annotations[file_name] = record

    # 레이블 JSON 저장
    with open(label_file, "w", encoding="utf-8") as f:
        json.dump(annotations, f, indent=4, ensure_ascii=False)

    print(f"[+] Successfully saved {len(annotations)} samples to {output_dir}")

if __name__ == "__main__":
    main()
