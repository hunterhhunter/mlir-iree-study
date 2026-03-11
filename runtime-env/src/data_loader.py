import numpy as np
import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor

class ImageNetDataset(Dataset):
    """
    로컬 디렉토리 구조를 기반으로 하는 ImageNet 검증 데이터셋 클래스.
    """
    def __init__(self, data_dir, label_file, processor):
        self.data_dir = data_dir
        self.processor = processor
        self.samples = []
        
        # 레이블 파일 로드 (파일명 정답_인덱스)
        if os.path.exists(label_file):
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 2:
                        self.samples.append((parts[0], int(parts[1])))
        else:
            print(f"[WARN] Label file {label_file} not found. Running in inference-only mode.")
            # 파일만 리스팅
            for f in os.listdir(data_dir):
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.samples.append((f, -1))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, label = self.samples[idx]
        img_path = os.path.join(self.data_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        
        # transformers ImageProcessor를 통한 전처리 (Resize, CenterCrop, Normalize)
        pixel_values = self.processor(images=image, return_tensors="pt")["pixel_values"]
        # (1, 3, 224, 224) -> (3, 224, 224)
        return pixel_values.squeeze(0), label, img_name

class CustomDataLoader:
    """
    IREE 런타임과 호환되는 고도화된 데이터 로더.
    PyTorch DataLoader를 래핑하여 NumPy 출력을 보장합니다.
    """
    def __init__(self, data_dir="datasets/imagenet_1k/val", label_file="datasets/imagenet_1k/val_labels.txt", 
                 model_id="google/mobilenet_v2_1.0_224", batch_size=1):
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.dataset = ImageNetDataset(data_dir, label_file, self.processor)
        self.batch_size = batch_size
        
        self.loader = DataLoader(
            self.dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=0  # IREE 환경과의 안정성을 위해 기본값 0
        )

    def __iter__(self):
        for batch in self.loader:
            pixel_values, labels, filenames = batch
            # IREE는 NumPy 배열을 입력으로 받으므로 변환
            yield pixel_values.numpy(), labels.numpy(), filenames

    def get_total_samples(self):
        return len(self.dataset)

    def get_processor_info(self):
        return self.processor
