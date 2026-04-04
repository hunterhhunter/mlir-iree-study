import os
from typing import Tuple
from src.core.model_spec import Task

def resolve_dataset_paths(task: Task, dataset_path: str, image_dir_arg: str, label_dir_arg: str) -> Tuple[str, str]:
    """
    Convention over Configuration (CoC) 데이터셋 스니핑 전담 해결사 플러그인.
    로더(Loader) 클래스들을 대신해 폴더 내부 냄새를 맡고 정확한 절대 주소를 규명합니다.
    """
    if not dataset_path:
        raise ValueError("[Resolver] --dataset 경로가 제공되지 않았습니다.")
        
    image_dir = ""
    label_path = ""
    
    # 1. 사용자가 --image-dir, --label-dir을 명시적으로 주었다면 100% 최우선 신뢰
    if image_dir_arg:
        image_dir = os.path.join(dataset_path, image_dir_arg)
    if label_dir_arg:
        label_path = os.path.join(dataset_path, label_dir_arg)
        
    # 2. 파라미터가 누락되었다면, Task(작업)별로 폴더 구조를 스니핑(추론)합니다.
    if task == Task.IMAGE_CLASSIFICATION:
        if not image_dir or not label_path:
            # ImageNet 구조 스니핑
            val_dir = os.path.join(dataset_path, "val")
            if os.path.exists(val_dir):
                if not image_dir:
                    image_dir = val_dir
            else:
                if not image_dir:
                    image_dir = dataset_path # 폴백
            
            # label_path가 지정 안 되어 있다면만 탐색
            if not label_path:
                label_path = os.path.join(dataset_path, "val_labels.txt")
            
    elif task == Task.OBJECT_DETECTION:
        if not image_dir or not label_path:
            # COCO 구조 스니핑
            img_val = os.path.join(dataset_path, "images", "val2017")
            img_train = os.path.join(dataset_path, "images", "train2017")
            if os.path.exists(img_val):
                image_dir = img_val
                if not label_path:
                    label_path = os.path.join(dataset_path, "labels", "val2017")
            elif os.path.exists(img_train):
                image_dir = img_train
                if not label_path:
                    label_path = os.path.join(dataset_path, "labels", "train2017")
            else:
                image_dir = os.path.join(dataset_path, "images", "train2017")
                if not label_path:
                    label_path = os.path.join(dataset_path, "labels", "train2017")
                
    elif task == Task.NLP_CLASSIFICATION:
        # NLP 파이프라인은 image_dir, label_dir 비전 전용 경로가 무의미하므로 무시
        pass

    elif task == Task.NLP_GENERATION:
        # LlamaLoader가 dataset_path 하위 val.json을 직접 탐색하므로 특별 처리 불필요
        pass

    elif task == Task.TIME_SERIES_FORECASTING:
        # ETTmLoader가 csv_path kwarg를 필요로 함.
        # main.py에서 loader_kwargs["csv_path"] = args.dataset 로 전달하므로 여기서는 pass.
        pass

    return image_dir, label_path
