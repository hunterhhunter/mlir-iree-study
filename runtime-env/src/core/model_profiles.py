import onnx
from typing import Dict, Any
from .model_spec import Task, Model_Spec

# =====================================================================
# MLPerf 스타일의 선언적 프로필 레지스트리 (Declarative Profile Registry)
# =====================================================================
# 모델 코어 클래스(Model_Spec/Factory)를 수정하지 않고도, 
# 이 딕셔너리에 새로운 모델 블록을 추가하는 것만으로 완벽한 확장이 가능합니다. (OCP 엄수)

# 💡 팁(Tip): ONNX 바이너리에서 입력/출력 텐서의 이름을 런타임에 동적으로 식별해야 하는 경우
# 해당 키를 "__auto__"로 지정하면 팩토리가 안전하게 스니핑하여 덮어씁니다.

SUPPORTED_PROFILES: Dict[str, Dict[str, Any]] = {
    "resnet50": {
        "task": Task.IMAGE_CLASSIFICATION,
        "input_shapes": {"__auto__": (1, 3, 224, 224)},
        "input_dtype": {"__auto__": "float32"},
        "output_shapes": {"__auto__": (1, 1000)},
        "default_model_path": "models/Kalray_resnet50/resnet50-v1-7s.onnx",
        "default_dataset_path": "datasets/imagenet_1k",
        "prepare_model_script": "models/prepare_resnet50_kalray.py",
        "prepare_dataset_script": "datasets/prepare_imagenet_1k.py"
    },
    "yolov5m": {
        "task": Task.OBJECT_DETECTION,
        "input_shapes": {"__auto__": (1, 3, 640, 640)},
        "input_dtype": {"__auto__": "float32"},
        "output_shapes": {"__auto__": (1, 25200, 85)},
        "default_model_path": "models/yolov5m/yolov5m.onnx",
        "default_dataset_path": "datasets/coco128",
        "prepare_model_script": "models/prepare_yolov5m.py",
        "prepare_dataset_script": "datasets/prepare_coco128.py"
    },
    "bert-base-uncased": {
        "task": Task.NLP_CLASSIFICATION,
        "input_shapes": {"input_ids": (1, 128), "attention_mask": (1, 128)},
        "input_dtype": {"input_ids": "int64", "attention_mask": "int64"},
        "output_shapes": {"logits": (1, 2)},
        "default_model_path": "models/bert-base-uncased/bert_sst2.onnx",
        "default_dataset_path": "datasets/sst2_numpy",
        "prepare_model_script": "models/prepare_bert_sst2.py",
        "prepare_dataset_script": "datasets/prepare_text_numpy.py"
    },
    "llama-3.1-8b": {
        "task": Task.NLP_GENERATION,
        "input_shapes": {"input_ids": (1, 128), "attention_mask": (1, 128)},
        "input_dtype": {"input_ids": "int64", "attention_mask": "int64"},
        "output_shapes": {"logits": (1, 128, 32000)},
        "default_model_path": "models/meta-llama_Llama-3.1-8B",
        "default_dataset_path": "datasets/squad2/val.json",
        "prepare_model_script": "models/prepare_llama_3_1_8b.py",
        "prepare_dataset_script": "datasets/prepare_squad2.py"
    },
    "llama-3.2-3b": {
        "task": Task.NLP_GENERATION,
        "input_shapes": {"input_ids": (1, 4096), "attention_mask": (1, 4096)},
        "input_dtype": {"input_ids": "int64", "attention_mask": "int64"},
        "output_shapes": {"logits": (1, 4096, 128256)},
        "default_model_path": "models/meta-llama_Llama-3.2-3B-ONNX",
        "default_dataset_path": "datasets/squad2/val.json",
        "prepare_model_script": "models/prepare_llama_3_2_3b.py",
        "prepare_dataset_script": "datasets/prepare_squad2.py"
    },
    "bert-base-uncased-squad-v1": {
        "task": Task.QUESTION_ANSWERING,
        "input_shapes": {"input_ids": (1, 384), "attention_mask": (1, 384)},
        "input_dtype": {"input_ids": "int64", "attention_mask": "int64"},
        "output_shapes": {"start_logits": (1, 384), "end_logits": (1, 384)},
        "default_model_path": "models/bert-base-uncased-squad-v1/squad.onnx",
        "default_dataset_path": "datasets/squad_numpy",
        "prepare_model_script": "models/prepare_bert_squad.py",
        "prepare_dataset_script": "datasets/prepare_squad_numpy.py"
    },
    "patchtst-fm-r1": {
        "task": Task.TIME_SERIES_FORECASTING,
        "input_shapes": {
            "past_values":        (1, 512, 7),
            "past_observed_mask": (1, 512, 7),
        },
        "input_dtype": {
            "past_values":        "float32",
            "past_observed_mask": "bool",
        },
        "output_shapes": {"__auto__": (1, 96, 7)},
        "default_model_path": "models/ibm-research_patchtst-fm-r1-ONNX/model.onnx",
        "default_dataset_path": "datasets/etth1/ETTh1.csv",
        "prepare_model_script": "models/prepare_patchtst.py",
        "prepare_dataset_script": "datasets/prepare_etth1.py"
    }
}

def _parse_onnx_io_names(onnx_path: str):
    """지정된 ONNX 모델을 로드하여 Input/Output 텐서 이름을 자동 추출합니다."""
    model = onnx.load(onnx_path)
    input_name = model.graph.input[0].name
    output_name = model.graph.output[0].name
    return input_name, output_name

def create_model_spec(model_name: str, onnx_path: str, task: Task = Task.IMAGE_CLASSIFICATION) -> Model_Spec:
    """
    MLPerf 스타일의 Profile Registry(SUPPORTED_PROFILES)에서 모델 규격을 동적으로 조회하여
    순수한 Model_Spec 인스턴스를 생성하는 팩토리 함수 (OCP 원칙 준수).
    """
    profile = SUPPORTED_PROFILES.get(model_name)
    if not profile:
        raise ValueError(f"[Factory] 지원되지 않는 모델 프로필입니다: '{model_name}'. 'model_profiles.py'를 확인하세요.")

    print(f"[Factory] Loading Profile for '{model_name}' (Task: {profile['task'].name})")
    
    # 레지스트리에서 딥카피 대신 새 딕셔너리로 구성
    spec_kwargs = {
        "task": profile["task"],
        "input_shapes": dict(profile["input_shapes"]),
        "input_dtype": dict(profile["input_dtype"]),
        "output_shapes": dict(profile["output_shapes"]),
    }
    
    # 단일 입력 기반의 비전 모델 등 ONNX 구조상 런타임 자동 탐지가 필요한 경우 (__auto__)
    if "__auto__" in spec_kwargs["input_shapes"]:
        input_n, output_n = _parse_onnx_io_names(onnx_path)
        print(f"[Factory] Detected ONNX I/O dynamically -> Input: '{input_n}', Output: '{output_n}'")

        spec_kwargs["input_shapes"] = {input_n: spec_kwargs["input_shapes"].pop("__auto__")}
        spec_kwargs["input_dtype"] = {input_n: spec_kwargs["input_dtype"].pop("__auto__", "float32")}

        if "__auto__" in spec_kwargs["output_shapes"]:
            spec_kwargs["output_shapes"] = {output_n: spec_kwargs["output_shapes"].pop("__auto__")}

    # 입력은 고정이고 출력만 __auto__인 경우 (예: patchtst-fm-r1) — 출력 이름만 스니핑
    elif "__auto__" in spec_kwargs["output_shapes"]:
        _, output_n = _parse_onnx_io_names(onnx_path)
        print(f"[Factory] Detected ONNX Output dynamically -> Output: '{output_n}'")
        spec_kwargs["output_shapes"] = {output_n: spec_kwargs["output_shapes"].pop("__auto__")}

    return Model_Spec(
        name=model_name,
        task=spec_kwargs["task"],
        input_shapes=spec_kwargs["input_shapes"],
        input_dtype=spec_kwargs["input_dtype"],
        output_shapes=spec_kwargs["output_shapes"],
        model_paths={"onnx": onnx_path}
    )
