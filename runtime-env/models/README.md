# Model Management

모델 파일을 수집하고 관리하는 디렉토리입니다.

## 모델 다운로드 방법
Hugging Face에서 원하는 포맷의 모델을 자동으로 다운로드할 수 있습니다.

```bash
python3 models/download_models_from_huggingface.py --name <HF_REPO_ID> --format <FORMAT>
```

- **예시**: `python3 models/download_models_from_huggingface.py --name google/mobilenet_v2_1.0_224 --format onnx`
- **결과**: `models/google_mobilenet_v2_1.0_224/model.onnx` 경로에 저장됩니다.
