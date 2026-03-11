# Models
컴파일 파이프라인의 각 단계별 아티팩트를 저장합니다.
- *.onnx: 원본 및 Opset 변환된 ONNX 모델
- *.mlir: IREE/Torch-MLIR Dialect로 변환된 중간 표현식
- *.vmfb: IREE HAL용 최종 컴파일 바이너리
