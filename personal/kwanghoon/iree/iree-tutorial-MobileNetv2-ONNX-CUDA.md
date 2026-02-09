# IREE 튜토리얼: MobileNetv2 ONNX 모델을 CUDA에서 실행하기

이 가이드는 ONNX 형식의 MobileNetv2 모델을 IREE를 사용해 MLIR로 변환하고, CUDA 타깃으로 컴파일한 뒤 GPU에서 실행하는 전 과정을 설명합니다. 공식 가이드([ONNX 프론트엔드](https://iree.dev/guides/ml-frameworks/onnx/) · [CUDA 배포](https://iree.dev/guides/deployment-configurations/gpu-cuda/))를 바탕으로 초보자도 따라 하기 쉽도록 정리했습니다.

## 1. 준비물 체크
- **운영체제**: Windows + WSL2(Ubuntu 22.04 기준)
- **GPU**: NVIDIA GeForce RTX 5070 Ti (Compute Capability 8.6) 또는 동급 CUDA GPU
- **CUDA 드라이버**: 최신 버전 권장, WSL용 라이브러리 경로 확인 필요
- **인터넷 연결**: ONNX 모델 다운로드 및 패키지 설치용

## 2. GPU 정보 확인
CUDA 타깃 아키텍처를 정확히 지정하려면 GPU 정보와 Compute Capability를 먼저 확인합니다.

```bash
nvidia-smi | grep CUDA
nvidia-smi --query-gpu=name,compute_cap --format=csv
```

출력의 compute_cap 값(예: 8.6)을 `--iree-cuda-target=sm_86`처럼 사용합니다.

## 3. Python 및 필수 패키지 설치
처음 구성한다면 WSL에서 기본 패키지 업데이트 후 Python과 venv 모듈을 설치합니다.

```bash
sudo apt update
sudo apt install python3 python3.10-venv
```

## 4. 작업 디렉터리와 가상 환경 준비

```bash
mkdir -p ~/iree
cd ~/iree
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
```

가상 환경을 활성화하면 셸 프롬프트에 `(venv)`가 표시됩니다.

## 5. IREE Python 패키지 설치
IREE 컴파일러와 런타임, ONNX 임포터 추가 모듈을 설치합니다.

```bash
python3 -m pip install iree-base-compiler iree-base-runtime
python3 -m pip install "iree-base-compiler[onnx]"
```

설치 후 `iree-compile`, `iree-import-onnx`, `iree-run-module` 같은 CLI 도구를 바로 사용할 수 있습니다.

## 6. IREE에서 CUDA 드라이버 인식 확인
GPU가 제대로 인식되는지 드라이버와 디바이스 목록을 확인합니다.

```bash
iree-run-module --list_drivers
iree-run-module --list_devices
```

WSL 환경이라면 CUDA 라이브러리 위치를 명시해야 할 수 있습니다.

```bash
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:${LD_LIBRARY_PATH}"
```

`ldconfig` 명령어로 라이브러리 위치를 확인할 수 있습니다.

```bash
ldconfig -p | grep libcuda
```

## 7. MobileNetv2 ONNX 모델 다운로드
공식 ONNX 모델 저장소에서 MobileNetv2 예제를 내려받습니다.

```bash
wget https://github.com/onnx/models/raw/refs/heads/main/validated/vision/classification/mobilenet/model/mobilenetv2-10.onnx
```

다운로드된 모델이 예상과 다르지 않은지 `onnxsim` 또는 `netron.app` 등을 활용해 확인해도 좋습니다.

## 8. ONNX → MLIR 변환
IREE ONNX 임포터를 사용해 MLIR 모듈을 생성합니다. 필요한 경우 opset 버전이나 다이나믹 입력 형태를 플래그로 조정하세요.

```bash
iree-import-onnx mobilenetv2-10.onnx \
  --opset-version=17 \
  -o mobilenetv2.mlir
```

변환 과정에서 경고가 발생하면 로그를 확인하고, 지원되지 않는 연산이 있는지 공식 문서를 참고합니다.

## 9. MLIR → VMFB (CUDA 타깃) 컴파일
`iree-compile`을 이용해 CUDA 실행 파일 형식인 VM FlatBuffer(vmfb)로 변환합니다. 최적화 레벨을 다양하게 시도해 성능을 비교할 수 있습니다.

```bash
iree-compile mobilenetv2.mlir \
  --iree-hal-target-device=cuda \
  --iree-cuda-target=sm_86 \
  --iree-opt-level=O2 \
  -o mobilenet_cuda.vmfb

iree-compile mobilenetv2.mlir \
  --iree-hal-target-device=cuda \
  --iree-cuda-target=sm_86 \
  --iree-opt-level=O0 \
  -o mobilenet_cuda_O0.vmfb

iree-compile mobilenetv2.mlir \
  --iree-hal-target-device=cuda \
  --iree-cuda-target=sm_86 \
  --iree-opt-level=O1 \
  -o mobilenet_cuda_O1.vmfb
```

- `--iree-hal-target-device`는 실행 대상(backend)을 지정합니다.
- `--iree-cuda-target`은 GPU 아키텍처 코드(sm_86 등)를 지정합니다.
- `--iree-opt-level`은 컴파일 최적화 수준(O0, O1, O2)을 조정합니다.

## 10. CPU에서 결과 sanity check
CUDA 실행 전 CPU에서 모델이 정상 동작하는지 먼저 확인해 봅니다. 입력을 제로 텐서로 설정해 추론을 빠르게 테스트합니다.

```bash
iree-run-module \
  --device=cpu \
  --module=mobilenet_cuda.vmfb \
  --function=torch-jit-export \
  --input="1x3x224x224xf32=0"
```

출력으로 1000차원 로짓 벡터가 표시되면 정상입니다.

## 11. CUDA에서 실행 및 성능 측정
CUDA 디바이스로 실행하여 GPU 연동을 확인합니다.

```bash
airee-run-module \
  --device=cuda \
  --module=mobilenet_cuda.vmfb \
  --function=torch-jit-export \
  --input="1x3x224x224xf32=0"
```

최적화 레벨별 성능을 비교하려면 `time` 또는 `iree-benchmark-module`을 사용할 수 있습니다.

```bash
time iree-run-module --device=cuda --module=mobilenet_cuda_O0.vmfb --function=torch-jit-export --input="1x3x224x224xf32=0"
time iree-run-module --device=cuda --module=mobilenet_cuda_O1.vmfb --function=torch-jit-export --input="1x3x224x224xf32=0"
time iree-run-module --device=cuda --module=mobilenet_cuda.vmfb --function=torch-jit-export --input="1x3x224x224xf32=0"
```

정밀한 측정을 원한다면 다음처럼 벤치마크 도구를 활용하세요.

```bash
airee-benchmark-module \
  --device=cuda \
  --module=mobilenet_cuda.vmfb \
  --function=torch-jit-export \
  --input="1x3x224x224xf32=0"
```

## 12. 자주 발생하는 문제와 해결 방법
- **CUDA 드라이버 미인식**: `libcuda.so` 경로가 다르면 `LD_LIBRARY_PATH`에 `/usr/lib/wsl/lib`를 추가합니다.
- **Unsupported ONNX op 오류**: `iree-import-onnx --help`로 대체 옵션을 확인하거나, 모델을 최신 opset으로 변환 후 재시도합니다.
- **Out-of-memory**: `--iree-codegen-cuda-enable-prefetch=false` 같은 추가 플래그로 메모리 사용을 줄이거나, 입력 배치 크기를 줄입니다.
- **성능 미달**: `--iree-opt-level`, `--iree-codegen-cuda-base-wg-size`, `--iree-codegen-cuda-heuristic-blocks` 등 성능 관련 플래그를 실험해 보세요.

## 13. 다음 단계
- [ONNX 가이드](https://iree.dev/guides/ml-frameworks/onnx/)에서 더 복잡한 모델 변환 사례를 살펴보기
- [CUDA 배포 가이드](https://iree.dev/guides/deployment-configurations/gpu-cuda/)를 통해 다양한 CUDA 설정과 문제 해결 패턴 배우기
- Python 바인딩(`iree-base-runtime`)을 활용해 모델을 애플리케이션에 통합하거나, `iree.compiler` API로 MLIR 파이프라인을 자동화하기

이 튜토리얼을 따라 MobileNetv2 ONNX 모델을 IREE로 손쉽게 GPU에 배포할 수 있습니다. 필요에 따라 플래그를 조정하여 다른 모델에도 동일한 흐름을 적용해 보세요.
