# Dataset Management

이 디렉토리는 모델 평가에 필요한 데이터를 수집하고 저장하는 공간입니다.

1. 주요 기능
- 외부 소스에서 데이터셋 다운로드
- 이미지 및 레이블 데이터의 로컬 저장 관리

2. 허깅페이스 데이터셋 다운로드 방법
load_dataset.py 스크립트를 사용하여 데이터를 확보할 수 있습니다.

명령어 예시:
python3 dataset/load_dataset.py --name ILSVRC/imagenet-1k --samples 1000

주요 옵션:
--name: 허깅페이스 데이터셋 명칭 (예: ILSVRC/imagenet-1k)
--samples: 다운로드할 샘플 개수
--split: 데이터셋 분할 선택 (train, validation, test)
--output: 저장될 루트 디렉토리 경로
