import sys
import os

# 프로젝트 루트를 sys.path에 추가 — pytest 외 uv run 등 직접 실행 시에도 src 패키지를 탐색할 수 있도록 합니다.
sys.path.insert(0, os.path.dirname(__file__))
