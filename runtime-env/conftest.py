import sys
import os

# 프로젝트 루트 및 src 디렉토리를 sys.path에 추가
# - runtime-env/         : conftest, tests 패키지
# - runtime-env/src/     : dataloader, preprocessor, core, runtimes, evaluators, compilers, adapters, utils
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "src"))
