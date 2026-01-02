cmake_minimum_required(VERSION 3.20)
project(toy-compiler)

# 1. MLIR 패키지 찾기
find_package(MLIR REQUIRED CONFIG)

# 2. MLIR 관련 변수 및 매크로 가져오기
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddMLIR)
include(HandleLLVMOptions) # [중요] RTTI 끄기 설정 등이 여기 들어있음

# 3. 헤더 파일 경로 설정
include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/build/include
  ${MLIR_INCLUDE_DIRS}
  ${LLVM_INCLUDE_DIRS}
)

# 4. 실행 파일 정의
add_executable(my_toy_compiler
  dummy_main.cpp
  lib/toy/Dialect.cpp
  lib/toy/Ops.cpp
)

target_compile_options(my_toy_compiler PRIVATE -fno-rtti)

# [핵심 2] 라이브러리 링크 (아까 추가한 헤더들에 대응하는 라이브러리 추가)
target_link_libraries(my_toy_compiler PRIVATE
  MLIRIR
  MLIRSupport
  MLIRParser
  MLIRSideEffectInterfaces
  MLIRBytecodeOpInterface  # [추가] Bytecode 관련
  MLIRCallInterfaces       # [추가] FuncOp 관련
  MLIRCastInterfaces       # [추가] Cast 관련
  MLIRFunctionInterfaces   # [추가] FunctionOpInterface 관련
  MLIRInferTypeOpInterface
)