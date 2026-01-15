cmake_minimum_required(VERSION 3.20)
project(toy-compiler)

# 1. 패키지 로드
find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_BINARY_DIR}/include
  ${MLIR_INCLUDE_DIRS}
  ${LLVM_INCLUDE_DIRS}
)

# ----------------------------------------------------------------------
# [Chapter 2 핵심] TableGen: .td 파일을 읽어 C++(.inc) 코드를 자동 생성
# ----------------------------------------------------------------------
set(LLVM_TARGET_DEFINITIONS include/toy/Ops.td)
mlir_tablegen(include/toy/Ops.h.inc -gen-op-decls)
mlir_tablegen(include/toy/Ops.cpp.inc -gen-op-defs)
mlir_tablegen(include/toy/Dialect.h.inc -gen-dialect-decls)
mlir_tablegen(include/toy/Dialect.cpp.inc -gen-dialect-defs)

add_custom_target(ToyCh2OpsIncGen DEPENDS
  include/toy/Ops.h.inc
  include/toy/Ops.cpp.inc
  include/toy/Dialect.h.inc
  include/toy/Dialect.cpp.inc
)


# ----------------------------------------------------------------------
# [Target 1] Chapter 1: AST 파서 (기존 유지)
# ----------------------------------------------------------------------
add_executable(toyc-ch1
  toyc.cpp
  parser/AST.cpp
)
target_link_libraries(toyc-ch1 PRIVATE LLVMSupport)
target_compile_options(toyc-ch1 PRIVATE -fno-rtti)


# ----------------------------------------------------------------------
# [Target 2] Chapter 2: MLIR 생성 (NEW!)
# ----------------------------------------------------------------------
add_executable(toyc-ch2
  toyc.cpp              # 메인 (업데이트 필요)
  parser/AST.cpp        # AST 덤프
  lib/toy/MLIRGen.cpp   # [중요] AST -> MLIR 변환 로직 (파일 확인 필요!)
  lib/toy/Dialect.cpp   # MLIR Dialect 정의
  # lib/toy/Ops.cpp       # MLIR Operation 정의
)

# .inc 파일 생성 후 빌드하도록 의존성 설정
add_dependencies(toyc-ch2 ToyCh2OpsIncGen)

# MLIR 라이브러리 링크
target_link_libraries(toyc-ch2 PRIVATE
  MLIRIR
  MLIRParser
  MLIRSideEffectInterfaces
  MLIRCallInterfaces
  MLIRCastInterfaces
  MLIRFunctionInterfaces
  MLIRSupport
  LLVMSupport
)

target_compile_options(toyc-ch2 PRIVATE -fno-rtti)