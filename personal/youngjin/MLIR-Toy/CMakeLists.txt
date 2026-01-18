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

# ==============================================================================
# [TableGen] Ops & Dialect (공통)
# ==============================================================================
set(LLVM_TARGET_DEFINITIONS include/toy/Ops.td)
mlir_tablegen(include/toy/Ops.h.inc -gen-op-decls)
mlir_tablegen(include/toy/Ops.cpp.inc -gen-op-defs)
mlir_tablegen(include/toy/Dialect.h.inc -gen-dialect-decls)
mlir_tablegen(include/toy/Dialect.cpp.inc -gen-dialect-defs)

add_custom_target(ToyOpsIncGen DEPENDS
  include/toy/Ops.h.inc
  include/toy/Ops.cpp.inc
  include/toy/Dialect.h.inc
  include/toy/Dialect.cpp.inc
)

# ==============================================================================
# [TableGen 2] Optimization Patterns .td로 작성한 최적화 코드 생성
# ==============================================================================
set(LLVM_TARGET_DEFINITIONS include/toy/ToyCombine.td)
# -gen-rewriters 옵션: .td를 읽어서 패턴 매칭 C++ 코드를 만들어라!
mlir_tablegen(include/toy/ToyCombine.inc -gen-rewriters)

add_custom_target(ToyCombineIncGen DEPENDS
  include/toy/ToyCombine.inc
)

# ==============================================================================
# [Target 1] Chapter 1: AST 파서 (toyc-ch1)
# ==============================================================================
add_executable(toyc-ch1
  toyc.cpp
  parser/AST.cpp
)
# [중요] 소스코드 안에서 "#ifdef CH1"으로 구분하기 위해 정의
target_compile_definitions(toyc-ch1 PRIVATE CH1) 
target_link_libraries(toyc-ch1 PRIVATE LLVMSupport)
target_compile_options(toyc-ch1 PRIVATE -fno-rtti)


# ==============================================================================
# [Target 2] Chapter 2: MLIR 생성 (toyc-ch2)
# ==============================================================================
add_executable(toyc-ch2
  toyc.cpp
  parser/AST.cpp
  lib/toy/MLIRGen.cpp
  lib/toy/Dialect.cpp
  lib/toy/ToyCombine.cpp
)
add_dependencies(toyc-ch2 ToyOpsIncGen)
target_compile_definitions(toyc-ch2 PRIVATE CH2) # CH2 정의
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


# ==============================================================================
# [Target 3] Chapter 3: 최적화 (toyc-ch3)
# ==============================================================================
add_executable(toyc-ch3
  toyc.cpp
  parser/AST.cpp
  lib/toy/MLIRGen.cpp
  lib/toy/Dialect.cpp
  lib/toy/ToyCombine.cpp 
)
add_dependencies(toyc-ch3 ToyOpsIncGen)
target_compile_definitions(toyc-ch3 PRIVATE CH3) # CH3 정의
target_link_libraries(toyc-ch3 PRIVATE
  MLIRIR
  MLIRParser
  MLIRPass                
  MLIRTransforms          
  MLIRAnalysis            
  MLIRSideEffectInterfaces
  MLIRCallInterfaces
  MLIRCastInterfaces
  MLIRFunctionInterfaces
  MLIRSupport
  LLVMSupport
)
target_compile_options(toyc-ch3 PRIVATE -fno-rtti)