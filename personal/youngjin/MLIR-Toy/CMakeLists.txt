cmake_minimum_required(VERSION 3.20)
project(toy-compiler)

# 패키지 로드 - MLIR/LLVM의 Config.cmake를 찾고 없으면 중지
find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)

# MLIR/LLVM의 커맨드 명령어 사용을 위해 (예: mlir-tablegen)
# 변수들은 위에서 채워짐
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

# 안의 인자는 package 정보에 이미 들어있음
# TableGen.cmake 파일을 로드한다는 의미
include(TableGen)
include(AddMLIR)
include(HandleLLVMOptions)

# 헤더 파일을 어디서 찾을지 알려주는 내용
# 특히 CMAKE_CURRENT_BINARY_DIR이 TableGen으로 생성한 .inc폴더가 위치해있어
# 경로를 찾을 수 있게 설정해주는게 중요
include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_BINARY_DIR}/include
  ${MLIR_INCLUDE_DIRS}
  ${LLVM_INCLUDE_DIRS}
)

# ==============================================================================
# [TableGen] Ops & Dialect (공통) - 컴파일 할 때마다 실행해야하는 명령어들
# ==============================================================================
# Ops.td 파일로 Code Generation
# set은 변수 할당, LLVM_TARGET_DEFINITIONS는 특수 변수
# 아래 줄은 LLVM_TARGET_DEFINITIONS = include/toy/Ops.td 와 같음
set(LLVM_TARGET_DEFINITIONS include/toy/Ops.td)

# 커맨드 명령어 실행 Ops (output_file_name, argument)
mlir_tablegen(include/toy/Ops.h.inc -gen-op-decls)
mlir_tablegen(include/toy/Ops.cpp.inc -gen-op-defs)
mlir_tablegen(include/toy/Dialect.h.inc -gen-dialect-decls)
mlir_tablegen(include/toy/Dialect.cpp.inc -gen-dialect-defs)

# "ToyOpsIncGen" 이라는 이름으로 내부 파일들을 하나로 묶어서 관리(가상 타겟)
# 이제 내부 파일 4개를 ToyOpsIncGen이라는 이름의 의존성 주입 가능
add_custom_target(ToyOpsIncGen DEPENDS
  include/toy/Ops.h.inc
  include/toy/Ops.cpp.inc
  include/toy/Dialect.h.inc
  include/toy/Dialect.cpp.inc
)

# ==============================================================================
# [Chapter 3, TableGen 2] Optimization Patterns .td로 작성한 최적화 코드 생성
# ==============================================================================
set(LLVM_TARGET_DEFINITIONS include/toy/ToyCombine.td)
# -gen-rewriters 옵션: .td를 읽어서 패턴 매칭 C++ 코드를 만들기
mlir_tablegen(include/toy/ToyCombine.inc -gen-rewriters)
add_custom_target(ToyCombineIncGen DEPENDS
  include/toy/ToyCombine.inc
)

# ==============================================================================
# [Chapter 4, TableGen 3] Shape Inference Interface
# ==============================================================================

# ShapeInferenceInterface.td 파일을 읽어서 인터페이스 헤더/구현 코드를 생성
set(LLVM_TARGET_DEFINITIONS include/toy/ShapeInferenceInterface.td)
mlir_tablegen(include/toy/ShapeInferenceOpInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(include/toy/ShapeInferenceOpInterfaces.cpp.inc -gen-op-interface-defs)

add_custom_target(ToyShapeInferenceIncGen DEPENDS
  include/toy/ShapeInferenceOpInterfaces.h.inc
  include/toy/ShapeInferenceOpInterfaces.cpp.inc
)

# ==============================================================================
# [Target 1] Chapter 1: AST 파서 (toyc-ch1)
# ==============================================================================
# 하나의 실행파일을 만드는 명령어
# (output_file_name 소스파일1 소스파일2 ...)
add_executable(toyc-ch1
  toyc.cpp         # 실제 MLIR Dump
  parser/AST.cpp   # AST 구성
)

# 소스코드 안에서 "#ifdef CH1"의 내용만 실행하기 위해 설정
target_compile_definitions(toyc-ch1 PRIVATE CH1)
# LLVMSupport 사용을 위한 link
target_link_libraries(toyc-ch1 PRIVATE LLVMSupport)
# C++ 컴파일러(기본 On)와 LLVM(기본 off) 간의 RTTI를 끄는 것으로 맞추기 (flag no)
target_compile_options(toyc-ch1 PRIVATE -fno-rtti)
  
# ==============================================================================
# [Target 2] Chapter 2: MLIR 생성 (toyc-ch2)
# ==============================================================================
add_executable(toyc-ch2
  toyc.cpp
  parser/AST.cpp
  lib/toy/MLIRGen.cpp
  lib/toy/Dialect.cpp
)

add_dependencies(toyc-ch2 ToyOpsIncGen)
target_compile_definitions(toyc-ch2 PRIVATE CH2) # CH2 정의
# 이건 import 구문에서 찾아 직접 입력해야하는데
# mlir/ir/...을 import 하면 MLIRIR의 규칙으로 여기에 작성하면 된다.
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

add_dependencies(toyc-ch3 ToyOpsIncGen ToyCombineIncGen)
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

# ==============================================================================
# [Target 4] Chapter 4: Inlining (toyc-ch4)
# ==============================================================================

add_executable(toyc-ch4
  toyc.cpp
  parser/AST.cpp
  lib/toy/MLIRGen.cpp
  lib/toy/Dialect.cpp
  lib/toy/ToyCombine.cpp 
  lib/toy/ShapeInferencePass.cpp
)

add_dependencies(toyc-ch4
  ToyOpsIncGen
  ToyCombineIncGen
  ToyShapeInferenceIncGen
)

# CH4 매크로 정의
target_compile_definitions(toyc-ch4 PRIVATE CH4)

target_link_libraries(toyc-ch4 PRIVATE
  MLIRIR MLIRParser MLIRPass MLIRTransforms MLIRAnalysis
  MLIRSideEffectInterfaces MLIRCallInterfaces MLIRCastInterfaces
  MLIRFunctionInterfaces MLIRSupport LLVMSupport
)
target_compile_options(toyc-ch4 PRIVATE -fno-rtti)

# ==============================================================================
# [Target 5] Chapter 5: partial lowering for optimization - Affine
# ==============================================================================
add_executable(toyc-ch5
  toyc.cpp
  parser/AST.cpp
  lib/toy/MLIRGen.cpp
  lib/toy/Dialect.cpp
  lib/toy/ToyCombine.cpp 
  lib/toy/ShapeInferencePass.cpp
  lib/toy/LowerToAffineLoops.cpp
)

add_dependencies(toyc-ch5
  ToyOpsIncGen
  ToyCombineIncGen
  ToyShapeInferenceIncGen
)

# CH5 매크로 정의
target_compile_definitions(toyc-ch5 PRIVATE CH5)

target_link_libraries(toyc-ch5 PRIVATE
  MLIRIR MLIRParser MLIRPass MLIRTransforms MLIRAnalysis
  MLIRSideEffectInterfaces MLIRCallInterfaces MLIRCastInterfaces
  MLIRFunctionInterfaces MLIRSupport LLVMSupport
  MLIRAffineDialect MLIRArithDialect MLIRFuncDialect MLIRMemRefDialect
  MLIRAffineTransforms MLIRFuncAllExtensions
)
target_compile_options(toyc-ch5 PRIVATE -fno-rtti)


# ==============================================================================
# [Target 6] Chapter 6: Lower to LLVM & JIT
# ==============================================================================

add_executable(toyc-ch6
  toyc.cpp
  parser/AST.cpp
  lib/toy/MLIRGen.cpp
  lib/toy/Dialect.cpp
  lib/toy/ToyCombine.cpp 
  lib/toy/ShapeInferencePass.cpp
  lib/toy/LowerToAffineLoops.cpp
  lib/toy/LowerToLLVM.cpp
)

add_dependencies(toyc-ch6
  ToyOpsIncGen
  ToyCombineIncGen
  ToyShapeInferenceIncGen
)

# [중요] CH5와 CH6를 모두 정의해야 전체 파이프라인이 활성화됩니다!
target_compile_definitions(toyc-ch6 PRIVATE CH5 CH6)

target_link_libraries(toyc-ch6
PRIVATE
    # [1] 기본 MLIR 코어 & 인터페이스
    MLIRAnalysis
    MLIRIR
    MLIRParser
    MLIRPass
    MLIRTransforms
    MLIRSupport
    MLIRSideEffectInterfaces
    MLIRCallInterfaces
    MLIRCastInterfaces
    MLIRFunctionInterfaces

    # [2] 하강(Lowering) 및 변환 패스 (CH5 필수)
    MLIRAffineToStandard
    MLIRSCFToControlFlow
    MLIRArithToLLVM
    MLIRFuncToLLVM
    MLIRMemRefToLLVM
    MLIRControlFlowToLLVM

    # [3] LLVM Dialect 및 관련 기능
    MLIRLLVMDialect
    MLIRLLVMCommonConversion
    MLIRLLVMIRTransforms  # <--- [중요] LLVM 인라이닝 인터페이스

    # [4] LLVM IR 번역 (Translation)
    MLIRTargetLLVMIRExport
    MLIRLLVMToLLVMIRTranslation
    MLIRBuiltinToLLVMIRTranslation

    # [5] JIT 실행 엔진
    MLIRExecutionEngine

    # [6] 방언 및 확장 기능 (최신 버전 대응)
    MLIRAffineDialect
    MLIRArithDialect
    MLIRFuncDialect
    MLIRMemRefDialect
    MLIRSCFDialect
    MLIRControlFlowDialect
    MLIRFuncAllExtensions # <--- [중요] Func 인라이닝 확장
)

target_compile_options(toyc-ch6 PRIVATE -fno-rtti)