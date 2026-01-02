// lib/toy/Dialect.cpp
#include "toy/Dialect.h"

// [핵심] 방언 구현부 포함 (.inc 파일)
#include "toy/Dialect.cpp.inc"

// [핵심] 네임스페이스 사용
// 이제 ToyDialect는 mlir::toy 안에 있습니다.
using namespace mlir;
using namespace mlir::toy;

// ToyDialect 초기화 함수
// (이미 using namespace mlir::toy를 했으므로 'ToyDialect::' 로 충분합니다)
void ToyDialect::initialize() {
  // ConstantOp, AddOp 등 등록
  addOperations<
#define GET_OP_LIST
#include "toy/Ops.cpp.inc"
      >();
}