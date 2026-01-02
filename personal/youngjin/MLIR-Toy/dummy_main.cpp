// dummy_main.cpp (디버깅 버전)
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "llvm/Support/raw_ostream.h"

#include "toy/Dialect.h"

using namespace mlir;
using namespace mlir::toy; // 네임스페이스 주의!

int main() {
  llvm::outs() << "[DEBUG] 1. Context 생성 중...\n";
  MLIRContext context;
  context.loadDialect<ToyDialect>();

  llvm::outs() << "[DEBUG] 2. Module 생성 중...\n";
  OpBuilder builder(&context);
  ModuleOp module = ModuleOp::create(builder.getUnknownLoc());
  
  // [핵심] 삽입 지점 설정 (이게 없으면 공중분해됨)
  if (!module.getBody()) {
    llvm::errs() << "[ERROR] 모듈에 Body가 없습니다!\n";
    return 1;
  }
  builder.setInsertionPointToEnd(module.getBody());

  llvm::outs() << "[DEBUG] 3. ConstantOp(a) 생성 중...\n";
  Location loc = builder.getUnknownLoc();
  Value a = builder.create<ConstantOp>(loc, 10.0);
  
  llvm::outs() << "[DEBUG] 4. ConstantOp(b) 생성 중...\n";
  Value b = builder.create<ConstantOp>(loc, 20.0);

  llvm::outs() << "[DEBUG] 5. AddOp(c) 생성 중...\n";
  Value c = builder.create<AddOp>(loc, a, b).getResult();

  llvm::outs() << "[DEBUG] 6. PrintOp 생성 중...\n";
  // t를 출력함
  builder.create<PrintOp>(loc, c);

  llvm::outs() << "[DEBUG] 7. 결과 출력:\n";
  llvm::outs() << "------------------------------------------------------\n";
  module.dump();
  llvm::outs() << "------------------------------------------------------\n";

  return 0;
}