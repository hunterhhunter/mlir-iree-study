//===- Dialect.cpp - Toy IR Dialect registration in MLIR ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// 이 파일은 Toy IR을 위한 방언(dialect)을 구현합니다: 커스텀 타입 파싱 및 
// 연산 검증(verification).
//
//===----------------------------------------------------------------------===//

#include "toy/Dialect.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <string>

using namespace mlir;
using namespace mlir::toy;

#include "toy/Dialect.cpp.inc"

//===----------------------------------------------------------------------===//
// ToyInlinerInterface
//===----------------------------------------------------------------------===//

/// 이 클래스는 Toy 연산들의 인라이닝을 처리하기 위한 인터페이스를 정의합니다.
struct ToyInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  //===--------------------------------------------------------------------===//
  // Analysis Hooks
  //===--------------------------------------------------------------------===//

  /// Toy 내의 모든 호출 연산은 인라인될 수 있습니다.
  bool isLegalToInline(Operation *call, Operation *callable,
                       bool wouldBeCloned) const final {
    return true;
  }

  /// Toy 내의 모든 연산은 인라인될 수 있습니다.
  bool isLegalToInline(Operation *, Region *, bool, IRMapping &) const final {
    return true;
  }

  // Toy 내의 모든 함수는 인라인될 수 있습니다.
  bool isLegalToInline(Region *, Region *, bool, IRMapping &) const final {
    return true;
  }

  //===--------------------------------------------------------------------===//
  // Transformation Hooks
  //===--------------------------------------------------------------------===//

  /// 인라인된 터미네이터(toy.return)를 필요에 따라 새로운 연산으로 교체하여 처리합니다.
  void handleTerminator(Operation *op, ValueRange valuesToRepl) const final {
    // 여기서는 "toy.return"만 처리하면 됩니다.
    auto returnOp = cast<ReturnOp>(op);

    // 값들을 반환 피연산자들로 직접 교체합니다.
    assert(returnOp.getNumOperands() == valuesToRepl.size());
    for (const auto &it : llvm::enumerate(returnOp.getOperands()))
      valuesToRepl[it.index()].replaceAllUsesWith(it.value());
  }

  /// 이 방언(dialect)의 호출과 호출 가능한 영역(region) 사이의 타입 불일치에 대한 
  /// 변환을 구체화(materialize)하려고 시도합니다. 이 메서드는 'input'을 유일한 
  /// 피연산자로 받고, 'resultType'을 단일 결과로 생성하는 연산을 만들어야 합니다.
  /// 만약 변환을 생성할 수 없다면, nullptr를 반환해야 합니다.
  Operation *materializeCallConversion(OpBuilder &builder, Value input,
                                       Type resultType,
                                       Location conversionLoc) const final {
    return CastOp::create(builder, conversionLoc, resultType, input);
  }
};

//===----------------------------------------------------------------------===//
// ToyDialect
//===----------------------------------------------------------------------===//

/// 방언 초기화, 인스턴스는 컨텍스트에 의해 소유됩니다. 이 시점이 방언의 
/// 타입과 연산들이 등록되는 지점입니다.
void ToyDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "toy/Ops.cpp.inc"
      >();
  addInterfaces<ToyInlinerInterface>();
}

//===----------------------------------------------------------------------===//
// Toy Operations
//===----------------------------------------------------------------------===//

/// 이항 연산을 위한 일반화된 파서입니다. 아래의 'printBinaryOp'의 다양한 
/// 형태를 파싱합니다.
static mlir::ParseResult parseBinaryOp(mlir::OpAsmParser &parser,
                                       mlir::OperationState &result) {
  SmallVector<mlir::OpAsmParser::UnresolvedOperand, 2> operands;
  SMLoc operandsLoc = parser.getCurrentLocation();
  Type type;
  if (parser.parseOperandList(operands, /*requiredOperandCount=*/2) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(type))
    return mlir::failure();

  // 타입이 함수 타입인 경우, 이 연산의 입력 및 결과 타입을 포함합니다.
  if (FunctionType funcType = llvm::dyn_cast<FunctionType>(type)) {
    if (parser.resolveOperands(operands, funcType.getInputs(), operandsLoc,
                               result.operands))
      return mlir::failure();
    result.addTypes(funcType.getResults());
    return mlir::success();
  }

  // 그렇지 않으면, 파싱된 타입은 피연산자와 결과 모두의 타입입니다.
  if (parser.resolveOperands(operands, type, result.operands))
    return mlir::failure();
  result.addTypes(type);
  return mlir::success();
}

/// 이항 연산을 위한 일반화된 프린터입니다. 모든 타입이 일치하는지 여부에 따라 
/// 두 가지 다른 형태로 출력합니다.
static void printBinaryOp(mlir::OpAsmPrinter &printer, mlir::Operation *op) {
  printer << " " << op->getOperands();
  printer.printOptionalAttrDict(op->getAttrs());
  printer << " : ";

  // 모든 타입이 동일하면 타입을 직접 출력합니다.
  Type resultType = *op->result_type_begin();
  if (llvm::all_of(op->getOperandTypes(),
                   [=](Type type) { return type == resultType; })) {
    printer << resultType;
    return;
  }

  // 그렇지 않으면 함수형 타입으로 출력합니다.
  printer.printFunctionalType(op->getOperandTypes(), op->getResultTypes());
}

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

/// 상수 연산을 생성합니다.
/// 빌더가 인자로 전달되며, 연산을 생성하기 위해 채워야 하는 상태(state)도 
/// 함께 전달됩니다.
void ConstantOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                       double value) {
  auto dataType = RankedTensorType::get({}, builder.getF64Type());
  auto dataAttribute = DenseElementsAttr::get(dataType, value);
  ConstantOp::build(builder, state, dataType, dataAttribute);
}

/// 'OpAsmPrinter' 클래스는 구두점, 속성, 피연산자, 타입 등을 파싱하기 위한 
/// 메서드 모음을 제공합니다. 각 메서드는 `ParseResult`를 반환합니다. 이 클래스는 
/// 실패 시 불리언 `true` 값으로, 성공 시 `false`로 변환될 수 있는 `LogicalResult`의 
/// 래퍼입니다. 이를 통해 일련의 파서 규칙을 쉽게 연결할 수 있습니다. 
/// 이러한 규칙은 앞서 설명한 `build` 메서드와 유사하게 `mlir::OperationState`를 
/// 채우는 데 사용됩니다.
mlir::ParseResult ConstantOp::parse(mlir::OpAsmParser &parser,
                                    mlir::OperationState &result) {
  mlir::DenseElementsAttr value;
  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseAttribute(value, "value", result.attributes))
    return failure();

  result.addTypes(value.getType());
  return success();
}

/// 'OpAsmPrinter' 클래스는 문자열, 속성, 피연산자, 타입 등을 포맷팅할 수 있는 
/// 스트림입니다.
void ConstantOp::print(mlir::OpAsmPrinter &printer) {
  printer << " ";
  printer.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"value"});
  printer << getValue();
}

/// 상수 연산에 대한 검증기(Verifier)입니다. 이는 op 정의의 
/// `let hasVerifier = 1`에 해당합니다.
llvm::LogicalResult ConstantOp::verify() {
  // 상수의 반환 타입이 unranked 텐서가 아닌 경우, 모양(shape)은 데이터를 
  // 담고 있는 속성의 모양과 일치해야 합니다.
  auto resultType =
      llvm::dyn_cast<mlir::RankedTensorType>(getResult().getType());
  if (!resultType)
    return success();

  // 속성 타입의 랭크가 상수 결과 타입의 랭크와 일치하는지 확인합니다.
  auto attrType = llvm::cast<mlir::RankedTensorType>(getValue().getType());
  if (attrType.getRank() != resultType.getRank()) {
    return emitOpError("return type must match the one of the attached value "
                       "attribute: ")
           << attrType.getRank() << " != " << resultType.getRank();
  }

  // 두 타입 간의 각 차원이 일치하는지 확인합니다.
  for (int dim = 0, dimE = attrType.getRank(); dim < dimE; ++dim) {
    if (attrType.getShape()[dim] != resultType.getShape()[dim]) {
      return emitOpError(
                 "return type shape mismatches its attribute at dimension ")
             << dim << ": " << attrType.getShape()[dim]
             << " != " << resultType.getShape()[dim];
    }
  }
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//

void AddOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                  mlir::Value lhs, mlir::Value rhs) {
  state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
  state.addOperands({lhs, rhs});
}

mlir::ParseResult AddOp::parse(mlir::OpAsmParser &parser,
                               mlir::OperationState &result) {
  return parseBinaryOp(parser, result);
}

void AddOp::print(mlir::OpAsmPrinter &p) { printBinaryOp(p, *this); }

/// AddOp의 출력 모양을 추론합니다. 이는 shape inference 인터페이스에 의해 요구됩니다.
void AddOp::inferShapes() { getResult().setType(getLhs().getType()); }

//===----------------------------------------------------------------------===//
// CastOp
//===----------------------------------------------------------------------===//

/// CastOp의 출력 모양을 추론합니다. 이는 shape inference 인터페이스에 의해 요구됩니다.
void CastOp::inferShapes() { getResult().setType(getInput().getType()); }

/// 주어진 입력 및 결과 타입 집합이 이 cast 연산과 호환되면 true를 반환합니다.
/// 이는 `CastOpInterface`가 이 연산을 검증하고 추가적인 유틸리티를 제공하기 위해 요구됩니다.
bool CastOp::areCastCompatible(TypeRange inputs, TypeRange outputs) {
  if (inputs.size() != 1 || outputs.size() != 1)
    return false;
  // 입력은 반드시 동일한 요소 타입을 가진 텐서여야 합니다.
  TensorType input = llvm::dyn_cast<TensorType>(inputs.front());
  TensorType output = llvm::dyn_cast<TensorType>(outputs.front());
  if (!input || !output || input.getElementType() != output.getElementType())
    return false;
  // 만약 두 타입 모두 랭크(차원)가 있다면, 모양이 일치해야 합니다.
  return !input.hasRank() || !output.hasRank() || input == output;
}

//===----------------------------------------------------------------------===//
// FuncOp
//===----------------------------------------------------------------------===//

void FuncOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                   llvm::StringRef name, mlir::FunctionType type,
                   llvm::ArrayRef<mlir::NamedAttribute> attrs) {
  // FunctionOpInterface는 FuncOp의 상태를 채우고 진입 블록(entry block)을 
  // 생성하는 편리한 `build` 메서드를 제공합니다.
  buildWithEntryBlock(builder, state, name, type, attrs, type.getInputs());
}

mlir::ParseResult FuncOp::parse(mlir::OpAsmParser &parser,
                                mlir::OperationState &result) {
  // FunctionOpInterface에서 제공하는 유틸리티 메서드에 함수 연산 파싱을 위임합니다.
  auto buildFuncType =
      [](mlir::Builder &builder, llvm::ArrayRef<mlir::Type> argTypes,
         llvm::ArrayRef<mlir::Type> results,
         mlir::function_interface_impl::VariadicFlag,
         std::string &) { return builder.getFunctionType(argTypes, results); };

  return mlir::function_interface_impl::parseFunctionOp(
      parser, result, /*allowVariadic=*/false,
      getFunctionTypeAttrName(result.name), buildFuncType,
      getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name));
}

void FuncOp::print(mlir::OpAsmPrinter &p) {
  // FunctionOpInterface에서 제공하는 유틸리티 메서드에 함수 연산 출력을 위임합니다.
  mlir::function_interface_impl::printFunctionOp(
      p, *this, /*isVariadic=*/false, getFunctionTypeAttrName(),
      getArgAttrsAttrName(), getResAttrsAttrName());
}

//===----------------------------------------------------------------------===//
// GenericCallOp
//===----------------------------------------------------------------------===//

void GenericCallOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                          StringRef callee, ArrayRef<mlir::Value> arguments) {
  // Generic call은 초기에 항상 unranked 텐서를 반환합니다.
  state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
  state.addOperands(arguments);
  state.addAttribute("callee",
                     mlir::SymbolRefAttr::get(builder.getContext(), callee));
}

/// 제네릭 호출 연산의 피호출자(callee)를 반환합니다. 
/// 이는 호출 인터페이스(call interface)에서 요구됩니다.
CallInterfaceCallable GenericCallOp::getCallableForCallee() {
  return (*this)->getAttrOfType<SymbolRefAttr>("callee");
}

/// 제네릭 호출 연산의 피호출자를 설정합니다. 
/// 이는 호출 인터페이스에서 요구됩니다.
void GenericCallOp::setCalleeFromCallable(CallInterfaceCallable callee) {
  (*this)->setAttr("callee", cast<SymbolRefAttr>(callee));
}

/// 호출된 함수로 전달되는 인자 피연산자들을 가져옵니다. 
/// 이는 호출 인터페이스에서 요구됩니다.
Operation::operand_range GenericCallOp::getArgOperands() { return getInputs(); }

/// 호출된 함수로 전달되는 인자 피연산자들을 변경 가능한 범위(mutable range)로 가져옵니다. 
/// 이는 호출 인터페이스에서 요구됩니다.
MutableOperandRange GenericCallOp::getArgOperandsMutable() {
  return getInputsMutable();
}

//===----------------------------------------------------------------------===//
// MulOp
//===----------------------------------------------------------------------===//

void MulOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                  mlir::Value lhs, mlir::Value rhs) {
  state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
  state.addOperands({lhs, rhs});
}

mlir::ParseResult MulOp::parse(mlir::OpAsmParser &parser,
                               mlir::OperationState &result) {
  return parseBinaryOp(parser, result);
}

void MulOp::print(mlir::OpAsmPrinter &p) { printBinaryOp(p, *this); }

/// MulOp의 출력 모양을 추론합니다. 이는 shape inference 인터페이스에 의해 요구됩니다.
void MulOp::inferShapes() { getResult().setType(getLhs().getType()); }

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

llvm::LogicalResult ReturnOp::verify() {
  // 연산 정의에 붙은 'HasParent' 트레이트 덕분에 부모 연산이 함수임을 알 수 있습니다.
  auto function = cast<FuncOp>((*this)->getParentOp());

  /// ReturnOp는 최대 하나의 선택적 피연산자만 가질 수 있습니다.
  if (getNumOperands() > 1)
    return emitOpError() << "expects at most 1 return operand";

  // 피연산자의 수와 타입은 함수 시그니처와 일치해야 합니다.
  const auto &results = function.getFunctionType().getResults();
  if (getNumOperands() != results.size())
    return emitOpError() << "does not return the same number of values ("
                         << getNumOperands() << ") as the enclosing function ("
                         << results.size() << ")";

  // 연산에 입력이 없으면 완료입니다.
  if (!hasOperand())
    return mlir::success();

  auto inputType = *operand_type_begin();
  auto resultType = results.front();

  // 함수의 결과 타입이 피연산자 타입과 일치하는지 확인합니다.
  if (inputType == resultType ||
      llvm::isa<mlir::UnrankedTensorType>(inputType) ||
      llvm::isa<mlir::UnrankedTensorType>(resultType))
    return mlir::success();

  return emitError() << "type of return operand (" << inputType
                     << ") doesn't match function result type (" << resultType
                     << ")";
}

//===----------------------------------------------------------------------===//
// TransposeOp
//===----------------------------------------------------------------------===//

void TransposeOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                        mlir::Value value) {
  state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
  state.addOperands(value);
}

void TransposeOp::inferShapes() {
  auto arrayTy = llvm::cast<RankedTensorType>(getOperand().getType());
  SmallVector<int64_t, 2> dims(llvm::reverse(arrayTy.getShape()));
  getResult().setType(RankedTensorType::get(dims, arrayTy.getElementType()));
}

llvm::LogicalResult TransposeOp::verify() {
  auto inputType = llvm::dyn_cast<RankedTensorType>(getOperand().getType());
  auto resultType = llvm::dyn_cast<RankedTensorType>(getType());
  if (!inputType || !resultType)
    return mlir::success();

  auto inputShape = inputType.getShape();
  if (!std::equal(inputShape.begin(), inputShape.end(),
                  resultType.getShape().rbegin())) {
    return emitError()
           << "expected result shape to be a transpose of the input";
  }
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "toy/Ops.cpp.inc"