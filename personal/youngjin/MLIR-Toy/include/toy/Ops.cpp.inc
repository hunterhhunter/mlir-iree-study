/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Definitions                                                             *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: Ops.td                                                               *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_OP_LIST
#undef GET_OP_LIST

::mlir::toy::AddOp,
::mlir::toy::ConstantOp,
::mlir::toy::PrintOp
#endif // GET_OP_LIST

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// Local Utility Method Definitions
//===----------------------------------------------------------------------===//


static ::llvm::LogicalResult __mlir_ods_local_type_constraint_Ops1(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((::llvm::isa<::mlir::TensorType>(type))) && ([](::mlir::Type elementType) { return (elementType.isF64()); }(::llvm::cast<::mlir::ShapedType>(type).getElementType())))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be tensor of 64-bit float values, but got " << type;
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_Ops1(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  if (attr && !((::llvm::isa<::mlir::DenseFPElementsAttr>(attr) &&::llvm::cast<::mlir::DenseElementsAttr>(attr).getType().getElementType().isF64())))
    return emitError() << "attribute '" << attrName
        << "' failed to satisfy constraint: 64-bit float elements attribute";
  return ::mlir::success();
}
static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_Ops1(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_Ops1(attr, attrName, [op]() {
    return op->emitOpError();
  });
}
namespace mlir::toy {


//===----------------------------------------------------------------------===//
// ::mlir::toy::AddOp definitions
//===----------------------------------------------------------------------===//

namespace detail {

AddOpGenericAdaptorBase::AddOpGenericAdaptorBase(AddOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}


} // namespace detail
AddOpAdaptor::AddOpAdaptor(AddOp op) : AddOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult AddOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::LogicalResult AddOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }
      return ::mlir::success();
}

::mlir::Attribute AddOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code AddOp::computePropertiesHash(const Properties &prop) {
  using llvm::hash_value;
  return llvm::hash_combine();
}

std::optional<mlir::Attribute> AddOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
  return std::nullopt;
}

void AddOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
}

void AddOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
}

// 발표 - 검증 함수
::llvm::LogicalResult AddOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    return ::mlir::success();
}

void AddOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes(resultType0);
}

AddOp AddOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type resultType0, ::mlir::Value lhs, ::mlir::Value rhs) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultType0)>(resultType0), std::forward<decltype(lhs)>(lhs), std::forward<decltype(rhs)>(rhs));
  auto __res__ = ::llvm::dyn_cast<AddOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

AddOp AddOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type resultType0, ::mlir::Value lhs, ::mlir::Value rhs) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultType0)>(resultType0), std::forward<decltype(lhs)>(lhs), std::forward<decltype(rhs)>(rhs));
}

void AddOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(AddOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      odsState.getRawProperties(),
                      odsState.regions, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::mlir::detail::reportFatalInferReturnTypesError(odsState);
        
}

AddOp AddOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value lhs, ::mlir::Value rhs) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(lhs)>(lhs), std::forward<decltype(rhs)>(rhs));
  auto __res__ = ::llvm::dyn_cast<AddOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

AddOp AddOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value lhs, ::mlir::Value rhs) {
  return create(builder, builder.getLoc(), std::forward<decltype(lhs)>(lhs), std::forward<decltype(rhs)>(rhs));
}

void AddOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

AddOp AddOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(lhs)>(lhs), std::forward<decltype(rhs)>(rhs));
  auto __res__ = ::llvm::dyn_cast<AddOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

AddOp AddOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(lhs)>(lhs), std::forward<decltype(rhs)>(rhs));
}

void AddOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

AddOp AddOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(attributes)>(attributes));
  auto __res__ = ::llvm::dyn_cast<AddOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

AddOp AddOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(attributes)>(attributes));
}

void AddOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(AddOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.getRawProperties(),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

AddOp AddOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(operands)>(operands), std::forward<decltype(attributes)>(attributes));
  auto __res__ = ::llvm::dyn_cast<AddOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

AddOp AddOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  return create(builder, builder.getLoc(), std::forward<decltype(operands)>(operands), std::forward<decltype(attributes)>(attributes));
}

void AddOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.useProperties(const_cast<Properties&>(properties));
  odsState.addAttributes(discardableAttributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

AddOp AddOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(properties)>(properties), std::forward<decltype(discardableAttributes)>(discardableAttributes));
  auto __res__ = ::llvm::dyn_cast<AddOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

AddOp AddOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(properties)>(properties), std::forward<decltype(discardableAttributes)>(discardableAttributes));
}

void AddOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.useProperties(const_cast<Properties &>(properties));
  odsState.addAttributes(discardableAttributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(AddOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.getRawProperties(),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

AddOp AddOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(operands)>(operands), std::forward<decltype(properties)>(properties), std::forward<decltype(discardableAttributes)>(discardableAttributes));
  auto __res__ = ::llvm::dyn_cast<AddOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

AddOp AddOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  return create(builder, builder.getLoc(), std::forward<decltype(operands)>(operands), std::forward<decltype(properties)>(properties), std::forward<decltype(discardableAttributes)>(discardableAttributes));
}

::llvm::LogicalResult AddOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Ops1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Ops1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Ops1(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult AddOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::llvm::LogicalResult AddOp::inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes) {
  inferredReturnTypes.resize(1);
  ::mlir::Builder odsBuilder(context);
  if (operands.size() <= 0)
    return ::mlir::failure();
  ::mlir::Type odsInferredType0 = operands[0].getType();
  inferredReturnTypes[0] = odsInferredType0;
  return ::mlir::success();
}

::llvm::LogicalResult AddOp::setPropertiesFromParsedAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }
  // keep track of used keys in the input dictionary to be able to error out
  // if there are some unknown ones.
  ::mlir::DenseSet<::mlir::StringAttr> usedKeys;
  ::mlir::MLIRContext *ctx = dict.getContext();
  (void)ctx;

  for (::mlir::NamedAttribute attr : dict) {
    if (!usedKeys.contains(attr.getName()))
      return emitError() << "unknown key '" << attr.getName() <<
          "' when parsing properties dictionary";
  }
  return ::mlir::success();
}


} // namespace mlir::toy
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::toy::AddOp)

namespace mlir::toy {


//===----------------------------------------------------------------------===//
// ::mlir::toy::ConstantOp definitions
//===----------------------------------------------------------------------===//

namespace detail {

ConstantOpGenericAdaptorBase::ConstantOpGenericAdaptorBase(ConstantOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

::mlir::DenseElementsAttr ConstantOpGenericAdaptorBase::getValue() {
  auto attr = getValueAttr();
  return attr;
}


} // namespace detail
ConstantOpAdaptor::ConstantOpAdaptor(ConstantOp op) : ConstantOpGenericAdaptor(op->getOperands(), op) {}

// 발표 - 타입 검증 F64
::llvm::LogicalResult ConstantOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_value = getProperties().value; (void)tblgen_value;
  if (!tblgen_value) return emitError(loc, "'toy.constant' op requires attribute 'value'");

  if (tblgen_value && !((::llvm::isa<::mlir::DenseFPElementsAttr>(tblgen_value) &&::llvm::cast<::mlir::DenseElementsAttr>(tblgen_value).getType().getElementType().isF64())))
    return emitError(loc, "'toy.constant' op attribute 'value' failed to satisfy constraint: 64-bit float elements attribute");
  return ::mlir::success();
}

::llvm::LogicalResult ConstantOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
    auto &propStorage = prop.value;
       auto attr = dict.get("value");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `value` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute ConstantOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.value;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("value",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code ConstantOp::computePropertiesHash(const Properties &prop) {
  using llvm::hash_value;
  return llvm::hash_combine(
    llvm::hash_value(prop.value.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> ConstantOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "value")
      return prop.value;
  return std::nullopt;
}

void ConstantOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "value") {
       prop.value = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.value)>>(value);
       return;
    }
}

void ConstantOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.value) attrs.append("value", prop.value);
}

::llvm::LogicalResult ConstantOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getValueAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_Ops1(attr, "value", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::llvm::LogicalResult ConstantOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readAttribute(prop.value)))
    return ::mlir::failure();
  return ::mlir::success();
}

void ConstantOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;
  writer.writeAttribute(prop.value);
}

::mlir::DenseElementsAttr ConstantOp::getValue() {
  auto attr = getValueAttr();
  return attr;
}

void ConstantOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, mlir::DenseElementsAttr value) {
      build(odsBuilder, odsState, value.getType(), value);
    
}

ConstantOp ConstantOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, mlir::DenseElementsAttr value) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(value)>(value));
  auto __res__ = ::llvm::dyn_cast<ConstantOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

ConstantOp ConstantOp::create(::mlir::ImplicitLocOpBuilder &builder, mlir::DenseElementsAttr value) {
  return create(builder, builder.getLoc(), std::forward<decltype(value)>(value));
}

ConstantOp ConstantOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, double value) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(value)>(value));
  auto __res__ = ::llvm::dyn_cast<ConstantOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

ConstantOp ConstantOp::create(::mlir::ImplicitLocOpBuilder &builder, double value) {
  return create(builder, builder.getLoc(), std::forward<decltype(value)>(value));
}

void ConstantOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::DenseElementsAttr value) {
  odsState.getOrAddProperties<Properties>().value = value;
  odsState.addTypes(resultType0);
}

ConstantOp ConstantOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type resultType0, ::mlir::DenseElementsAttr value) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultType0)>(resultType0), std::forward<decltype(value)>(value));
  auto __res__ = ::llvm::dyn_cast<ConstantOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

ConstantOp ConstantOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type resultType0, ::mlir::DenseElementsAttr value) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultType0)>(resultType0), std::forward<decltype(value)>(value));
}

void ConstantOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::DenseElementsAttr value) {
  odsState.getOrAddProperties<Properties>().value = value;
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

ConstantOp ConstantOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::DenseElementsAttr value) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(value)>(value));
  auto __res__ = ::llvm::dyn_cast<ConstantOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

ConstantOp ConstantOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::DenseElementsAttr value) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(value)>(value));
}

void ConstantOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);

  if (!attributes.empty()) {
    ::mlir::OpaqueProperties properties =
      &odsState.getOrAddProperties<ConstantOp::Properties>();
    std::optional<::mlir::RegisteredOperationName> info =
      odsState.name.getRegisteredInfo();
    if (failed(info->setOpPropertiesFromAttribute(odsState.name, properties,
        odsState.attributes.getDictionary(odsState.getContext()), nullptr)))
      ::llvm::report_fatal_error("Property conversion failed.");
  }
}

ConstantOp ConstantOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(attributes)>(attributes));
  auto __res__ = ::llvm::dyn_cast<ConstantOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

ConstantOp ConstantOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(attributes)>(attributes));
}

void ConstantOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.useProperties(const_cast<Properties&>(properties));
  odsState.addAttributes(discardableAttributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

ConstantOp ConstantOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(properties)>(properties), std::forward<decltype(discardableAttributes)>(discardableAttributes));
  auto __res__ = ::llvm::dyn_cast<ConstantOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

ConstantOp ConstantOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(properties)>(properties), std::forward<decltype(discardableAttributes)>(discardableAttributes));
}

::llvm::LogicalResult ConstantOp::verifyInvariantsImpl() {
  auto tblgen_value = getProperties().value; (void)tblgen_value;
  if (!tblgen_value) return emitOpError("requires attribute 'value'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_Ops1(*this, tblgen_value, "value")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Ops1(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult ConstantOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult ConstantOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::DenseElementsAttr valueAttr;
  ::llvm::SmallVector<::mlir::Type, 1> allResultTypes;

  if (parser.parseCustomAttributeWithFallback(valueAttr, ::mlir::Type{})) {
    return ::mlir::failure();
  }
  if (valueAttr) result.getOrAddProperties<ConstantOp::Properties>().value = valueAttr;
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(allResultTypes))
    return ::mlir::failure();
  result.addTypes(allResultTypes);
  return ::mlir::success();
}

// 발표 - print 함수 자동 생성
void ConstantOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
_odsPrinter.printStrippedAttrOrType(getValueAttr());
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("value");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << getOperation()->getResultTypes();
}


} // namespace mlir::toy
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::toy::ConstantOp)

namespace mlir::toy {


//===----------------------------------------------------------------------===//
// ::mlir::toy::PrintOp definitions
//===----------------------------------------------------------------------===//

namespace detail {

PrintOpGenericAdaptorBase::PrintOpGenericAdaptorBase(PrintOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}


} // namespace detail
PrintOpAdaptor::PrintOpAdaptor(PrintOp op) : PrintOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult PrintOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::LogicalResult PrintOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }
      return ::mlir::success();
}

::mlir::Attribute PrintOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code PrintOp::computePropertiesHash(const Properties &prop) {
  using llvm::hash_value;
  return llvm::hash_combine();
}

std::optional<mlir::Attribute> PrintOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
  return std::nullopt;
}

void PrintOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
}

void PrintOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
}

::llvm::LogicalResult PrintOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    return ::mlir::success();
}

void PrintOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input) {
  odsState.addOperands(input);
}

PrintOp PrintOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value input) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(input)>(input));
  auto __res__ = ::llvm::dyn_cast<PrintOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

PrintOp PrintOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value input) {
  return create(builder, builder.getLoc(), std::forward<decltype(input)>(input));
}

void PrintOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input) {
  odsState.addOperands(input);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

PrintOp PrintOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value input) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(input)>(input));
  auto __res__ = ::llvm::dyn_cast<PrintOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

PrintOp PrintOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value input) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(input)>(input));
}

void PrintOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

PrintOp PrintOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(attributes)>(attributes));
  auto __res__ = ::llvm::dyn_cast<PrintOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

PrintOp PrintOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(attributes)>(attributes));
}

void PrintOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.useProperties(const_cast<Properties&>(properties));
  odsState.addAttributes(discardableAttributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

PrintOp PrintOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(properties)>(properties), std::forward<decltype(discardableAttributes)>(discardableAttributes));
  auto __res__ = ::llvm::dyn_cast<PrintOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

PrintOp PrintOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(properties)>(properties), std::forward<decltype(discardableAttributes)>(discardableAttributes));
}

::llvm::LogicalResult PrintOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Ops1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult PrintOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult PrintOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand inputRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> inputOperands(&inputRawOperand, 1);  ::llvm::SMLoc inputOperandsLoc;
  (void)inputOperandsLoc;
  ::mlir::Type inputRawType{};
  ::llvm::ArrayRef<::mlir::Type> inputTypes(&inputRawType, 1);

  inputOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(inputRawOperand))
    return ::mlir::failure();
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    inputRawType = type;
  }
  if (parser.resolveOperands(inputOperands, inputTypes, inputOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void PrintOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getInput();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getInput().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::TensorType>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}


} // namespace mlir::toy
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::toy::PrintOp)


#endif // GET_OP_CLASSES

