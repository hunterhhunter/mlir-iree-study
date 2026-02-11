# MLIR Toy Tutorial Chapter 7: Adding a Composite Type to Toy

## 7장 목표
- 6장에서는 toy -> llvm -> JIT까지 실행이 가능하도록 하는 lowering 과정을 학습. 7장에서는 구조체를 선언하는 방법을 다룸.
- 구조체 인터페이스를 정의하고 레서, AST, 파서를 구성
- Toy 언어에 Type을 추가해 MLIR에서 타입을 다루는 방법을 학습

## Struct 선언 워크플로우
### 1. AST 작성 (AST.h)
1. 함수와 구조체를 포함하는 Record 타입(RecordAST) 선언
	toy 언어에서는 함수의 선언과 구조체의 선언이 같은 레벨에서 이루어지기 때문에 Module 레벨에서 따로 관리하는게 아닌 RecordAST를 만들어 그 아래에서 관리하기 위함
2. 구조체 타입(StructAST)을 선언 
	Record 타입을 상속하고 vector\<unique_ptr\<VarDeclExprAST>> variables를 멤버 변수 넣어 내부의 변수를 관리하게 함.
3. ModuleAST가 RecordAST를 보유하도록 변경
4. 구조체 표현식 AST인 StructLiteralExprAST 선언
	StructAST는 struct Complex = {...} 전체를 파싱한 결과이며 선언이지만, 
	StructLiteralExprAST는 {1, 2} 부분으로 실행될 때 값을 만들어내는 표현식
### 2. 렉서와 파서 작성 (Lexer.h, Parser.h)
1. 렉서에 구조체 토큰 등록 및 struct 단어 등록
	struct 단어를 렉서가 인지하고 번호를 부여할 수 있도록 생성
2. 타입이 추가되며 두 스타일로 바뀐 변수 선언을 파싱할 함수 수정
	- var a(일반 텐서), mystruct a(구조체 타입)을 다룰 함수를 작성
	- `parseDeclaration`: var로 시작하는 선언인지 StructName으로 시작하는 선언인지 확인 후 핸들링
	- `parseVarDeclaration`: var로 시작하는 변수 선언 처리
	- `parseTypedDeclaration`: 타입 이름을 확인 후 변수 이름과 초기값만 파싱후 VarDeclExprAST를 반환
	- `parseDeclarationOrCallExpr`: Name으로 시작하는 문장이 함수 호출인지 구조체 선언인지 판별
3. 블럭을 파싱하는 함수(parseBlock)에서 2에서 작성한 `parseDeclarationOrCallExpr`함수를 사용하도록 변경
	식별자(변수명, 구조체명, 함수명)가 나왔을 때 어떤 의미인지를 확인하기 위해 호출
4. 구조체 파싱 함수(parseStruct)와 구조체 리터럴 파싱 함수(parseStructLiteralExpr) 작성
	구조체 파싱 함수는 정의할 때 사용, 구조체 리터럴 파싱 함수는 값 생성시(함수 내부)에 사용
5. 구조체와 구조체 리터럴을 dump하는 함수 작성 (AST.cpp)
### 3. 구조체 Operation 선언
1. Toy_StructType 선언
	Toy언어의 타입을 Toy_Type에 F64Tensor, Toy_StructType를 추가함으로써 선언
2. 구조체 상수 Operation 선언 `StructConstantOp`
	숫자(Tensor)로만 이루어진 구조체를 선언할 때 사용되는 Operation
3. 구조체 접근 Operation 선언 `StructAccessOp`
	a.b 에 접근하는 Operation
	상수 Folding을 가능하게 해서 상수 접근 연산 자체를 상수로 변환시킬 수 있게함.
4. Generic_call, ReturnOp가 입력과 출력으로 Toy_Type을 지원하도록 변경
### 4. Toy -> MLIR Code Generation 로직 작성 (MLIRGen.cpp)
1. 멤버 변수 선언
	1. 함수 이름과 생성된 FuncOp를 Mapping하여 관리하는 functionMap 멤버 변수 선언
		RecordAST 내부에는 StructAST와 FuncAST가 뒤섞여 있기 때문에 함수를 따로 매핑해두면 검색이 빨라짐.
	2. 구조체 이름과 생성된 StructAST를 Mapping하여 관리하는 structMap 멤버 변수 선언
	3. symbolTable 멤버변수 선언
		Toy -> MLIR 과정에서 발생하는 정보 손실(이름, 인덱스, 구조체 정보)를 막기 위해 `varDeclExprAST`를 저장하는 테이블을 선언해 정보를 보존
2. 타입이 구조체인지 변수인지 확인 후 각각의 type을 return하는 함수 `getType` 작성
3. `mlirGen(ModuleAST &ModuleAST)`에서 RecordAST를 사용하도록 변경
4. 현재 scope에서 symbolTable에 VarDeclExprAST를 삽입하는 함수 declare 선언
5. `mlirGen(StructAST  &str)` 함수를 선언
	인자들을 가져와 타입을 한 vector에 넣고 structMap에 등록
6. \`*getStructFor(ExprAST \*expr)` 함수 작성
	a.b가 입력되면 멤버 b의 구조체 타입을 찾아 반환함. = a의 VarDeclExprAST를 반환
7. 멤버변수의 인덱스를 계산하는 `getMemberIndex` 함수를 작성
8. `mlirGen(BinaryExprAST &binop)`에서 .으로 구조체 접근 로직 추가
9. `getConstantAttr(StructLiteralExprAST &lit)` 함수 작성
	구조체 리터럴 표현식 AST를 구조를 유지한채로 타입과 실제 데이터값을 동시에 반환하는 함수
10. `mlirGen(StructLiteralExprAST &lit)` 함수 작성
11. `mlirGen(CallExprAST &call)`함수에서 `functionMap`에서 존재하는 함수인지 확인하는 로직 추가
12. `mlirGen(VarDeclExprAST &vardecl)`함수에서 구조체인지 검사하는 로직 작성
13. `mlirGen(ExprAST &expr)`함수에 StructLiteralExprAST의 경우를 추가
### StructType 정의 및 구현 (Dialect.h, Dialect.cpp)
1. StructType 클래스 정의 (Dialect.h)
	`mlir::Type::TypeBase<StructType, mlir::Type, detail::StructTypeStorage>`를 상속받는데 템플릿 인자들은 차레로 
	Concrete Type: 최종적으로 생성될 클래스
	Base Class: Is-a 관계로 mlir::Type의 일종
	Storage Class: 실제 구현 클래스를 입력
2. `StructType`의 구현체인 `StructTypeStorage` 클래스 구현 (Dialect.cpp)
3. StructType의 get, getElementTypes 메서드 구현
	get: 생성자 역할의 Factory Method로 입력 인자의 구성을 가진 StructType을 검색하고 있으면 기존 객체 주소 반환하고, 없으면  등록해 그 주소를 반환
		-> 그래서 전역 타입 관리 객체인 Base를 사용함. + 전역 관리자인 MLIRContext를 사용
4. `ToyDialect::parseType` 메서드 작성
	MLIR로 변환된 결과를 StructType 객체로 변환하는 함수
	예: !\<toy.struct\<tensor\<\*xf64>>>
5. `ToyDialect::**printType` 메서드 작성
	메모리 객체 -> 텍스트를 위한 함수
6. StructAccessOp의 build, verify 함수 구현
7. 타입별로 상수를 검증하기위한 `verifyConstantForType` 함수 작성
	재귀로 구현되어 구조체도 검증 가능
8. ConstantOp, StructConstantOp의 verify에 `verifyConstantForType`를 사용하도록 변경
9. ToyDialect::initialize에 addTypes\<StructType>();을 추가
