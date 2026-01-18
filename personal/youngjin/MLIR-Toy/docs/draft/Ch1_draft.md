챕터 1은 렉서, AST 노드, 파서를 직접 구현하여 작성한 코드가 AST로 변환되는 것을 보여주는 챕터이다.

웹 튜토리얼(링크)은 핵심 내용만을 담지만 깃허브(링크)의 코드를 보면 자세한 내용을 볼 수 있다.

Lexer 렉서
렉서는 실제 코드를 잘라서 토큰화해주는 역할을 맡는다.

토큰을 구분하는 구분자 enum: int(Token)을 선언하고 내부에 ;, (, ), {, }, [, ], EOF, return, var, def, identifier, number을 숫자에 매핑

렉서 클래스에는 다음과 같은 핵심 메서드가 있다.
(표)
Token getter(cur, next), location getter(Location, line, col), consume 다음 토큰으로 이동(next token getter 사용), string id getter, getNextChar 다음 문자를 읽어오는데 \\n과 EOF 예외를 처리, getTok while문을 이용해 한 단어를 토큰화 및 분류(identifier, number, 주석, EOF)해서 return

LexerBuffer라는 Lexer의 구현 클래스에는 readNextLine 함수의 존재로 다음줄 전체를 llvm::StringRef로 읽어오는 기능이 구현되어있음.

AST에는 AST의 노드들을 정의해뒀다.
노드는 아래 표로 정리해뒀다
(표) VarDecl 변수 정의, return 반환, Num 숫자, Literal 리터럴, BinOp 오퍼레이션(피연산자 2개), call 함수콜, print 출력

AST 노드에서 중요한건 ExprAST라는 부모 클래스로 모든 것이 관리된다는 점으로 디버깅을 위한 중요 정보인 Location은 std::move로 복사 없이 소유권을 부모 클래스로 넘긴다.

그리고 노드는 ExprAST를 상속받아 구현되는데 큰 내용은 없지만 특징은 classof라는 LLVM style의 RTTI(Run-Time Type Information)을 구현하는 것으로 보통의 C++에서 dynamic_cast나 typeid를 사용해 부모 포인터가 실제 어떤 자식 클래스를 가리키는지 확인하는 방법 대신 직접 만든 RTTI 시스템을 이용하게 해야함.
일종의 LLVM RTTI에 등록하는 과정으로 볼 수 있음.

파서는 트리 구조를 vector를 사용해 노드를 관리한다.
렉서와 AST의 노드 정보로 텍스트를 실제 노드로 만드는 내용이 구현되어있다.
특이한 점은 템플릿으로 parseError 함수를 구현해 파싱 에러를 커스텀한 것이 있다.

알게된 문법&디자인패턴 정리
1. using name = ... / type Alias로 typedef와 같은 문법 - = 뒤의 복잡한 자료형들을 name으로 바꿀 수 있음 -> 가독성+이해도+작성편이성 모두 증가
2. ExprAST라는 기본 부모 클래스 설계 후 상속으로 노드를 구현한 상속 디자인 패턴 + 무거운 객체(디버깅용)를 std::move로 비용 0에 가깝게 참조자 자체를 부모로 넘겨 소유권 이전 -> 모든 노드는 EsprAST의 Location을 참조하면됨 -> 효율적인 관리
3. std::move / 무거운 객체, 구조체 등 힙을 차지하는 애들 + unique_ptr로 복사가 불가능한 애들 + shared_ptr처럼 참조할 때마다 오버헤드가 발생하는 애들의 소유권을 이전하는데 사용되는데 내부 로직은 인자로 들어가는 순간 참조자(&&)로 캐스팅 후 포인터만 얕은 복사 후 원본을 nullptr로 만드는 과정을 진행해서 일반적인 복사 반환보다 훨씬 빠름
4. llvm::StringRef <-> String간 암시적 변환 / llvm::StringRef에 string을 포함하는 생성자가 있어 string이 반환형이더라도 llvm::StringRef로 자동 변환되어 반환됨.
5. *와 &의 차이점 / 포인터의 경우 ->로 역참조해서 봐야함, 참조자의 경우 .으로 원본 그 자체에 접근이 가능함.
6. 객체를 반환하면 기본적으로 복사가 일어남. 그래서 객체 포인터로 넘겨 복사를 방지
7. 객체 절단 / 자식 클래스를 return할 때 복사되어 전달하는데 이 때 반환형이 부모 클래스라면 부모 클래스만큼만 복사해 자식클래스가 사라짐.
8. vector.back()과 .empty()의 차이 / back은 뒤 원소를 반환하며 없으면 nullptr로 확인이 가능, empty()는 비어있는지만 확인
9. template typename R(리턴타입), T(기대타입) 컴파일러가 자동으로 처리해주는 템플릿 문법