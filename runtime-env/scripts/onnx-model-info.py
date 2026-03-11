import onnx
import argparse
import os

def get_type_name(elem_type):
    """
    ONNX TensorProto의 정수형 타입 코드를 가독성 있는 문자열로 매핑합니다.
    """
    mapping = {
        1: "FLOAT32",
        2: "UINT8",
        3: "INT8",
        4: "UINT16",
        5: "INT16",
        6: "INT32",
        7: "INT64",
        9: "BOOL",
        10: "FLOAT16",
        11: "DOUBLE",
        12: "UINT32",
        13: "UINT64"
    }
    return mapping.get(elem_type, f"Unknown({elem_type})")

def print_io_info(io_list, label):
    """
    입력 또는 출력 리스트의 메타데이터를 파싱하여 출력합니다.
    """
    print(f"\n[{label} Information]")
    if not io_list:
        print(" - None")
        return

    for item in io_list:
        name = item.name
        t = item.type.tensor_type
        dtype = get_type_name(t.elem_type)
        
        shape = []
        for dim in t.shape.dim:
            if dim.HasField("dim_value"):
                # 정적 차원 (예: 224)
                shape.append(str(dim.dim_value))
            elif dim.HasField("dim_param"):
                # 가변 차원 (예: batch, dynamic_axes)
                shape.append(dim.dim_param)
            else:
                # 정의되지 않은 차원
                shape.append("?")
        
        print(f" - Name: {name}")
        print(f"   Shape: {' x '.join(shape)}")
        print(f"   Dtype: {dtype}")

def main():
    parser = argparse.ArgumentParser(description="ONNX Model Input/Output Inspector")
    parser.add_argument("model_path", type=str, help="Path to the .onnx model file")
    args = parser.parse_args()

    # 파일 존재 여부 확인
    if not os.path.exists(args.model_path):
        print(f"[ERROR] File not found: {args.model_path}")
        return

    try:
        # ONNX 모델 로드
        model = onnx.load(args.model_path)
        
        print(f"==========================================")
        print(f"=== [Model Analysis: {os.path.basename(args.model_path)}] ===")
        print(f"==========================================")
        
        # 모델 메타데이터 출력
        opset = model.opset_import[0].version if model.opset_import else "N/A"
        print(f"Opset Version: {opset}")
        print(f"Producer: {model.producer_name} (v{model.producer_version})")

        # 입출력 시그니처 분석
        print_io_info(model.graph.input, "Input")
        print_io_info(model.graph.output, "Output")
        
        print("\n" + "="*40)
        print("Analysis Complete.")
        
    except Exception as e:
        print(f"[CRITICAL] Failed to load/analyze model: {e}")

if __name__ == "__main__":
    main()
