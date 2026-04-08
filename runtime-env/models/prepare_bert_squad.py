import os
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

def export_bert_squad():
    model_id = "csarron/bert-base-uncased-squad-v1"
    print(f"[*] SQuAD 모델 다운로드 중: {model_id}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForQuestionAnswering.from_pretrained(model_id)
    model.eval()

    # Sequence Length 제한인 384 크기의 텐서를 추론하기 위해 더미 입력 생성
    dummy_question = "What is the primary function of this benchmarking framework?"
    dummy_context = "This benchmarking framework evaluates NPU latencies meticulously. It performs zero-latency loading and mathematically elegant vectorizations."
    
    inputs = tokenizer(
        dummy_question, dummy_context, return_tensors="pt", 
        max_length=384, padding="max_length", truncation=True
    )
    
    project_root = os.path.dirname(os.path.dirname(__file__))
    export_dir = os.path.join(project_root, "models", "bert-base-uncased-squad-v1")
    os.makedirs(export_dir, exist_ok=True)
    
    export_path = os.path.join(export_dir, "squad.onnx")
    print(f"[*] ONNX 바이너리로 굽는 중 (Static Shape, SeqLength: 384): {export_path}...")
    
    # ONNX 최적화 그래프 파서 옵션 부여
    with torch.no_grad():
        torch.onnx.export(
            model, 
            (inputs["input_ids"], inputs["attention_mask"]), 
            export_path,
            input_names=["input_ids", "attention_mask"],
            output_names=["start_logits", "end_logits"],
            dynamic_axes={
                "input_ids": {0: "batch_size"},
                "attention_mask": {0: "batch_size"},
                "start_logits": {0: "batch_size"},
                "end_logits": {0: "batch_size"}
            },
            opset_version=14,
            do_constant_folding=True
        )
    print(f"\n[SUCCESS] SQuAD 모델 성공적으로 내보냈습니다! 위치: {export_path}")

if __name__ == "__main__":
    export_bert_squad()
