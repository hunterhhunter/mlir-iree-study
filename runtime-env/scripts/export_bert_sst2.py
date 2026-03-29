import os
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def export_bert_sst2():
    model_id = "textattack/bert-base-uncased-SST-2"
    print(f"[*] 모델 다운로드 중: {model_id}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    model.eval()

    dummy_text = "This is a great benchmark tool."
    inputs = tokenizer(
        dummy_text, return_tensors="pt", 
        max_length=128, padding="max_length", truncation=True
    )
    
    project_root = os.path.dirname(os.path.dirname(__file__))
    export_dir = os.path.join(project_root, "models", "bert-base-uncased")
    os.makedirs(export_dir, exist_ok=True)
    
    export_path = os.path.join(export_dir, "bert_sst2.onnx")
    print(f"[*] ONNX 바이너리로 굽는 중: {export_path}...")
    
    with torch.no_grad():
        torch.onnx.export(
            model, 
            (inputs["input_ids"], inputs["attention_mask"]), 
            export_path,
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes=None,
            opset_version=14,
            do_constant_folding=True
        )
    print(f"\n[SUCCESS] 성공적으로 내보냈습니다! 모델 위치: {export_path}")

if __name__ == "__main__":
    export_bert_sst2()
