import torch
from transformers import AutoModel

model = AutoModel.from_pretrained(
    "ibm-research/patchtst-fm-r1",
    trust_remote_code=True,
    torch_dtype=torch.float32
).eval()

print("모델 로드 완료")
print(model)

batch_size = 1
num_channels = 1
context_length = 512

dummy_input = torch.randn(batch_size, num_channels, context_length)
past_observed_mask = torch.ones(batch_size, num_channels, context_length, dtype=torch.bool)

torch.onnx.export(
    model,
    (dummy_input, past_observed_mask),
    "models/patchtst-fm-r1.onnx",
    opset_version=17,
    input_names=["past_values", "past_observed_mask"],
    output_names=["prediction_outputs"],
    dynamic_axes={
        "past_values":        {0: "batch", 2: "context_length"},
        "past_observed_mask": {0: "batch", 2: "context_length"},
    }
)
print("Export 완료")
