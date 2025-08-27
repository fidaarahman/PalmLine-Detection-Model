from ultralytics import YOLO
import torch

model = YOLO("best.pt")
dummy_input = torch.randn(1, 3, 640, 640)
with torch.no_grad():
    outputs = model.model(dummy_input)  
print("Type of output:", type(outputs))
print("Length of tuple:", len(outputs))

for i, out in enumerate(outputs):
    if isinstance(out, torch.Tensor):
        print(f"Output[{i}] shape:", out.shape)
    else:
        print(f"Output[{i}] is not a tensor:", type(out))
