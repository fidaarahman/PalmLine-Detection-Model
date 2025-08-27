# import torch
# import onnxruntime as ort
# import numpy as np
# from ultralytics import YOLO

# # === 1. Load the PyTorch model ===
# pt_model = YOLO("best.pt").model
# pt_model.eval()
# pt_model.to(torch.float32)

# # === 2. Prepare the same dummy input ===
# dummy_input = torch.randn(1, 3, 640, 640).float()
# pt_output = pt_model(dummy_input)

# # Extract tensor from PyTorch model output
# if isinstance(pt_output, (list, tuple)):
#     pt_output_tensor = pt_output[0]
# else:
#     pt_output_tensor = pt_output

# pt_output_numpy = pt_output_tensor.detach().cpu().numpy()

# # === 3. Load and run ONNX model ===
# ort_session = ort.InferenceSession("best.onnx")

# # Get input name dynamically
# input_name = ort_session.get_inputs()[0].name

# # Run inference
# onnx_outputs = ort_session.run(None, {input_name: dummy_input.numpy()})
# onnx_output_numpy = onnx_outputs[0]

# # === 4. Compare outputs ===
# are_close = np.allclose(pt_output_numpy, onnx_output_numpy, rtol=1e-03, atol=1e-05)

# print("✅ Are PyTorch and ONNX outputs similar?", are_close)
# print("ℹ️ Output difference (mean absolute):", np.mean(np.abs(pt_output_numpy - onnx_output_numpy)))
import cv2
import torch
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO


def preprocess_image(image_path):
    original = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (640, 640))
    img_input = img_resized / 255.0
    img_input = np.transpose(img_input, (2, 0, 1))  # CHW
    img_input = np.expand_dims(img_input, axis=0)   # Add batch
    return original, img_input.astype(np.float32)


def draw_boxes(image, boxes, color=(0, 255, 0), label="MODEL"):
    img_draw = image.copy()
    for i, box in enumerate(boxes):
        if len(box) < 6:
            continue
        x1, y1, x2, y2, conf, cls = box[:6]
        try:
            conf = float(np.squeeze(conf))
        except:
            continue
        if conf < 0.3:
            continue
        cv2.rectangle(img_draw, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(img_draw, f"{label} {int(cls)} {conf:.2f}", (int(x1), int(y1)-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return img_draw


def run_pytorch_model(image_path):
    print("Running PyTorch model...")
    original, input_tensor = preprocess_image(image_path)
    model = YOLO("best.pt").model
    model.eval().to(torch.float32)
    with torch.no_grad():
        output = model(torch.tensor(input_tensor))
    boxes = output[0].cpu().numpy() if isinstance(output, (list, tuple)) else output.cpu().numpy()
    image_with_boxes = draw_boxes(original, boxes, color=(0, 255, 0), label="PT")
    cv2.imwrite("result_pt.jpg", image_with_boxes)
    print("✅ Saved result to result_pt.jpg")


def run_onnx_model(image_path):
    print("Running ONNX model...")
    original, input_tensor = preprocess_image(image_path)
    ort_session = ort.InferenceSession("best.onnx")
    input_name = ort_session.get_inputs()[0].name
    onnx_output = ort_session.run(None, {input_name: input_tensor})
    boxes = onnx_output[0]
    image_with_boxes = draw_boxes(original, boxes, color=(0, 0, 255), label="ONNX")
    cv2.imwrite("result_onnx.jpg", image_with_boxes)
    print("✅ Saved result to result_onnx.jpg")


# === MAIN EXECUTION ===
image_path = "C:/Users/fidau/Downloads/Palm lines recognition.v10i.yolov8/asim2.jpg"  # 🔁 Replace with your test image

run_pytorch_model(image_path)
run_onnx_model(image_path)
