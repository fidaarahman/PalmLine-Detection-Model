from ultralytics import YOLO

# Load the retrained PyTorch model
model = YOLO('best.pt')

# Export to TFLite (float32 for better accuracy)
model.export(format='tflite', imgsz=640, dynamic=False, int8=False)
