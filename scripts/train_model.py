from ultralytics import YOLO

# Load the YOLOv8 keypoint detection model
model = YOLO('yolov8n-pose.pt')  # Use the downloaded model

# Train the model on CPU
model.train(
    data="data.yaml",   # Path to dataset configuration
    epochs=10,          # Number of training epochs (Increase if needed)
    batch=8,            # Reduce batch size for CPU training
    imgsz=640,          # Image size
    device="cpu"        # Run on CPU
)

# Validate the model
results = model.val(device="cpu")

# Test the model on a new image (Replace 'test.jpg' with an actual test image path)
# model.predict("test.jpg", save=True, conf=0.5, device="cpu")

# Export the trained model for deployment
model.export(format="onnx")
