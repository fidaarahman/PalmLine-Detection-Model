from ultralytics import YOLO

# Load the YOLOv8 keypoint detection model
model = YOLO('yolov8n-pose.pt')  # Use the downloaded model

# Train the model
model.train(
    data="data.yaml",   # Path to dataset configuration
    epochs=40,          # Number of training epochs (Increase if needed)
    batch=16,           # Adjust based on GPU capacity
    imgsz=640,          # Image size
    device="cuda"       # Use GPU if available, else use "cpu"
)

# Validate the model
results = model.val()

# # Test the model on a new image (Replace 'test.jpg' with an actual test image path)
# model.predict("pa.jpg", save=True, conf=0.5)

# Export the trained model for deployment
model.export(format="onnx")  # Save model in ONNX format
