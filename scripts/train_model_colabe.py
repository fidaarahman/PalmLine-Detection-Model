from ultralytics import YOLO

# Load the YOLOv8 keypoint detection model
model_path = "path to your model"  
model = YOLO(model_path)  

# Train the model on GPU
model.train(
    data="path to your data set",  # Correct dataset path
    epochs=10,          # Number of training epochs (Increase if needed)
    batch=8,            # Reduce batch size if low memory
    imgsz=640,          # Image size
    device="cuda"       # Use GPU for training
)

# Validate the model on GPU
results = model.val(device="cuda")

# Test the model on a new image (Replace 'test.jpg' with an actual test image path)
# model.predict("test.jpg", save=True, conf=0.5, device="cuda")

# Export the trained model for deployment
model.export(format="onnx")
