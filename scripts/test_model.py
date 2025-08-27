from ultralytics import YOLO

model = YOLO("path to your model")
results = model.predict(source="test.jpg", save=True, imgsz=640)
results[0].show()  # Optional: shows image with detections