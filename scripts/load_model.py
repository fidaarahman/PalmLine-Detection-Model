# #pip install ultralytics # this is the library by which just we will downlaod it and the model will itself doneload

# #the below script will load the .pt file 
# from ultralytics import YOLO

# # Load YOLOv8 Pose Estimation Model
# model = YOLO('yolov8n-pose.pt')  # Use the downloaded file

from ultralytics import YOLO

# Load your model
model = YOLO("best.pt")

# Check model type
print("Model type:", model.task)
