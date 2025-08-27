# from ultralytics import YOLO

# model = YOLO("best_float32.tflite")
# results = model.predict(source="saad2.jpeg", conf=0.25, iou=0.6, imgsz=640, verbose=False)
# results[0].show() 

# from ultralytics import YOLO

# model = YOLO("best.pt")
# print("✅ Number of classes:", model.model.nc)
# print("✅ Class names:", model.names)

import tensorflow as tf

# Load your TFLite model
interpreter = tf.lite.Interpreter(model_path="best_float32.tflite")
interpreter.allocate_tensors()

# Get output tensor details
output_details = interpreter.get_output_details()
output_shape = output_details[0]['shape']

print("📦 TFLite Output Shape:", output_shape)
print("📌 Output Tensor Name:", output_details[0]['name'])

# [1, N, 8400] → N should be 9 if you have 4 classes (4 boxes + 1 obj conf + 4 class scores)

 # Optional: shows image with detections
# from ultralytics import YOLO
# import cv2
# import torch

# # Load the model
# model = YOLO("best.pt")

# # Read the image
# img = cv2.imread("asim2.jpg")

# # Perform inference
# results = model(img)

# # Print keypoints and boxes info
# print("Keypoints shape:", results[0].keypoints.shape)
# print("Keypoints raw tensor:", results[0].keypoints.data)
# print("Boxes shape:", results[0].boxes.data.shape)
# print("Boxes:", results[0].boxes.data)

# # Draw boxes and keypoints on the image
# img_with_boxes = img.copy()

# # Draw boxes
# for box in results[0].boxes.data:
#     x1, y1, x2, y2 = map(int, box[:4])
#     cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)

# # Draw keypoints (if any)
# for keypoints in results[0].keypoints.data:
#     for x, y, confidence in keypoints:
#         if confidence > 0.5:  # Filter out weak keypoints if needed
#             cv2.circle(img_with_boxes, (int(x), int(y)), 5, (0, 0, 255), -1)

# # Display the image with boxes and keypoints
# cv2.imshow("Detected Image", img_with_boxes)

# # Save the image with detections
# cv2.imwrite("detected_image.jpg", img_with_boxes)

# # Wait for a key press to close the window
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# from ultralytics import YOLO

# # Load your trained YOLOv8 model (replace with your model's file path)
# model = YOLO('best.pt')  # Or use your custom trained model file
# # Convert the model to TensorFlow Lite format
# model.export(format="tflite")
