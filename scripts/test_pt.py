from ultralytics import YOLO
import cv2

# Load model and image
model = YOLO("model/best.pt")
image_path = "2nd.png"
results = model(image_path)[0]

# Load original image
image = cv2.imread(image_path)
h, w = image.shape[:2]

# Class name map
class_map = {0: 'fate', 1: 'head', 2: 'heart', 3: 'life'}

# Draw lines in detected boxes
for box in results.boxes.data.cpu().numpy():
    x1, y1, x2, y2, conf, cls = box
    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
    class_name = class_map[int(cls)]

    # Draw custom line (as diagonal or curve placeholder)
    if class_name == 'heart':
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    elif class_name == 'head':
        cv2.line(image, (x1, y2), (x2, y1), (255, 255, 0), 2)
    elif class_name == 'life':
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 1)
    elif class_name == 'fate':
        cx = (x1 + x2) // 2
        cv2.line(image, (cx, y1), (cx, y2), (255, 255, 255), 2)

    # Add label
    cv2.putText(image, f"{class_name}", (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

# Save output
cv2.imwrite("custom_lines_result.jpg", image)
print("✅ Saved as custom_lines_result.jpg")
