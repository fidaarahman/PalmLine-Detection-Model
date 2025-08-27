import os
import cv2
import numpy as np

# Paths
image_folder = 'path to your train images folder'
label_folder = 'path to your train images labeles'
output_folder = 'path of the new folder'

os.makedirs(output_folder, exist_ok=True)

for image_file in os.listdir(image_folder):
    if not image_file.endswith(('.jpg', '.png')):
        continue

    img_path = os.path.join(image_folder, image_file)
    label_path = os.path.join(label_folder, image_file.replace('.jpg', '.txt').replace('.png', '.txt'))

    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    mask = np.zeros((h, w, 3), dtype=np.uint8)

    if not os.path.exists(label_path):
        continue

    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            class_id = int(parts[0])
            keypoints = list(map(float, parts[1:]))

            # Group points into (x, y) pairs
            points = []
            for i in range(0, len(keypoints), 2):
                x = int(keypoints[i] * w)
                y = int(keypoints[i + 1] * h)
                points.append([x, y])

            # Convert to numpy array and draw lines
            points_np = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(mask, [points_np], isClosed=False, color=(255, 255, 255), thickness=2)

    out_path = os.path.join(output_folder, image_file)
    cv2.imwrite(out_path, mask)

