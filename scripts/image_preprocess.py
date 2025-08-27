import cv2
import numpy as np
from PIL import Image

# Load image
image = cv2.imread('new.png')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Convert to HSV to detect colored lines
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define color masks for red, green, blue, black
def mask_color(hsv, lower, upper):
    return cv2.inRange(hsv, np.array(lower), np.array(upper))

# Red line (adjust if needed)
mask_red1 = mask_color(hsv, [0, 50, 50], [10, 255, 255])
mask_red2 = mask_color(hsv, [160, 50, 50], [180, 255, 255])
mask_red = cv2.bitwise_or(mask_red1, mask_red2)

# Green line
mask_green = mask_color(hsv, [35, 50, 50], [85, 255, 255])

# Blue line
mask_blue = mask_color(hsv, [90, 50, 50], [130, 255, 255])

# Black line
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, mask_black = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY_INV)

# Combine all masks
combined_mask = cv2.bitwise_or(mask_red, mask_green)
combined_mask = cv2.bitwise_or(combined_mask, mask_blue)
combined_mask = cv2.bitwise_or(combined_mask, mask_black)

# Create transparent image
rgba = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
rgba[:, :, 3] = combined_mask  # Set alpha channel from mask

# Save result
cv2.imwrite('output_lines_only.png', rgba)

# Optional: Show preview
# Image.fromarray(rgba).show()
