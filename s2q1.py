import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("demo.png")

if image is None:
    print("Error: Could not read the image.")
    exit()

# Convert BGR to RGB for Matplotlib
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

height, width = image.shape[:2]

# Move image 100 pixels right and 50 pixels down
tx = 100
ty = 50

translation_matrix = np.float32([
    [1, 0, tx],
    [0, 1, ty]
])

translated = cv2.warpAffine(
    image,
    translation_matrix,
    (width, height)
)


# Scale image by 1.5 times
scaled = cv2.resize(
    image,
    None,
    fx=1.5,
    fy=1.5,
    interpolation=cv2.INTER_LINEAR
)

# Rotate image by 45 degrees
center = (width // 2, height // 2)

rotation_matrix = cv2.getRotationMatrix2D(
    center,
    45,
    1.0
)

rotated = cv2.warpAffine(
    image,
    rotation_matrix,
    (width, height)
)


horizontal_reflection = cv2.flip(image, 1)


vertical_reflection = cv2.flip(image, 0)

images = [
    image_rgb,
    cv2.cvtColor(translated, cv2.COLOR_BGR2RGB),
    cv2.cvtColor(scaled, cv2.COLOR_BGR2RGB),
    cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB),
    cv2.cvtColor(horizontal_reflection, cv2.COLOR_BGR2RGB),
    cv2.cvtColor(vertical_reflection, cv2.COLOR_BGR2RGB)
]

titles = [
    "Original",
    "Translation",
    "Scaling",
    "Rotation",
    "Horizontal Reflection",
    "Vertical Reflection"
]

plt.figure(figsize=(12, 8))

for i in range(len(images)):
    plt.subplot(2, 3, i + 1)
    plt.imshow(images[i])
    plt.title(titles[i])
    plt.axis("off")

plt.tight_layout()
plt.show()