import cv2
import numpy as np
import matplotlib.pyplot as plt


image = cv2.imread("lena.png", cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Error: Could not read the image.")
    exit()


def resize_image(image, scale, interpolation):

    height, width = image.shape

    new_width = int(width * scale)
    new_height = int(height * scale)

    resized = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=interpolation
    )

    return resized



nearest_2x = resize_image(
    image,
    2,
    cv2.INTER_NEAREST
)

bilinear_2x = resize_image(
    image,
    2,
    cv2.INTER_LINEAR
)

bicubic_2x = resize_image(
    image,
    2,
    cv2.INTER_CUBIC
)



nearest_4x = resize_image(
    image,
    4,
    cv2.INTER_NEAREST
)

bilinear_4x = resize_image(
    image,
    4,
    cv2.INTER_LINEAR
)

bicubic_4x = resize_image(
    image,
    4,
    cv2.INTER_CUBIC
)


# 1/2 Shrinking
nearest_half = resize_image(
    image,
    0.5,
    cv2.INTER_NEAREST
)

bilinear_half = resize_image(
    image,
    0.5,
    cv2.INTER_LINEAR
)

bicubic_half = resize_image(
    image,
    0.5,
    cv2.INTER_CUBIC
)


# 1/4 Shrinking
nearest_quarter = resize_image(
    image,
    0.25,
    cv2.INTER_NEAREST
)

bilinear_quarter = resize_image(
    image,
    0.25,
    cv2.INTER_LINEAR
)

bicubic_quarter = resize_image(
    image,
    0.25,
    cv2.INTER_CUBIC
)


# Display 2x Zoom
images_2x = [
    image,
    nearest_2x,
    bilinear_2x,
    bicubic_2x
]

titles_2x = [
    "Original",
    "Nearest Neighbor - 2x",
    "Bilinear - 2x",
    "Bicubic - 2x"
]

plt.figure(figsize=(12, 8))

for i in range(4):
    plt.subplot(2, 2, i + 1)
    plt.imshow(images_2x[i], cmap="gray")
    plt.title(titles_2x[i])
    plt.axis("off")

plt.tight_layout()
plt.show()


# Display 4x Zoom
images_4x = [
    nearest_4x,
    bilinear_4x,
    bicubic_4x
]

titles_4x = [
    "Nearest Neighbor - 4x",
    "Bilinear - 4x",
    "Bicubic - 4x"
]

plt.figure(figsize=(12, 4))

for i in range(3):
    plt.subplot(1, 3, i + 1)
    plt.imshow(images_4x[i], cmap="gray")
    plt.title(titles_4x[i])
    plt.axis("off")

plt.tight_layout()
plt.show()


# Display 1/2 Shrinking
images_half = [
    nearest_half,
    bilinear_half,
    bicubic_half
]

titles_half = [
    "Nearest Neighbor - 1/2",
    "Bilinear - 1/2",
    "Bicubic - 1/2"
]

plt.figure(figsize=(12, 4))

for i in range(3):
    plt.subplot(1, 3, i + 1)
    plt.imshow(images_half[i], cmap="gray")
    plt.title(titles_half[i])
    plt.axis("off")

plt.tight_layout()
plt.show()


# Display 1/4 Shrinking
images_quarter = [
    nearest_quarter,
    bilinear_quarter,
    bicubic_quarter
]

titles_quarter = [
    "Nearest Neighbor - 1/4",
    "Bilinear - 1/4",
    "Bicubic - 1/4"
]

plt.figure(figsize=(12, 4))

for i in range(3):
    plt.subplot(1, 3, i + 1)
    plt.imshow(images_quarter[i], cmap="gray")
    plt.title(titles_quarter[i])
    plt.axis("off")

plt.tight_layout()
plt.show()


def nearest_neighbor_manual(image, scale):

    old_height, old_width = image.shape

    new_height = int(old_height * scale)
    new_width = int(old_width * scale)

    # Create empty output image
    output = np.zeros(
        (new_height, new_width),
        dtype=np.uint8
    )

    # Mapping new pixels to old pixels
    for y in range(new_height):

        for x in range(new_width):

            # Find nearest original pixel
            original_y = int(y / scale)
            original_x = int(x / scale)

            # Prevent index from going outside image
            original_y = min(original_y, old_height - 1)
            original_x = min(original_x, old_width - 1)

            output[y, x] = image[
                original_y,
                original_x
            ]

    return output


# Manual 2x Nearest Neighbor
manual_2x = nearest_neighbor_manual(
    image,
    2
)

# Manual 1/2 Nearest Neighbor
manual_half = nearest_neighbor_manual(
    image,
    0.5
)

# Display Manual Results
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(manual_2x, cmap="gray")
plt.title("Manual Nearest Neighbor - 2x")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(manual_half, cmap="gray")
plt.title("Manual Nearest Neighbor - 1/2")
plt.axis("off")

plt.tight_layout()
plt.show()