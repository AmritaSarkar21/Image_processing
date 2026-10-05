import cv2 as cv
import os
import numpy as np
image_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "demo.png"
)
image = cv.imread(image_path)
if image is None:
    print("Error: demo.png not found!")
    exit()
image = cv.resize(image, (400, 500))
bright_image = cv.add(image, np.ones(image.shape, dtype=np.uint8) * 50)
cv.imshow("Increased Brightness", bright_image)
cv.waitKey(0)
cv.destroyAllWindows()
dark_image = cv.subtract(image, np.ones(image.shape, dtype=np.uint8) * 50)
cv.imshow("Decreased Brightness", dark_image)
cv.waitKey(0)
cv.destroyAllWindows()
increased_contrast = cv.convertScaleAbs(
    image,
    alpha=1.5,
    beta=0
)
cv.imshow("Increased Contrast", increased_contrast)
cv.waitKey(0)
cv.destroyAllWindows()
decreased_contrast = cv.convertScaleAbs(
    image,
    alpha=0.5,
    beta=0
)
cv.imshow("Decreased Contrast", decreased_contrast)
cv.waitKey(0)
cv.destroyAllWindows()
negative_image = 255 - image
cv.imshow("Negative Image", negative_image)
cv.waitKey(0)
cv.destroyAllWindows()
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
threshold_value = 127
ret, binary_image = cv.threshold(
    gray,
    threshold_value,
    255,
    cv.THRESH_BINARY
)
cv.imshow("Binary Threshold Image", binary_image)
cv.waitKey(0)
cv.destroyAllWindows()
print("Image Shape:", image.shape)
print("Image Size:", image.size)
print("Data Type:", image.dtype)
print("Threshold Value:", threshold_value)