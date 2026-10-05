import cv2 as cv
import numpy as np
import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
image_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "demo.png"
)
image = cv.imread(image_path)
if image is None:
    print("Error: Image could not be loaded.")
    print("Check whether demo.png exists in the same folder as this Python file.")
    exit()
cv.imshow("Original Image", image)
cv.waitKey(0)
cv.destroyAllWindows()
image = cv.resize(image, (400, 500))
cv.imshow("Resized Image", image)
cv.waitKey(0)
cv.destroyAllWindows()
print("Height and Width:", image.shape)
print("Number of pixels:", image.size)
print("Data type:", image.dtype)
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
min_val, max_val, min_loc, max_loc = cv.minMaxLoc(gray)
print("Maximum pixel intensity:", max_val)
print("Minimum pixel intensity:", min_val)
print("Location of maximum intensity:", max_loc)
print("Location of minimum intensity:", min_loc)
print("Pixel intensity at (50, 50):", gray[50, 50])
rgb_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
plt.imshow(rgb_image)
plt.axis("off")
plt.show()