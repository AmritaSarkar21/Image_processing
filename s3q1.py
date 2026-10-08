import numpy as np
import matplotlib.pyplot as plt


# Creating binary image
binary = np.array([
    [0,0,0,0,0],
    [0,1,1,0,0],
    [0,1,0,1,0],
    [0,0,1,1,0],
    [0,0,0,0,0],
])

plt.imshow(binary, cmap='gray')
plt.title('Binary Image')

for i in range(5):
    for j in range(5):
        plt.text(j, i, str(binary[i, j]), ha='center', va='center', color='red')

plt.show()


# 4 neighbour connectivity

def get_4_neighbour(image, r, c):
    neighbour = []
    rows, cols = image.shape
    for dr, dc in [(-1,0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r+dr, c+dc
        if 0 <= nr < rows and 0 <= nc < cols and image[nr, nc] == 1:
            neighbour.append((nr, nc))

        return neighbour

print("Pixel (2, 1)", binary[2,1])
print("4 neighbours: ", get_4_neighbour(binary, 2, 1))

# 8 neighbour connectivity

def get_4_neighbour(image, r, c):
    neighbour = []
    rows, cols = image.shape
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]: 
            if dr == 0 and dc == 0: 
                continue
            nr, nc = r+dr, c+dc
            if 0 <= nr < rows and 0 <= nc < cols and image[nr, nc] == 1:
                neighbour.append((nr, nc))

    return neighbour

print("Pixel (2, 1)", binary[2,1])
print("4 neighbours: ", get_4_neighbour(binary, 2, 1))