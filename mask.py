import cv2
import numpy as np

def mask_generator(file_path):
    # Read in the original image
    image = cv2.imread(file_path)

    # Get the dimensions of the image
    height, width, channels = image.shape

    # Create a white mask of the same size
    mask = np.zeros((height, width, channels), dtype=np.uint8)
    mask.fill(255)

    return mask


if(__name__ == "__main__"):
    mask_generator("dataset/background/10.jpg")