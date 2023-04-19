import cv2
import numpy as np

def mask_generator(file_path:str, inverse:bool = False):
    # Read in the original image
    image = cv2.imread(file_path)

    # Get the dimensions of the image
    height, width, channels = image.shape

    # Create a mask of the same size
    mask = np.zeros((height, width, channels), dtype=np.uint8)
    
    if inverse:
        mask.fill(255)

    return mask


if(__name__ == "__main__"):
    mask_generator("dataset/background/10.jpg")