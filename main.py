import cv2
import albumentations as A
import prespective_transform as pt

import os
import random


# Read in the document and mask images
document_image, mask_image = pt.transfrom("dataset/documents/3_nouvel-obs_hbhnr300_constructedPdf_Nouvelobs2402PDF.orig.png")

# Set the path to the folder containing the background images
folder_path = 'dataset/background/'

# Get a list of all the image file names in the folder
image_names = os.listdir(folder_path)

# Randomly select 6 images from the list
selected_image_names = random.sample(image_names, 6)




# Define the brightness object
transformed = A.Compose([
    A.RandomBrightnessContrast()
])

# Apply the transformations
for i in range(len(document_image)):
    cv2.imwrite("c"+str(i)+".png", document_image[i])
    
    # Apply the transformations to the document and mask images
    transformed_img = transformed(image = document_image[i])["image"]

    cv2.imwrite("g"+str(i)+".png",transformed_img[i])