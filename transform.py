import cv2
import numpy as np
import mask


def transfrom(file_path:str, num_of_transforms:int = 6, probality:float = 0.7):

    # Read in the document and mask images
    document_image = cv2.imread(file_path)
    
    mask_image = mask.mask_generator(file_path, True)

    # Get the height and width of the images
    height, width = document_image.shape[:2]


    td:list = []
    tm:list = []

    # Loop over each transform
    for i in range(num_of_transforms):
        # Check if the transformation should be applied based on the probability
        if np.random.uniform(0, 1) < probality:
            # Define the source and destination points for the perspective transformation
            src_pts = np.float32([[0, 0], [width - 1, 0], [0, height - 1], [width - 1, height - 1]])#type: ignore
            dst_pts = np.float32([[np.random.randint(0, width*0.2), np.random.randint(0, height*0.2)],#type: ignore
                      [np.random.randint(width*0.8, width-1), np.random.randint(0, height*0.2)],#type: ignore
                      [np.random.randint(0, width*0.2), np.random.randint(height*0.8, height-1)],#type: ignore
                      [np.random.randint(width*0.8, width-1), np.random.randint(height*0.8, height-1)]]) #type: ignore

            # Generate a random transformation matrix
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)#type: ignore

            # Apply the transformation to the document and mask images
            transformed_document = cv2.warpPerspective(document_image, M, (width, height))
            transformed_mask = cv2.warpPerspective(mask_image, M, (width, height))

            # Storing the transformed images in list
            td.append(transformed_document)
            tm.append(transformed_mask)
        else:
            # Storing the transformed images in list
            td.append(document_image)
            tm.append(mask_image)
    
    return td,tm