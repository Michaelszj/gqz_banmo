import os
import glob
import cv2

def read_images(path):
    image_files = glob.glob(os.path.join(path, '*.png'))  # Change the file extension if needed
    image_files.sort()  # Sort the image files alphabetically
    images = []
    for file in image_files:
        image = cv2.imread(file)  # Read the image using OpenCV or any other library
        images.append(image)
    return images

# Usage
seqname='bailang'
imgpath = f'logdir/eval-{seqname}-1/eval/'
maskpath = 'database/DAVIS/Annotations/Full-Resolution/'+seqname


images = read_images(path)
