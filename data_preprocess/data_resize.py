
# A temporary dataset generator for image datasets.
import cv2
import os
import numpy
import sys
sys.path.insert(1, '../gainmap')
from dataset import make_dataset

# Function: resize a dataset and save to another folder.
def resize(root, num):
    folderA = os.path.join(root, 'target')
    folderB = os.path.join(root, 'example')
    folderA = make_dataset(folderA)
    folderB = make_dataset(folderB)
    os.makedirs(os.path.join(root, 'target_resize'), exist_ok=True)
    os.makedirs(os.path.join(root, 'example_resize'), exist_ok=True)

    for i in range(num):
        if i%100 == 0:
            print(i)
        imgA = cv2.imread(folderA[i])
        imgB = cv2.imread(folderB[i])
        imgA = cv2.resize(imgA, (128, 128))
        imgB = cv2.resize(imgB, (128, 128))

        cv2.imwrite(
            folderA[i].replace('target', 'target_resize'),
            imgA
        )
        cv2.imwrite(
            folderB[i].replace('example', 'example_resize'),
            imgB
        )
def guideImage(filename):
    img = cv2.imread(filename)
    size1 = (256, 336)
    size2 = (1614, 2119)
    img = cv2.resize(img, size1)
    img = cv2.resize(img, size2)
    cv2.imwrite(
            'guideimage2.png',
            img
    )
guideImage('guideimage2.png')
#resize('../datasets/newset', 1200)