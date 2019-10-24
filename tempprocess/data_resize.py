
# A temporary dataset generator for image datasets.
import cv2
import os
import numpy
import sys
sys.path.insert(1, '../gainmap')
from dataset import make_dataset

# Function: resize a dataset and save to another folder.
def resize(root, num):
    folderA = os.path.join(root, '0')
    folderB = os.path.join(root, '1')
    folderA = make_dataset(folderA)
    folderB = make_dataset(folderB)
    os.makedirs(os.path.join(root, '0_resize'), exist_ok=True)
    os.makedirs(os.path.join(root, '1_resize'), exist_ok=True)

    for i in range(num):
        if i%100 == 0:
            print(i)
        imgA = cv2.imread(folderA[i])
        imgB = cv2.imread(folderB[i])
        imgA = cv2.resize(imgA, (128, 128))
        imgB = cv2.resize(imgB, (128, 128))

        cv2.imwrite(
            "../gainmap/checkpoints/dataset/0_resize/%d.png"%(i),
            imgA
        )
        cv2.imwrite(
            "../gainmap/checkpoints/dataset/1_resize/%d.png"%(i),
            imgB
        )

resize('../gainmap/checkpoints/dataset', 1200)