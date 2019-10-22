import numpy as np
import cv2
import sys


# Apply affine transform
def applyAffineTransform(src, srcTri, dstTri, size):
    # Given a pair of triangles, find the affine transform.
    warpMat = cv2.getAffineTransform(np.float32(srcTri), np.float32(dstTri) )
    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine(src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101 )
    return dst

def morphTriangle(imageI, imageE, image_morph, triangleI, triangleE):

    # Find bounding rectangle for each triangle
    rectI = cv2.boundingRect(np.float32([triangleI]))
    rectE = cv2.boundingRect(np.float32([triangleE]))
    rect = cv2.boundingRect(np.float32([triangleI]))

    # Offset points by left top corner of the respective rectangles
    tIRect = []
    tERect = []
    tRect = []

    for i in range(3):
        tRect.append(((triangleI[i][0] - rectI[0]),(triangleI[i][1] - rectI[1])))
        tIRect.append(((triangleI[i][0] - rectI[0]),(triangleI[i][1] - rectI[1])))
        tERect.append(((triangleE[i][0] - rectE[0]),(triangleE[i][1] - rectE[1])))


    # Get mask by filling triangle
    mask = np.zeros((rect[3], rect[2], 3), dtype = np.float32)
    cv2.fillConvexPoly(mask, np.int32(tRect), (1.0, 1.0, 1.0), 16, 0)

    # Apply warpImage to small rectangular patches
    # imgIRect = imageI[rectI[1]:rectI[1] + rectI[3], rectI[0]:rectI[0] + rectI[2]]
    imgERect = imageE[rectE[1]:rectE[1] + rectE[3], rectE[0]:rectE[0] + rectE[2]]

    size = (rect[2], rect[3])
    warpImageE = applyAffineTransform(imgERect, tERect, tRect, size)

    # Alpha blend rectangular patches
    imgRect = warpImageE

    # Copy triangular region of the rectangular patch to the output image
    image_morph[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]] = image_morph[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]] * ( 1 - mask ) + imgRect * mask
