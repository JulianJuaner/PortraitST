import cv2
import os
import numpy as np
import argparse
import dlib
import imutils


# shape2np function
def shape_to_np(shape, dtype="int"):
    # initialize 68 (x,y) coordinates
    coordinates = np.zeros((68, 2), dtype=dtype)

    for i in range(68):
        coordinates[i] = (shape.part(i).x, shape.part(i).y)

    return coordinates

# visualize landmarks for check
def visualize_facial_landmarks(image, shape, colors=None, alpha=0.75):
    overlay = image.copy()
    output = image.copy()

    if colors is None:
        colors = [(19, 199, 109), (79, 76, 240), (230, 159, 23),
                  (168, 100, 168), (158, 163, 32),
                  (163, 38, 32), (180, 42, 220)]



# The main function of aligment.
# METHOD: Facial landmarks + morphing + SIFT flow
if __name__ == '__main__':
    print('There is the main function of alignment part')
    arg = argparse.ArgumentParser()
    arg.add_argument("-p", "--shape-predictor", required=True, help="path to facial landmark predictor")
    arg.add_argument("-i", "--input_image", required=True, help="path to image")
    args = arg.parse_args()

    # initialize dlib's face detector and use facial landmark predictor
    face_detector = dlib.get_frontal_face_detector()
    landmark_predictor = dlib.shape_predictor(args.shape_predictor)

    # load the imagea
    image = cv2.imread(args.input_image)
    image = imutils.resize(image, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect face
    rects = face_detector(gray, 1)

    shape = landmark_predictor(gray, reats[0].rect)
    shape = shape_to_np(shape)

    output = visualize_facial_landmarks(image, shape)
    cv2.imshow("landmarks", output)
    cv2.waitKey(0)
