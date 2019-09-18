import cv2
import os
import numpy as np
import argparse
import dlib
import imutils
import delaunay
import morph

# funtion for preprocessing image and detect face&landmarks
def process_and_detect(image_original, face_detector, landmark_predictor, name):
    image = image_original.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect face
    rects = face_detector(gray, 1)

    landmarks = []
    if len(rects) == 0:
        print(name+' No face')
    else:
        landmarks = np.array([[p.x, p.y] for p in landmark_predictor(gray, rects[0]).parts()])
        # print("landmarks are: {}\n".format(landmarks))
        for idx, point in enumerate(landmarks):
            pos = (point[0], point[1])

            # cv2.circle(image, pos, 2, color=(139, 0, 0))
            # cv2.putText(image, str(idx + 1), pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    # draw landmarks and index
    # cv2.imwrite("./tmp/tmp.png", image)

    # -------------------------------------------------------------------------------------
    # delaunay trianglations
    # additional 8 points
    additional = np.array([
     [0, 0],
     [0, int(image.shape[0]/2)],
     [0, image.shape[0]-1],
     [int(image.shape[1]/2), image.shape[0]-1],
     [image.shape[1]-1, image.shape[0]-1],
     [image.shape[1]-1, int(image.shape[0]/2)],
     [image.shape[1]-1, 0],
     [int(image.shape[1]/2), 0]])

    landmarks = np.r_[landmarks, additional]

    subdiv = cv2.Subdiv2D((0, 0, image.shape[1], image.shape[0]))

    for point in landmarks:
        subdiv.insert((point[0], point[1]))
    triangle_list = delaunay.get_triangles(subdiv, image.shape)
    delaunay.draw_delaunay(image, triangle_list)
    triangle_list = np.array(triangle_list)
    cv2.imwrite('./tmp/'+'delaunay_'+name, image)
    triangle_index = []
    for i in range(len(triangle_list)):
        sub_triangle_index = []
        pointI1 = [triangle_list[i, 0], triangle_list[i, 1]]
        pointI2 = [triangle_list[i, 2], triangle_list[i, 3]]
        pointI3 = [triangle_list[i, 4], triangle_list[i, 5]]
        for idx, point in enumerate(landmarks):
            if pointI1[0]==point[0] and pointI1[1]==point[1]:
                sub_triangle_index.append(idx)
                break
        for idx, point in enumerate(landmarks):
            if pointI2[0]==point[0] and pointI2[1]==point[1]:
                sub_triangle_index.append(idx)
                break
        for idx, point in enumerate(landmarks):
            if pointI3[0]==point[0] and pointI3[1]==point[1]:
                sub_triangle_index.append(idx)
                break
        triangle_index.append(sub_triangle_index)
    print('index:{}'.format(np.array(triangle_index)))
    return np.array(triangle_index), landmarks

# The main function of aligment.
# METHOD: Facial landmarks + morphing + SIFT flow
if __name__ == '__main__':
    # print('There is the main function of alignment part')
    arg = argparse.ArgumentParser()
    # arg.add_argument("-p", "--shape_predictor", required=True, help="path to facial landmark predictor")
    arg.add_argument("-i", "--input_image", required=True, help="path to image")
    arg.add_argument("-e", "--example_image", required=True, help="path to example image")
    args = arg.parse_args()

    # initialize dlib's face detector and use facial landmark predictor
    face_detector = dlib.get_frontal_face_detector()
    landmark_predictor = dlib.shape_predictor('../weights/shape_predictor_68_face_landmarks.dat')

    # load the image
    imageI = cv2.imread(args.input_image)
    imageI = cv2.resize(imageI, (500, 750))
    imageE = cv2.imread(args.example_image)
    imageE = cv2.resize(imageE, (500, 750))

    imageE_triangle_list, pointsE = process_and_detect(imageE, face_detector, landmark_predictor, args.example_image.split('/')[-1])
    imageI_triangle_list, pointsI = process_and_detect(imageI, face_detector, landmark_predictor, args.input_image.split('/')[-1])
    # ------------------------------------------------------------------------------------
    # morphing
    # initialize the final morphing image
    image_morph = np.zeros(imageE.shape, dtype = imageE.dtype)
    # print(imageI_triangle_list)
    # print('input_image triangle numbers: {}\nexample_image triangle numbers: {}'.format(len(imageI_triangle_list), len(imageE_triangle_list)))
    # print(type(imageI_triangle_list))
    count = 0

    for i in range(142):
        triangleI = [tuple(pointsI[imageI_triangle_list[i, 0]]), tuple(pointsI[imageI_triangle_list[i, 1]]), tuple(pointsI[imageI_triangle_list[i, 2]])]
        triangleE = [tuple(pointsE[imageI_triangle_list[i, 0]]), tuple(pointsE[imageI_triangle_list[i, 1]]), tuple(pointsE[imageI_triangle_list[i, 2]])]

        count+=1
        print('morph {} triangleI: {} andE {}'.format(count, imageI_triangle_list[i], imageE_triangle_list[i]))
        morph.morphTriangle(imageI, imageE, image_morph, triangleI, triangleE)

    cv2.imwrite('./tmp/morph_tmp.png', image_morph)
