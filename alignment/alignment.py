import cv2
import os
import numpy as np
import argparse
import dlib
import imutils
import delaunay
import morph
import time

def crop_image(image_original, face_detector):
    image = image_original.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # detect face
    rects = face_detector(gray, 1)
    # print('y1,y2,x1,x2:{} {} {} {}'.format(rects[0].top(), rects[0].bottom(), rects[0].left(), rects[0].right()))
    targetY1 = int(rects[0].top()-0.5*(rects[0].bottom()-rects[0].top()))
    targetY2 = int(rects[0].bottom()+0.3*(rects[0].bottom()-rects[0].top()))
    targetX1 = int(rects[0].left()-0.25*(rects[0].right()-rects[0].left()))
    targetX2 = int(rects[0].right()+0.25*(rects[0].right()-rects[0].left()))
    if targetY1<0:
        targetY1=0
    if targetY2>gray.shape[0]:
        targetY2=gray.shape[0]
    if targetX1<0:
        targetX1=0
    if targetX2>gray.shape[1]:
        targetX2=gray.shape[1]
    # cropped_image = gray[targetY1:targetY2, targetX1:targetX2]
    # cv2.imwrite("./tmp/cropped.png", cropped_image)
    return [targetY1, targetY2, targetX1, targetX2]

# funtion for preprocessing image and detect face&landmarks
def face_and_landmark_detect(image_original, face_detector, landmark_predictor, name):
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
    return landmarks

def get_triangle_and_index(landmarks, image, name):
    subdiv = cv2.Subdiv2D((0, 0, image.shape[1], image.shape[0]))

    for point in landmarks:
        subdiv.insert((point[0], point[1]))
    triangle_list = delaunay.get_triangles(subdiv, image.shape)
    delaunay.draw_delaunay(image, triangle_list)
    triangle_list = np.array(triangle_list)
    # cv2.imwrite('./tmp/'+'delaunay_'+name, image)
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

    return np.array(triangle_index)

def load_image_and_morph(args):
    t1 = time.time()
    # initialize dlib's face detector and use facial landmark predictor
    face_detector = dlib.get_frontal_face_detector()
    landmark_predictor = dlib.shape_predictor('../weights/shape_predictor_68_face_landmarks.dat')
    t2 = time.time()
    # load the image

    imageI = cv2.imread(args.input_image)
    imageE = cv2.imread(args.example_image)
    # targetI = crop_image(imageI, face_detector)
    # targetE = crop_image(imageE, face_detector)
    # imageI = imageI[targetI[0]:targetI[1], targetI[2]:targetI[3]]
    # imageE = imageE[targetE[0]:targetE[1], targetE[2]:targetE[3]]

    imageI = cv2.resize(imageI, (500, 650))
    imageE = cv2.resize(imageE, (500, 650))
    # cv2.imwrite('./tmp/resizedI.png', imageI)
    # cv2.imwrite('./tmp/resizedE.png', imageE)
    input_images_number, example_images_number = 1,1

    t3 = time.time()
    pointsI = face_and_landmark_detect(imageI, face_detector, landmark_predictor, args.input_image.split('/')[-1])
    pointsE = face_and_landmark_detect(imageE, face_detector, landmark_predictor, args.example_image.split('/')[-1])
    imageI_triangle_list = get_triangle_and_index(pointsI, imageI, args.input_image.split('/')[-1])
    t4 = time.time()

    # ------------------------------------------------------------------------------------
    # morphing
    # initialize the final morphing image
    image_morph = np.zeros(imageE.shape, dtype = imageE.dtype)

    for i in range(imageI_triangle_list.shape[0]):
        triangleI = [tuple(pointsI[imageI_triangle_list[i, 0]]), tuple(pointsI[imageI_triangle_list[i, 1]]), tuple(pointsI[imageI_triangle_list[i, 2]])]
        triangleE = [tuple(pointsE[imageI_triangle_list[i, 0]]), tuple(pointsE[imageI_triangle_list[i, 1]]), tuple(pointsE[imageI_triangle_list[i, 2]])]
        morph.morphTriangle(imageI, imageE, image_morph, triangleI, triangleE)

    t5 = time.time()
    print('Complete load model in {}s'.format(round(t2-t1, 2)))
    print('Complete load image_resize in {}s'.format(round(t3-t2, 2)))
    print('Complete face_landmarks_triangle_index in {}s'.format(round(t4-t3, 2)))
    print('Complete morph in {}s'.format(round(t5-t4, 2)))
    cv2.imwrite('./tmp/morph_crop.png', image_morph)
    return input_images_number, example_images_number

def get_filelist(dirpath):
    file_list = []
    suffix = ['jpg', 'png', 'bmp', 'jpeg', 'JPG', 'PNG', 'BMP', 'JPEG']
    for r, ds, fs in os.walk(dirpath):
        for fn in fs:
            if os.path.splitext(fn)[1] in suffix:
                fname = os.path.join(r, fn)
                file_list.append(fname)
    return file_list


# The main function of aligment.
# METHOD: Facial landmarks + morphing + SIFT flow
if __name__ == '__main__':
    start = time.time()
    # print('There is the main function of alignment part')
    arg = argparse.ArgumentParser()
    # arg.add_argument("-p", "--shape_predictor", required=True, help="path to facial landmark predictor")
    arg.add_argument("-i", "--input_image", required=True, help="path to image")
    arg.add_argument("-e", "--example_image", required=True, help="path to example image")
    arg.add_argument("-id", "--input_dir", required=False, help="path to image directory")
    arg.add_argument("-ed", "--example_dir", required=False, help="path to example image directory")
    args = arg.parse_args()

    input_images_number, example_images_number = load_image_and_morph(args)

    end = time.time()
    # statistic
    print('Complete with {} input_images and {} example_images in {}s'.format(1, 1, round(end-start, 2)))
