import cv2
import os
import numpy as np
import argparse
import dlib
import imutils
import delaunay
import morph
import time

def crop_image(imageI_original, imageE_original, face_detector, landmarkI, landmarkE):
    imageI = imageI_original.copy()
    imageE = imageE_original.copy()

    hI, wI = imageI.shape[0], imageI.shape[1]
    hE, wE = imageE.shape[0], imageE.shape[1]

    xminI, yminI = landmarkI[0:81].min(axis=0)
    xmaxI, ymaxI = landmarkI[0:81].max(axis=0)

    xminE, yminE = landmarkE[0:81].min(axis=0)
    xmaxE, ymaxE = landmarkE[0:81].max(axis=0)
    # print(landmarkI, landmarkE)
    # print('rawI: {}'.format([yminI, ymaxI, xminI, xmaxI]))
    # print('rawE: {}'.format([yminE, ymaxE, xminE, xmaxE]))
    targetY1 = int(yminE-yminI)
    targetY2 = int(hI-ymaxI+ymaxE)
    targetX1 = int(xminE-xminI)
    targetX2 = int(wI-xmaxI+xmaxE)
    # print('before crop: {}'.format([targetY1, targetY2, targetX1, targetX2]))
    if targetY1<0:
        targetY1=0
    if targetY2>hE:
        targetY2=hE
    if targetX1<0:
        targetX1=0
    if targetX2>wE:
        targetX2=wE
    # print('crop: {}'.format([targetY1, targetY2, targetX1, targetX2]))
    return [targetY1, targetY2, targetX1, targetX2]

# funtion for preprocessing image and detect face&landmarks
def face_and_landmark_detect(image_original, face_detector, landmark_predictor_68, landmark_predictor_81, name):
    image = image_original.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect face
    rects = face_detector(gray, 1)

    landmarks = []
    if len(rects) == 0:
        print(name+' No face')
    else:
        landmarks_68 = np.array([[p.x, p.y] for p in landmark_predictor_68(gray, rects[0]).parts()])
        landmarks_81 = np.array([[p.x, p.y] for p in landmark_predictor_81(gray, rects[0]).parts()])
        landmarks = np.r_[landmarks_68, landmarks_81[68:81]]
        # print("landmarks are: {}\n".format(landmarks))
        # for idx, point in enumerate(landmarks):
        #     pos = (point[0], point[1])
        #     cv2.circle(image, pos, 2, color=(139, 0, 0))
        #     cv2.putText(image, str(idx + 1), pos, cv2.FONT_HERSHEY_SIMPLEX, 0.2, (187, 255, 255), 1, cv2.LINE_AA)
    # draw landmarks and index
    # cv2.imwrite("./tmp/landmark_"+name+".png", image)

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
    landmark_predictor_68 = dlib.shape_predictor('../weights/shape_predictor_68_face_landmarks.dat')
    landmark_predictor_81 = dlib.shape_predictor('../weights/shape_predictor_81_face_landmarks.dat')
    t2 = time.time()
    # load the image

    imageI = cv2.imread(args.input_image)
    imageE = cv2.imread(args.example_image)

    input_images_number, example_images_number = 1,1

    t3 = time.time()
    imageI = cv2.resize(imageI, (500, 650))
    imageE = cv2.resize(imageE, (500, 650))
    pointsI = face_and_landmark_detect(imageI, face_detector, landmark_predictor_68, landmark_predictor_81, args.input_image.split('/')[-1])
    pointsE = face_and_landmark_detect(imageE, face_detector, landmark_predictor_68, landmark_predictor_81, args.example_image.split('/')[-1])

    # targetI = crop_image(imageI, face_detector, pointsI)
    targetE = crop_image(imageI, imageE, face_detector, pointsI, pointsE)
    # imageI = imageI[targetI[0]:targetI[1], targetI[2]:targetI[3]]
    imageE = imageE[targetE[0]:targetE[1], targetE[2]:targetE[3]]


    imageE = cv2.resize(imageE, (500, 650))
    cv2.imwrite('./tmp/delete/noncrop_'+args.input_image.split('/')[-1]+'.png', imageI)
    cv2.imwrite('./tmp/delete/crop_'+args.example_image.split('/')[-1]+'.png', imageE)

    pointsI = face_and_landmark_detect(imageI, face_detector, landmark_predictor_68, landmark_predictor_81, args.input_image.split('/')[-1])
    pointsE = face_and_landmark_detect(imageE, face_detector, landmark_predictor_68, landmark_predictor_81, args.example_image.split('/')[-1])
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
    cv2.imwrite('./tmp/improve_crop_new_'+args.input_image.split('/')[-1]+'_'+args.example_image.split('/')[-1]+'.png', image_morph)
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
