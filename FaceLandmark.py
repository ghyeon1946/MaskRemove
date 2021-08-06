import cv2
import numpy as np
import dlib
import os

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

path_dir = "FRDataSet2"

def create_file(file_list, file_num):
    for i in range(file_num-1):
        img_file = path_dir + '/' + file_list[i]
        f = open(img_file.rstrip('.jpg') + ".txt", "w")

        frame = cv2.imread(img_file, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = detector(gray)

        for face in faces:
            x1 = face.left()
            y1 = face.top()
            x2 = face.right()
            y2 = face.bottom()

            landmarks = predictor(gray, face)

            for n in range(0, 68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                cv2.circle(frame, (x, y), 6, (255, 0, 0), -1)

                f.write(str(x) + ",")
                f.write(str(y) + '\n')

        f.close()

file_list = os.listdir(path_dir)

create_file(file_list, len(file_list))