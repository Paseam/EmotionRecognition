# -*- coding: utf-8 -*-
# Created on Wed Dec 27 2017 10:22:44
# Author: WuLC
# EMail: liangchaowu5@gmail.com

import os
import cv2
import dlib
import fire

def detect_face_with_dlib(img_path, face_path, face_detector):
    img = cv2.imread(img_path)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    detected = False
    faces = face_detector(img, 1)
    for face in faces:
        left_top, right_bottom = (face.left(), face.top()), (face.right(), face.bottom())
        detected = True
        cv2.rectangle(img, left_top, right_bottom, (0, 255, 255), 2)

        # save the main face
        if face.right() - face.left() > 100:
            cropped_img = img[face.top():face.bottom(), face.left():face.right()]
            cv2.imwrite(face_path, cropped_img)
            
    if detected:
        return True
    else:
        print('{0} detect no face'.format(img_path))
        return False


def crop_faces(dataset_path):
    dlib_face_detector = dlib.get_frontal_face_detector()
    for i in range(1, 11):
        img_dir = '{0}/original/Group{1}/'.format(dataset_path, i)
        des_dir = '{0}/face/Group{1}/'.format(dataset_path, i)
        if not os.path.exists(des_dir):
            os.makedirs(des_dir)
        for img_name in os.listdir(img_dir):
            img_path = img_dir +  img_name
            face_path = des_dir + img_name
            detect_face_with_dlib(img_path, face_path, dlib_face_detector)


def process_train_set(dataset_path):
    dataset_path = 'F:/FaceExpression/TrainSet/CKPlus_10G'
    dataset_path = 'F:/FaceExpression/TrainSet/KDEF_10G'
    crop_faces(dataset_path)


def process_test_set(dataset_path):
    dlib_face_detector = dlib.get_frontal_face_detector()
    img_dir = '{0}/original/'.format(dataset_path)
    des_dir = '{0}/face/'.format(dataset_path)
    for img_name in os.listdir(img_dir):
        img_path = img_dir + img_name
        face_path = des_dir + img_name
        detect_face_with_dlib(img_path, face_path, dlib_face_detector)

if __name__ == '__main__':
    fire.Fire()