'''
Uses weights and models implementation' from
https://github.com/deepinsight/insightface
'''

import imutils
import os
import cv2
from os import path, listdir, makedirs
import argparse
import numpy as np
import glob
from multiprocessing import Pool
from os import path, makedirs
from sklearn.metrics.pairwise import cosine_similarity
import sys
sys.path.insert(0, '../insightface/deploy/')
import face_model
import time


def rotate(mat, angle):
    if angle==0:
        return mat
    height, width = mat.shape[:2]
    image_center = (width/2, height/2)
    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)
    abs_cos = abs(rotation_mat[0,0]) 
    abs_sin = abs(rotation_mat[0,1])
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat

def read_image_from_folder(folder_name,f= None):
    print("reading...")
    images = []
    for path, subdirs, files in os.walk(folder_name):
        for name in files:
            filename = os.path.join(path, name)
            img = cv2.imread(filename)
            if f is not None:
                f.write(str(img.shape[0])+" "+str(img.shape[1])+"\n")
            images.append(img)
    print("end read")
    return images


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract features with CNN')
    parser.add_argument('--folder', '-f', help='image folder')
    parser.add_argument('--savefolder', '-sf', help='image folder')
    parser.add_argument('--thresh', '-th', default=0.5, help='threshold')
    # ArcFace params
    parser.add_argument('--image-size', default='112,112', help='')
    parser.add_argument('--model', help='path to model.',
                        default='../../insightface/models/model-r100-ii/model,0')
    parser.add_argument('--ga-model', default='', help='path to load model.')
    parser.add_argument('--gender_model', default='',
                        help='path to load model.')
    parser.add_argument('--gpu', default=0, type=int, help='gpu id')
    parser.add_argument('--det', default=1, type=int,
                        help='mtcnn: 1 means using R+O, 0 means detect from begining')
    parser.add_argument('--flip', default=0, type=int,
                        help='whether do lr flip aug')
    parser.add_argument('--threshold', default=1.24,
                        type=float, help='ver dist threshold')

    args = parser.parse_args()
    args.gpu = -1
    args.det = 0
    args.model = ""
    img_heights = [1000, 800, 600, 400, 250]
    model = face_model.FaceModel(args)
    print("Thresh:", float(args.thresh))
    first = True
    savepath = args.savefolder
    cp_path = savepath
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    f = open(os.path.join(savepath, "result.txt"), "w+")
    f3 = open(os.path.join(savepath, "face_size.txt"), "w+")
    f2 = open(os.path.join(savepath, "origin_image_size.txt"), "w+")
    first = True
    for img_height in img_heights:
        savepath = os.path.join(cp_path, str(img_height))
        if not os.path.exists(savepath):
            os.mkdir(savepath)
        f.write("img_height: " + str(img_height)+"\n")
        f3.write("img_height: " + str(img_height)+"\n")
        count = 0
        count_rotate = 0
        total_time = 0
        subtract = 0
        rotate_time = 0
        number_images = 0
        for path, subdirs, files in os.walk(args.folder):
            for name in files:
                filename = os.path.join(path, name)
                img = cv2.imread(filename)
                if img is None:
                    continue
                number_images = number_images + 1
                if number_images % 20 ==0:
                    print("Processing....")
                temp = img.copy()
                if first:
                    f2.write(str(temp.shape[0]) + " " + str(temp.shape[1])+ "\n")
                if img.shape[0] >= img_height:
                    img = imutils.resize(img, height=img_height)
                for i in range(4):
                    start = time.time()
                    img = rotate(img, 90*i)
                    end = time.time()
                    if i > 0:
                        count_rotate = count_rotate + 1
                        rotate_time = rotate_time + end - start
                    _, cropped, t, size = model.get_input(img)
                    total_time = total_time + t
                    if cropped is not None:
                        f3.write(str(int(size[0])) + " " + str(int(size[1])) + "\n")
                        break
                if cropped is None:
                    count = count + 1
                    cv2.imwrite(os.path.join(
                        savepath, str(count)+str(img_height)+".jpg"),temp)
                    continue
        first = False
        f3.write("-----------------------------\n")
        f.write("Can not detect: "+str(count)+"\n")
        f.write("Rotate: "+str(count_rotate)+"\n")
        f.write("Avg rotate time: " + str(rotate_time/count_rotate) + "\n")
        f.write("Total times:" + str(total_time)+"\n")
        f.write("Avg times:" + str(total_time/number_images)+"\n")
        f.write("Total images processed: "+ str(number_images)+"\n")
        f.write("--------------------------------------\n")

            # cv2.imshow("cropped", cropped)
            # key = cv2.waitKey(1)
            # if key == ord('q'):
            #     break
    print("STOP")