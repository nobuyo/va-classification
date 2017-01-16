import os
import sys
import glob
import cv2
from PIL import Image
import dlib
from joblib import Parallel, delayed
from time import time

"""
INPUT_DIRは(1.画像を収集する)で取得した画像のディレクトリ名
OUTPUT_DIRは出力ディレクトリ名(フォルダ構成はINPUT_DIRと同じになる)
"""

detector = dlib.get_frontal_face_detector()
INPUT_DIR = './aimi'
OUTPUT_DIR = './aimi_crop'
IMAGE_SIZE = 128

print("start cropping...")

# 各女優のディレクトリのリストを取得
dir_list = os.listdir(INPUT_DIR)
already = os.listdir(INPUT_DIR)

# for x in range(1,already.length):
#     dir_list.pop(0)

for i, dir_name in enumerate(dir_list):
    if not os.path.isdir(INPUT_DIR + '/' +dir_name):
        continue
    if not os.path.exists(os.path.join(OUTPUT_DIR, dir_name)):
        os.mkdir(os.path.join(OUTPUT_DIR, dir_name))
    image_files = glob.glob(os.path.join(INPUT_DIR, dir_name, "*.jpg"))

    for j, image_file in enumerate(image_files):
        print(image_file)
        try:
            img = cv2.imread(image_file)
            dets = detector(img, 1)
            open_img = Image.open(image_file)
        except Exception as e:
            print('file broken: skipped: %s %s' %(image_file,e))

        for k, d in enumerate(dets):
            # サイズが80以下の画像はスキップする
            if d.right()-d.left() < 80 or d.bottom()-d.top() < 80:
                continue
            image_file = image_file.replace(INPUT_DIR, OUTPUT_DIR)
            # １つの画像に復数の顔があると出力のファイル名がかぶるので変更
            output_file = image_file.replace('.jpg', '_'+str(k)+'.jpg')
            try:
                cropped_img = open_img.crop((d.left(), d.top(), d.right(), d.bottom()))
                cropped_img.resize((IMAGE_SIZE,IMAGE_SIZE)).save(output_file, 'JPEG', quality=100, optimize=True)
            except Exception as e:
                print('file broken: skipped: %s %s' %(image_file,e))
