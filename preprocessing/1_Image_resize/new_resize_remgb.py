import cv2
import glob
import os
from os import listdir
#from rembg import remove

imdir = 'D:/4_Working/test_image/ori_image/'
clodir = 'D:/4_Working/test_image/ori_cloth/'
image_result_dir = 'D:/4_Working/test_dataset/test/image/'
cloth_result_dir = 'D:/4_Working/test_dataset/test/cloth/'
cnt = 0
for images in os.listdir(imdir):
    cnt = cnt + 1
    if (images.endswith(".png") or images.endswith(".jpg") or images.endswith(".jpeg")):
        img = cv2.imread(imdir + images, 1)
        resized = cv2.resize(img, dsize = (768, 1024), fx = 0.5, fy= 0.5, interpolation = cv2.INTER_CUBIC)
        #output = remove(resized, bgcolor=[255,255,255,0])
        cv2.imwrite(str(image_result_dir) + images, resized)
for images in os.listdir(clodir):
    cnt = cnt + 1
    if (images.endswith(".png") or images.endswith(".jpg") or images.endswith(".jpeg")):
        img = cv2.imread(clodir + images, 1)
        resized = cv2.resize(img, dsize = (768, 1024), fx = 0.5, fy= 0.5, interpolation = cv2.INTER_CUBIC)
        #output = remove(resized, bgcolor=[255,255,255,0])
        cv2.imwrite(str(cloth_result_dir) + images, resized)
# cnt = 0
# for images in os.listdir(clodir):
#     cnt = cnt + 1
#     if (images.endswith(".png") or images.endswith(".jpg") or images.endswith(".jpeg")):
#         img = cv2.imread(clodir + images, 1)
#         resized = cv2.resize(img, dsize = (768, 1024), fx = 0.5, fy= 0.5, interpolation = cv2.INTER_CUBIC)
#         cv2.imwrite(str(cloth_result_dir) + str(cnt) +'.jpg', resized)
