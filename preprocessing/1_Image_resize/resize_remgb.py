import cv2
import glob
import os
from os import listdir
from rembg import remove

imdir = 'D:/4_Working/1_6_VirtualTryOn/preprocessing/test_dataset/test/ori_image/'
clodir = 'D:/4_Working/1_6_VirtualTryOn/preprocessing/test_dataset/test/ori_cloth/'
image_result_dir = 'D:/4_Working/1_6_VirtualTryOn/preprocessing/test_dataset/test/image/'
cloth_result_dir = 'D:/4_Working/1_6_VirtualTryOn/preprocessing/test_dataset/test/cloth/'
cloth_mask_dir = 'D:/4_Working/1_6_VirtualTryOn/preprocessing/test_dataset/test/cloth-mask/'
cnt = 0
for images in os.listdir(imdir):
    cnt = cnt + 1
    if (images.endswith(".png") or images.endswith(".jpg") or images.endswith(".jpeg")):
        img = cv2.imread(imdir + images, 1)
        resized = cv2.resize(img, dsize = (768, 1024), fx = 0.5, fy= 0.5, interpolation = cv2.INTER_CUBIC)
        output = remove(resized, bgcolor=[255,255,255,0])
        cv2.imwrite(str(image_result_dir) + str(cnt) +'.jpg', output)

cnt = 0
for images in os.listdir(clodir):
    cnt = cnt + 1
    if (images.endswith(".png") or images.endswith(".jpg") or images.endswith(".jpeg")):
        img = cv2.imread(clodir + images, 1)
        resized = cv2.resize(img, dsize = (768, 1024), fx = 0.5, fy= 0.5, interpolation = cv2.INTER_CUBIC)
        cv2.imwrite(str(cloth_result_dir) + str(cnt) +'.jpg', resized)
