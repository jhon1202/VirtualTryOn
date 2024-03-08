import os
import cv2
import sys
import math
import pickle
import timeit
import imutils
import scipy.io
import argparse
import datetime
import numpy as np
import tkinter as tk
from imutils import paths
from tkinter import filedialog
import matplotlib.pyplot as plt

def getCams(configdir):

    cams = []
    for i in range(4):
        cam_filepath = os.path.join(configdir, 'camera%02d' % (i + 1), 'calibration.mat')
        cam_np = scipy.io.loadmat(cam_filepath)['cam']
        cam_dict = {}
        for key in cam_np.dtype.fields.keys():  # ['K', 'R', 'fc', 'cc', 'alpha_c', 'kc', 'T', 'P', 'H', ]
            cam_dict[key] = cam_np[key][0][0]
        cam_dict['fc'] = cam_dict['fc'].flatten()
        cam_dict['kc'] = cam_dict['kc'].flatten()
        cam_dict['cc'] = cam_dict['cc'].flatten()
        cams.append(cam_dict)
    return cams

def getCalibration(camNumber, image):
    cams = getCams(configdir)
    cameraMatrix1 = cams[camNumber]['K']
    distCoeff1 = cams[camNumber]['kc']
    rvecs1 = cams[camNumber]['R']
    tvecs1 = cams[camNumber]['R']    
    h,  w = image.shape[:2]
    alpha_c1 = cams[camNumber]['alpha_c']
    newCameraMatrix1, roi1 = cv2.getOptimalNewCameraMatrix(cameraMatrix1, distCoeff1, (w,h), int(alpha_c1), (w,h))
    dst = cv2.undistort(image, cameraMatrix1, distCoeff1, None, newCameraMatrix1)
    x, y, w, h = roi1
    return dst;


def feature_matching(img1, img2, savefig=False):
    sift = cv2.xfeatures2d_SIFT.create()
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50) 
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches2to1 = flann.knnMatch(des2,des1,k=2)

    matchesMask_ratio = [[0,0] for i in range(len(matches2to1))]
    match_dict = {}
    for i,(m,n) in enumerate(matches2to1):
        if m.distance < 0.7*n.distance:
            matchesMask_ratio[i]=[1,0]
            match_dict[m.trainIdx] = m.queryIdx

    good = []
    recip_matches = flann.knnMatch(des1,des2,k=2)
    matchesMask_ratio_recip = [[0,0] for i in range(len(recip_matches))]

    for i,(m,n) in enumerate(recip_matches):
        if m.distance < 0.7*n.distance: # ratio
            if m.queryIdx in match_dict and match_dict[m.queryIdx] == m.trainIdx: 
                good.append(m)
                matchesMask_ratio_recip[i]=[1,0]
                
    if savefig:
        draw_params = dict(matchColor = (0,255,0), singlePointColor = (255,0,0), matchesMask = matchesMask_ratio_recip, flags = 0)
        img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,recip_matches,None,**draw_params)

        plt.figure(),plt.xticks([]),plt.yticks([])
        plt.imshow(img3,)
        plt.savefig("feature_matching.png",bbox_inches='tight')

    return ([ kp1[m.queryIdx].pt for m in good ],[ kp2[m.trainIdx].pt for m in good ])

def getTransform(src, dst, method='affine'):
    pts1,pts2 = feature_matching(src,dst)

    src_pts = np.float32(pts1).reshape(-1,1,2)
    dst_pts = np.float32(pts2).reshape(-1,1,2)

    if method == 'affine':
        M, mask = cv2.estimateAffine2D(src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold=5.0)

    if method == 'homography':
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 6.0)

    matchesMask = mask.ravel().tolist()

    return (M, pts1, pts2, mask)
    
def merge2images(left, right, left_mask, right_mask):
    masked1 = left * left_mask / 255
    masked2 = right * right_mask / 255  
    result = masked1 + masked2
    return result.astype('uint8')

def Perspective_warping2(img1, img2, M):
    warpOut = cv2.warpPerspective(img2, M, (img1.shape[1],img1.shape[0]))
    result = merge2images(img1, warpOut)
    return result

def Perspective_warping3(img1, img2, img3, M, M1):
    out1 = cv2.warpPerspective(img3, M, (img1.shape[1],img1.shape[0]))
    out2 = cv2.warpPerspective(img2, M1, (img1.shape[1],img1.shape[0]))
    res1 = merge2images(img1, out1)    
    res2 = merge2images(out2, res1)
    return res2

def FourPointTransform(image, pin):   
    w , h = image.shape[1], image.shape[0]
    coord = [ 
        [int(w * pin[0][0]), int(h * pin[0][1])], 
        [int(w * pin[1][0]), int(h * pin[1][1])], 
        [int(w * pin[2][0]), int(h * pin[2][1])], 
        [int(w * pin[3][0]), int(h * pin[3][1])]]  
    widthA = np.sqrt(((coord[3][0] - coord[2][0]) ** 2) + ((coord[3][1] - coord[2][1]) ** 2))
    widthB = np.sqrt(((coord[1][0] - coord[0][0]) ** 2) + ((coord[1][1] - coord[0][1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))     
    heightA = np.sqrt(((coord[1][0] - coord[3][0]) ** 2) + ((coord[1][1] - coord[3][1]) ** 2))
    heightB = np.sqrt(((coord[0][0] - coord[2][0]) ** 2) + ((coord[0][1] - coord[2][1]) ** 2))
    maxHeight = max(int(heightA), int(heightB)) 
    pts1 = np.float32(coord)    
    pts2 = np.float32([[0, 0], [maxWidth-1, 0], [0, maxHeight-1], [maxWidth-1, maxHeight-1]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result_cv = cv2.warpPerspective(image, matrix, (maxWidth,maxHeight))     
    return result_cv

currentTimePos = datetime.datetime.now()
def resetTimer():
    global currentTimePos
    currentTimePos = datetime.datetime.now()
    
def printTimeElapsed( comment ):
    timeDiff = datetime.datetime.now() - currentTimePos
    millisec = timeDiff.microseconds
    print(comment + " (%d ms)" % int(timeDiff.total_seconds() * 1000))
    resetTimer()
    
def main(configdir, videodir):
    resetTimer()
    
    print("Syncronizing ...")
    videodir += "\\"
    allLines , allts = {}, {}    
    allLines[0] = open(videodir + "video01_000.log").readlines()
    allLines[1] = open(videodir + "video02_000.log").readlines()
    allLines[2] = open(videodir + "video03_000.log").readlines()
    allLines[3] = open(videodir + "video04_000.log").readlines()
    
    for i in range(0, 4):
        allts[i] = []
        for line in allLines[i]:
            allts[i].append( float(line.split(" ")[1]))        
    
    frametoforword = np.zeros((4))
    maxts = max(allts[0][0], allts[1][0], allts[2][0], allts[3][0])
    
    for i in range(0, 4):
        for j in range(len(allts[i])):
            if(allts[i][j] >= maxts):
                frametoforword[i] = j
                print("  Video %d found starting point at frame %d" % (i,j))
                break      
    frametoforword += 130
    
    
    # read videos
    cap1 = cv2.VideoCapture(videodir + 'video01_000.ts')
    cap2 = cv2.VideoCapture(videodir + 'video02_000.ts')
    cap3 = cv2.VideoCapture(videodir + 'video03_000.ts')
    cap4 = cv2.VideoCapture(videodir + 'video04_000.ts')
    total1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT)) - int(frametoforword[0])
    total2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT)) - int(frametoforword[1])
    total3 = int(cap3.get(cv2.CAP_PROP_FRAME_COUNT)) - int(frametoforword[2])
    total4 = int(cap4.get(cv2.CAP_PROP_FRAME_COUNT)) - int(frametoforword[3])
    
    # set min frame count to output
    counts = []
    counts.append(total1)
    counts.append(total2)
    counts.append(total3)
    counts.append(total4)
    # counts.append(1200)
    minCount = min(counts)
    
    # sync frames
    for i in range(int(frametoforword[0])): cap1.read() # SYNC MANUALLY
    for i in range(int(frametoforword[1])): cap2.read() # SYNC MANUALLY
    for i in range(int(frametoforword[2])): cap3.read() # SYNC MANUALLY
    for i in range(int(frametoforword[3])): cap4.read() # SYNC MANUALLY
    printTimeElapsed("  Time elapsed for sync \t :")
    
    # initialize output video setting
    outputFPS = 60.0
    outputSize = (int(5989), int(2773))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    vid_writer = cv2.VideoWriter('output (%d x %d) %d fps.avi' % (outputSize[0] , outputSize[1], outputFPS), fourcc, outputFPS, outputSize)    

    initialized = False    
    if cap1.isOpened() and cap2.isOpened() and cap3.isOpened() and cap4.isOpened():
        for cnt in range(minCount):
            resetTimer()
            if(not initialized):
                print("Initializing ...")      
            else:
                print("Processing frame%4d." % cnt)    
                
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()
            ret3, frame3 = cap3.read()
            ret4, frame4 = cap4.read()
            # printTimeElapsed("  Time elapsed for read \t :")
            
            img1 = getCalibration(0, frame1)
            img2 = getCalibration(1, frame2)
            img3 = getCalibration(2, frame3)
            img4 = getCalibration(3, frame4)
            # printTimeElapsed("  Time elapsed for calibration \t :")
            
            # warp perspective 3 left images (img2, img1, img3)
            img2 = cv2.copyMakeBorder(img2, 0, 0, int(img2.shape[0]/4), int(img2.shape[0]/4), cv2.BORDER_CONSTANT) 
            if(not initialized): (mm1, _ , _ , _ ) = getTransform(img3, img2,'homography')
            if(not initialized): (mm2, _ , _ , _ ) = getTransform(img1, img2,'homography')
            outLeft1 = cv2.warpPerspective(img3, mm1, (img2.shape[1], img2.shape[0]))
            outLeft2 = cv2.warpPerspective(img1, mm2, (img2.shape[1], img2.shape[0]))
            if(not initialized): img2_mask, outLeft1_mask = getMaskForMerge2(img2, outLeft1)            
            resLeft = merge2images(img2, outLeft1, img2_mask, outLeft1_mask)
            if(not initialized): outLeft2_mask, resLeft_mask = getMaskForMerge2(outLeft2, resLeft)
            cropLeft = merge2images(outLeft2, resLeft, outLeft2_mask, resLeft_mask)       
            if(initialized) : printTimeElapsed("  Time elapsed for left \t :")
            #cv2.imwrite("crop-left.png", cropLeft)
            
        
            # warp perspective 3 right images (img3, img2, img4)
            img3 = cv2.copyMakeBorder(img3, 0, 0, int(img2.shape[0]/4), int(img2.shape[0]/4), cv2.BORDER_CONSTANT)   
            if(not initialized): (mm3, _ , _ , _ ) = getTransform(img4, img3,'homography')
            if(not initialized): (mm4, _ , _ , _ ) = getTransform(img2, img3,'homography')     
            outRight1 = cv2.warpPerspective(img4, mm3, (img3.shape[1], img3.shape[0]))
            outRight2 = cv2.warpPerspective(img2, mm4, (img3.shape[1], img3.shape[0]))            
            if(not initialized): img2_mask, outRight1_mask = getMaskForMerge2(img3, outRight1)            
            resRight = merge2images(img3, outRight1, img2_mask, outRight1_mask)
            if(not initialized): out2_mask, resRight_mask = getMaskForMerge2(outRight2, resRight)
            cropRight = merge2images(outRight2, resRight, out2_mask, resRight_mask)
            if(initialized) : printTimeElapsed("  Time elapsed for right \t :")
            #cv2.imwrite("crop-right.png", cropRight)
                 
            # Merge 2
            cropLeft = cv2.copyMakeBorder(cropLeft, 0, 0, 0, cropLeft.shape[1], cv2.BORDER_CONSTANT)            
            if(not initialized): (mm5, _ , _ , _ ) = getTransform(cropRight, cropLeft,'homography')
            cropRightWarp = cv2.warpPerspective(cropRight, mm5, (cropLeft.shape[1],cropLeft.shape[0]))
            if(not initialized): cropLeftMask, cropRightMask = getMaskForMerge2(cropLeft, cropRightWarp)
            merged2 = merge2images(cropLeft, cropRightWarp, cropLeftMask, cropRightMask)            
            transformed2 = FourPointTransform(merged2, [[0.07, 0.37], [0.79,0.201],[0.03,0.807],[1,1]])       
            if(initialized) : printTimeElapsed("  Time elapsed for merge \t :")
            
            resultSize = transformed2.shape
            resized = cv2.resize(transformed2, outputSize)
            vid_writer.write(resized)
            vid_writer.write(resized)
            # if(initialized) : printTimeElapsed("  Time elapsed for resize \t :")
                        
            # cv2.imwrite("result.png" , resized)
            if(not initialized):
                printTimeElapsed("  Time elased for init \t : ")
            initialized = True
            # break
            
    vid_writer.release()    
    print("Finished ... ")

# @click.command()
# @click.option('--configdir', help='Path to the config directory of the relevant msr-file.')
if __name__ == '__main__':
    root = tk.Tk()
    root.withdraw()
    configdir = "config" # configdir = filedialog.askdirectory()
    videodir = "video"   # videodir = filedialog.askdirectory()
    
    main(configdir, videodir)
