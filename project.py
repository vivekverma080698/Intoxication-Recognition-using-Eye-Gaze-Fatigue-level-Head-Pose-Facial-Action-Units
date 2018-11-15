import csv
import numpy
import pandas
from phogDescriptor import PHogFeatures
import csv
import copy
import shutil
from sklearn.cluster import KMeans
from skimage import exposure
import cv2
import pandas
import os,scipy
from time import sleep
import subprocess
from sklearn import preprocessing
import matplotlib.pyplot as plt

#------------------------------------ STAGE 1 --------------------------------------#

# def STAGE1():
#     OPENFACE_LOCATION = '/home/black/OpenFace-master/build/bin/FeatureExtraction'
#
#     input_folder_video = './video'
#     output_folder = './Features'
#
#     files = []
#     for f in os.listdir('./video'):
#         files.append(input_folder_video+'/'+f)
#
#     c1 = '"{OPENFACE_LOCATION}" {videos} -out_dir '+ output_folder +' -2Dfp -3Dfp -pose -aus -gaze'
#     videoPaths = ""
#     for f in range(0,len(files)):
#         videoPaths+='-f "'+files[f]+'" '
#
#     com1 = c1.format(OPENFACE_LOCATION = OPENFACE_LOCATION , videos= videoPaths)
#     subprocess.call(com1, shell=True)
#
#
# #------------------------------------ STAGE 2 -------------------------------------#
# def processCSV_file():
#     input_folder = './Features'
#     input_folder_video = './video'
#     for f in os.listdir(input_folder):
#         filePath = os.path.join(input_folder, f)
#         if (f[-4:] == '.csv'):
#             filePathOrig = os.path.join(input_folder_video, f)
#             filePathOrig = filePathOrig.replace('.csv', '.mp4')
#             # do_further_processing(filePath,filePathOrig)
#             print filePathOrig,filePath
# processCSV_file()
#
# def do_further_processing(filePath,filePathOrig):
#     try:
#         print(filePath)
#         with open(filePath) as csvfile:
#             reader = pandas.read_csv(csvfile)
#             landmarkX = ' eye_lmk_x_'
#             landmarkY = ' eye_lmk_y_'
#             frame = []
#             for index, row in reader.iterrows():
#                 listx = []
#                 listy = []
#                 gazelist = []
#                 for i in range(56):
#                     landmarkx = landmarkX + str(i)
#                     landmarky = landmarkY + str(i)
#                     valuex = row[landmarkx].tolist()
#                     listx.append(valuex)
#                     valuey = row[landmarky].tolist()
#                     listy.append(valuey)
#                 listx = listx + listy
#                 gaze_0_x = row[' gaze_0_x'].tolist()
#                 gaze_0_y = row[' gaze_0_y'].tolist()
#                 gaze_0_z = row[' gaze_0_z'].tolist()
#                 gaze_1_x = row[' gaze_1_x'].tolist()
#                 gaze_1_y = row[' gaze_1_y'].tolist()
#                 gaze_1_z = row[' gaze_1_z'].tolist()
#
#                 gazelist.append(gaze_0_x)
#                 gazelist.append(gaze_0_y)
#                 gazelist.append(gaze_0_z)
#                 gazelist.append(gaze_1_x)
#                 gazelist.append(gaze_1_y)
#                 gazelist.append(gaze_1_z)
#
#                 gazelist = listx + gazelist
#                 frame.append(gazelist)
#             find_K_cluster(frame,filePathOrig)
#             #print(frame)
#     except:
#         print('File not Found ',filePath)
#
# def find_K_cluster(frame,filePathOrig):
#     try:
#         print('Processing file ',filePathOrig)
#         frame = preprocessing.scale(frame)
#
#         kmeans = KMeans(n_clusters=10, random_state=0).fit(frame)
#         frame_number = []
#         for cluster in list(kmeans.cluster_centers_):
#             values = []
#             for framevector in frame:
#                 values.append(scipy.spatial.distance.euclidean(cluster, framevector))
#             frame_number.append(numpy.argmin(values))
#         vidcap = cv2.VideoCapture(filePathOrig)
#         success,image = vidcap.read()
#         count = 0
#         framenumberindex=1
#         direc = './Frames/{}/'.format(filePathOrig[8:-4])
#         if not os.path.exists(direc):
#             os.mkdir(direc)
#         while success:
#             if (framenumberindex in frame_number):
#                 name = './Frames/{}/frame{}.jpg'.format(filePathOrig[8:-4],count)
#                 cv2.imwrite(name, image)
#             success, image = vidcap.read()
#             count += 1
#             framenumberindex = framenumberindex + 1
#     except:
#         print("happed something wrong")

#----------------------------------- STAGE 3 -------------------------------------#

# OPENFACE_LOCATION = '/home/black/OpenFace-master/build/bin/FeatureExtraction'
#
# input_folder_frame = './Frames'
# output_folder = './FrameFeature'
#
# for frames_folder in os.listdir(input_folder_frame):
#     print (frames_folder)
#     flag =0
#     output_folder_frame = os.path.join(output_folder, frames_folder)
#     for frames_ in os.listdir(os.path.join(input_folder_frame,frames_folder)):
#         framespath = os.path.join(input_folder_frame,os.path.join(frames_folder,frames_))
#         if not os.path.exists(output_folder_frame):
#             os.mkdir(output_folder_frame)
#         c1 = '"{OPENFACE_LOCATION}" -f "{videos}" -out_dir "' + output_folder_frame + '" -2Dfp -3Dfp -pose -aus -gaze'
#         com1 = c1.format(OPENFACE_LOCATION = OPENFACE_LOCATION , videos= framespath)
#         subprocess.call(com1, shell=True, executable="/bin/bash")

#----------------------------------- STAGE 4 -------------------------------------#

def EyeCheckUp(eye0,eye1):#BGR
    cimg = cv2.cvtColor(eye0,cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(cimg,cv2.HOUGH_GRADIENT,1,500, param1=50,param2=30,minRadius=0,maxRadius=0)

    circles = numpy.uint16(numpy.around(circles))
    for i in circles[0,:]:
        # draw the outer circle
        cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
        # draw the center of the circle
        # cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)

    cv2.imshow('detected circles',cimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
#     h, w ,ch = eye0.shape
#     originalEye0 = copy.copy(eye0)
#     whitepixel_0=0
#     #
#     image = eye0.reshape((eye0.shape[0] * eye0.shape[1], 3))
#
#     clt = KMeans(n_clusters=3)
#     labels = clt.fit_predict(image)
#     quant = clt.cluster_centers_.astype("uint8")[labels]
#
#     # reshape the feature vectors to images
#     quant = quant.reshape((h, w, 3))
#     image = image.reshape((h, w, 3))
#
#     # convert from L*a*b* to RGB
#     quant = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)
#     image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)
#
#     cv2.imshow("Using Kmean", numpy.hstack([eye0, image, quant]))
#
# #<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>><>
#
#     for i in range(0,h):
#         for j in range(0,w):
#             if eye0[i][j][0] > 20 and eye0[i][j][1] > 40 and eye0[i][j][2] > 95 and (max(eye0[i][j][0],eye0[i][j][1],eye0[i][j][2])-min(eye0[i][j][0],eye0[i][j][1],eye0[i][j][2]))>15 and abs(eye0[i][j][2]-eye0[i][j][1])>15 and (eye0[i][j][2]>eye0[i][j][1]) and (eye0[i][j][2]>eye0[i][j][0]):
#                 whitepixel_0 = whitepixel_0 + 1
#                 cv2.circle(eye0, (j, i), 2, (255, 0, 0), -1)
#
#     othercondition = copy.copy(originalEye0)
#     for i in range(0,h):
#         for j in range(0,w):
#             if eye0[i][j][0] > 170 and eye0[i][j][1] > 210 and eye0[i][j][2] > 220 and (max(eye0[i][j][0],eye0[i][j][1],eye0[i][j][2])-min(eye0[i][j][0],eye0[i][j][1],eye0[i][j][2]))>15 and abs(eye0[i][j][2]-eye0[i][j][1])<15 and (eye0[i][j][2]>eye0[i][j][0]) and (eye0[i][j][0]>eye0[i][j][1]):
#                 whitepixel_0 = whitepixel_0 + 1
#                 cv2.circle(othercondition, (j, i), 2, (255, 0, 0), -1)
#
#     cv2.imshow('Using thresholds',numpy.hstack([originalEye0,eye0,othercondition]))
#
# # <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>><>
#
#     # eye_cascade = cv2.CascadeClassifier(
#     #     '/home/black/anaconda3/pkgs/opencv3-3.1.0-py36_0/share/OpenCV/haarcascades/haarcascade_eye.xml')
#     #
#     # gray = cv2.cvtColor(eye0, cv2.COLOR_BGR2GRAY)
#     # eyes = eye_cascade.detectMultiScale(gray)
#     # for (ex, ey, ew, eh) in eyes:
#     #     cv2.rectangle(eye0, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
#
# # <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>><>
#
#
#     img_yuv0 = cv2.cvtColor(originalEye0, cv2.COLOR_BGR2YUV)
#     img_yuv1 = cv2.cvtColor(originalEye0, cv2.COLOR_BGR2YUV)
#     img_yuv2 = cv2.cvtColor(originalEye0, cv2.COLOR_BGR2YUV)
#
#     # equalize the histogram of the Y channel
#     img_yuv0[:, :, 0] = cv2.equalizeHist(img_yuv0[:, :, 0])
#     img_yuv1[:, :, 1] = cv2.equalizeHist(img_yuv1[:, :, 1])
#     img_yuv2[:, :, 2] = cv2.equalizeHist(img_yuv2[:, :, 2])
#
#     # convert the YUV image back to RGB format
#     img_output0 = cv2.cvtColor(img_yuv0, cv2.COLOR_YUV2BGR)
#     img_output1 = cv2.cvtColor(img_yuv1, cv2.COLOR_YUV2BGR)
#     img_output2 = cv2.cvtColor(img_yuv2, cv2.COLOR_YUV2BGR)
#
#     method3 = copy.copy(originalEye0)
#     for i in range(0,h):
#         for j in range(0,w):
#             if img_output0[i][j][0] > 190 and img_output1[i][j][1] > 100 and img_output2[i][j][2] < 150:
#                 whitepixel_0 = whitepixel_0 + 1
#                 # if abs(img_output[i][j][2] - img_output[i][j][0]) < 40:
#                 cv2.circle(method3, (j, i), 2, (255, 0, 0), 1)
#
#     cv2.imshow('YUV ',numpy.hstack([originalEye0,img_output0,img_output1,img_output2,method3]))
#
# # <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>><>
#
#
#     eye1 = cv2.GaussianBlur(originalEye0, (3, 3), 0)
#     eye3 = cv2.cvtColor(eye1, cv2.COLOR_BGR2GRAY)
#     # ret, th1 = cv2.threshold(eye3,0,255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#     ret,th1 = cv2.threshold(eye3,100,255,cv2.THRESH_BINARY)
#
#     # cv2.imshow('Binary',scipy.sparse.hstack([originalEye0,th1]))
#     cv2.imshow('Binary',th1)
#     # #
#     # eye2 = cv2.GaussianBlur(eye1, (3, 3), 0)
#     # eye4 = cv2.cvtColor(eye2, cv2.COLOR_BGR2GRAY)
#     # ret,th2 = cv2.threshold(eye4, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#     #
#     # kernel = numpy.ones((3, 3), numpy.uint8)
#     # th1 = cv2.erode(th1, kernel, iterations=2)
#     # th2 = cv2.erode(th2, kernel, iterations=2)
#     #
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#
def eye_Fataigue():
    counter = 0
    input_folder_frame = './Frames'
    for file in os.listdir(input_folder_frame):
        for frame in os.listdir(os.path.join(input_folder_frame, file)):
            counter += 1
            framespath = os.path.join(input_folder_frame, os.path.join(file, frame))
            framespath_util = copy.copy(framespath)
            FrameCSVPath = framespath_util.replace('.jpg', '.csv')
            FrameCSVPath = FrameCSVPath.replace('./Frames', './FrameFeature')
            # try:
            with open(FrameCSVPath) as csvfile:
                reader = pandas.read_csv(csvfile)
                eye_pupil_X_0 = []
                eye_pupil_y_0 = []
                eye_pupil_X_1 = []
                eye_pupil_y_1 = []

                left_corner_0_x = 0
                left_corner_0_y = 0

                right_corner_0_x = 0
                right_corner_0_y = 0

                left_corner_1_x = 0
                left_corner_1_y = 0

                right_corner_1_x = 0
                right_corner_1_y = 0

                for index, row in reader.iterrows():
                    landmark_X_0 = ' eye_lmk_x_'
                    landmark_Y_0 = ' eye_lmk_y_'

                    left_corner_0_x = row[' eye_lmk_x_8']
                    left_corner_0_y = row[' eye_lmk_y_8']

                    right_corner_0_x = row[' eye_lmk_x_14']
                    right_corner_0_y = row[' eye_lmk_y_14']

                    left_corner_1_x = row[' eye_lmk_x_36']
                    left_corner_1_y = row[' eye_lmk_y_36']

                    right_corner_1_x = row[' eye_lmk_x_42']
                    right_corner_1_y = row[' eye_lmk_y_42']

                    for i in range(8):
                        landmarkx = landmark_X_0 + str(i)
                        landmarky = landmark_Y_0 + str(i)
                        valuex = row[landmarkx].tolist()
                        eye_pupil_X_0.append(valuex)
                        valuey = row[landmarky].tolist()
                        eye_pupil_y_0.append(valuey)
                    landmark_X_1 = ' eye_lmk_x_'
                    landmark_Y_1 = ' eye_lmk_y_'
                    for i in range(28, 36):
                        landmarkx = landmark_X_1 + str(i)
                        landmarky = landmark_Y_1 + str(i)
                        valuex = row[landmarkx].tolist()
                        eye_pupil_X_1.append(valuex)
                        valuey = row[landmarky].tolist()
                        eye_pupil_y_1.append(valuey)

                image = cv2.imread(framespath)

                # for (x, y) in zip(eye_pupil_X_0, eye_pupil_y_0):
                #     cv2.circle(image, (int(x), int(y)), 1, (0, 0, 255), -1)
                #
                # for (x, y) in zip(eye_pupil_X_1, eye_pupil_y_1):
                #     cv2.circle(image, (int(x), int(y)), 1, (0, 0, 255), -1)

                # cv2.rectangle(image, (int(eye_pupil_X_0[2]), int(left_corner_0_y)), (int(eye_pupil_X_0[6]), int(right_corner_0_y)), (0, 255, 0), 3)
                # print(left_corner_0_x , left_corner_0_y,right_corner_0_x,right_corner_0_y)

                # print(eye_pupil_X_0[2], eye_pupil_y_0[2])
                # cv2.circle(image, (int(eye_pupil_X_0[2]), int(eye_pupil_y_0[2])), 4, (0, 0, 255), 1)
                #
                # print(eye_pupil_X_0[6], eye_pupil_y_0[6])
                # cv2.circle(image, (int(eye_pupil_X_0[6]), int(eye_pupil_y_0[6])), 4, (0, 0, 255), 1)

                # cv2.rectangle(image, (int(left_corner_0_x),int(eye_pupil_y_0[2])), (int(right_corner_0_x), int(eye_pupil_y_0[6])), (0, 255, 0), 1)
                # cv2.rectangle(image, (int(left_corner_1_x),int(eye_pupil_y_1[2])), (int(right_corner_1_x), int(eye_pupil_y_1[6])), (0, 255, 0), 1)

                h1 = abs(int(eye_pupil_y_0[2]) - int(eye_pupil_y_0[6]))
                w1 = abs(int(right_corner_0_x) - int(left_corner_0_x))
                eye1 = image[int(eye_pupil_y_0[2]):int(eye_pupil_y_0[2]) + h1,
                       int(left_corner_0_x):int(left_corner_0_x) + w1]

                h2 = abs(int(eye_pupil_y_1[2]) - int(eye_pupil_y_1[6]))
                w2 = abs(int(right_corner_1_x) - int(left_corner_1_x))
                eye2 = image[int(eye_pupil_y_1[2]):int(eye_pupil_y_1[2]) + h2,
                       int(left_corner_1_x):int(left_corner_1_x) + w2]

                EyeCheckUp(eye1, eye2)

eye_Fataigue()

#----------------------------------- STAGE 6--------------------------------------#

# def crop_Face():
#     OPENFACE_LOCATION = '/home/black/OpenFace-master/build/bin/FeatureExtraction'
#     input_folder_frame = './Frames'
#     outputFolder = './CroppedFace/'
#     garbagefolder = './OpenFace_output'
#     counter = 0
#     for file in os.listdir(input_folder_frame):
#         if not os.path.exists(outputFolder+file):
#             os.mkdir(outputFolder+file)
#         for frame in os.listdir(os.path.join(input_folder_frame, file)):
#             framespath = os.path.join(input_folder_frame, os.path.join(file, frame))
#             # print framespath,file ,frame
#             outputFolder1 = os.path.abspath(garbagefolder)
#             c1 = '"{OPENFACE_LOCATION}" -f "{images}"  -out_dir "'+ outputFolder1 +'" -simalign -simsize 112'
#             com1 = c1.format(OPENFACE_LOCATION = OPENFACE_LOCATION , images= framespath)
#             subprocess.call(com1, shell=True)
#             name = frame.replace('jpg','bmp')
#             cmd2 = 'find ./OpenFace_output -name \'*.bmp\' -exec mv {} "'+ outputFolder+file+"/"+name+'" \;'
#             # print cmd2
#             subprocess.call(cmd2, shell=True)
#             subprocess.call('rm -r OpenFace_output/*', shell=True)
#             counter +=1
#     print counter
# crop_Face()

#---------------------------------------------------------------------------------------

# def getGazeFeatures(filePath):
#     features = []
#     for csv in os.listdir(filePath):
#         if csv[-3:] == 'csv':
#             with open(os.path.join(filePath,csv)) as csvfile:
#                 reader = pandas.read_csv(csvfile)
#                 for index, row in reader.iterrows():
#                     features.append(row[' gaze_angle_x'])
#                     features.append(row[' gaze_angle_y'])
#     return features
#
# def getEyeFatigueValues(filePath):
#     feature = []
#     for csv in os.listdir(filePath):
#         if csv[-3:] == 'csv':
#             csvpath = os.path.join(filePath, csv)
#             framespath = csvpath.replace('FrameFeature','Frames')
#             framespath = framespath.replace('csv', 'jpg')
#             with open(csvpath) as csvfile:
#                 reader = pandas.read_csv(csvfile)
#                 eye_pupil_X_0 = []
#                 eye_pupil_y_0 = []
#                 eye_pupil_X_1 = []
#                 eye_pupil_y_1 = []
#
#                 left_corner_0_x = 0
#                 right_corner_0_x = 0
#                 left_corner_1_x = 0
#                 right_corner_1_x = 0
#
#                 for index, row in reader.iterrows():
#                     landmark_X_0 = ' eye_lmk_x_'
#                     landmark_Y_0 = ' eye_lmk_y_'
#
#                     left_corner_0_x = row[' eye_lmk_x_8']
#                     right_corner_0_x = row[' eye_lmk_x_14']
#                     left_corner_1_x = row[' eye_lmk_x_36']
#                     right_corner_1_x = row[' eye_lmk_x_42']
#
#                     for i in range(8):
#                         landmarkx = landmark_X_0 + str(i)
#                         landmarky = landmark_Y_0 + str(i)
#                         valuex = row[landmarkx].tolist()
#                         eye_pupil_X_0.append(valuex)
#                         valuey = row[landmarky].tolist()
#                         eye_pupil_y_0.append(valuey)
#                     landmark_X_1 = ' eye_lmk_x_'
#                     landmark_Y_1 = ' eye_lmk_y_'
#                     for i in range(28, 36):
#                         landmarkx = landmark_X_1 + str(i)
#                         landmarky = landmark_Y_1 + str(i)
#                         valuex = row[landmarkx].tolist()
#                         eye_pupil_X_1.append(valuex)
#                         valuey = row[landmarky].tolist()
#                         eye_pupil_y_1.append(valuey)
#
#                 image = cv2.imread(framespath)
#                 h1 = abs(int(eye_pupil_y_0[2]) - int(eye_pupil_y_0[6]))
#                 w1 = abs(int(right_corner_0_x) - int(left_corner_0_x))
#                 eye1 = image[int(eye_pupil_y_0[2]):int(eye_pupil_y_0[2]) + h1, int(left_corner_0_x):int(left_corner_0_x) + w1]
#
#                 h2 = abs(int(eye_pupil_y_1[2]) - int(eye_pupil_y_1[6]))
#                 w2 = abs(int(right_corner_1_x) - int(left_corner_1_x))
#                 eye2 = image[int(eye_pupil_y_1[2]):int(eye_pupil_y_1[2]) + h2, int(left_corner_1_x):int(left_corner_1_x) + w2]
#                 feature.append(EyeCheckUp(eye1, eye2))
#     return feature
#
# def getPhogFeatures(filePath):
#     feature = []
#     phog = PHogFeatures()
#     for csv in os.listdir(filePath):
#         if csv[-3:] == 'csv':
#             csvpath = os.path.join(filePath, csv)
#             framespath = csvpath.replace('FrameFeature','CroppedFace')
#             framespath = framespath.replace('csv', 'bmp')
#             feature.append(phog.get_features(framespath))
#     return feature
#
# def ExtractFeatures():
#     input_folder = './FrameFeature'
#     for f in os.listdir(input_folder):
#         filePath = os.path.join(input_folder, f)
#         print filePath
        # videoFeatures = getGazeFeatures(filePath)
        # eyeFatiguresValues = getEyeFatigueValues(filePath)
        # phogFeatures = getPhogFeatures(filePath)
        # print len(phogFeatures[0])
# ExtractFeatures()


#----------------------------------- STAGE 5 -------------------------------------#

# class eye:
# 	def __init__(self, Ex,Ey,Ew,Eh,image_gray,image_color):
# 		self.Ex = Ex
# 		self.Ey = Ey
# 		self.Ew = Ew
# 		self.Eh = Eh
# 		self.image_gray = image_gray
# 		self.image_color = image_color
#
# list_of_eyes = []
# face_cascade = cv2.CascadeClassifier('/home/black/anaconda3/pkgs/opencv3-3.1.0-py36_0/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
# eye_cascade = cv2.CascadeClassifier('/home/black/anaconda3/pkgs/opencv3-3.1.0-py36_0/share/OpenCV/haarcascades/haarcascade_eye.xml')
# def fatigue_Determine(img):
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray, 1.3, 5)
#     for (x,y,w,h) in faces:
#         roi_gray = gray[y:y+h, x:x+w]
#         roi_color = img[y:y+h, x:x+w]
#         eyes = eye_cascade.detectMultiScale(roi_gray)
#         for (ex,ey,ew,eh) in eyes:
#             cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
#             list_of_eyes.append(eye(ex,ey,ew,eh,roi_gray,roi_color))
#
#     cv2.imshow('img',img)
#
#     print('Number of eyes ',len(list_of_eyes))
#
#     for eyes in list_of_eyes:
#         cropped = eyes.image_gray[eyes.Ey+10:eyes.Ey+eyes.Eh,eyes.Ex:eyes.Ex+eyes.Ew]
#         cropped2 = eyes.image_color[eyes.Ey+10:eyes.Ey+eyes.Eh,eyes.Ex:eyes.Ex+eyes.Ew]
#         th3=cropped
#         cropped2 = cv2.GaussianBlur(cropped, (3, 3), 0)
#         ret,th3 = cv2.threshold(cropped2,80,255,cv2.THRESH_BINARY)
#         cv2.imshow('Image',th3)
#         cv2.waitKey(0)
#     cv2.destroyAllWindows()
#
# img = cv2.imread('selfie3.jpg')
# fatigue_Determine(img)

# height, width, channels = img.shape
# cv2.countNonZero(img)

# import cv2
# import numpy as np
# #
# img = cv2.imread('eye.jpeg',0)
# img = cv2.medianBlur(img,5)
# cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
#
# circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,500, param1=50,param2=30,minRadius=0,maxRadius=0)
#
# circles = np.uint16(np.around(circles))
# for i in circles[0,:]:
#     # draw the outer circle
#     cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
#     # draw the center of the circle
#     cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
#
# cv2.imshow('detected circles',cimg)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#
#


# import cv2
# vidcap = cv2.VideoCapture('MY DRUNK GIRLFRIEND - nowthisisliving - YouTube_Drunk.mp4_61.0.avi_aligned.avi.mp4')
# success,image = vidcap.read()
# count = 0
# while success:
#   cv2.imwrite("frames/frame%d.jpg" % count, image)     # save frame as JPEG file
#   success,image = vidcap.read()
#   print('Read a new frame: ', success)
#   count += 1

# import numpy as np
# import cv2
# import time
#
# cap = cv2.VideoCapture('default.wmv')  # initialize video capture
# left_counter = 0  # counter for left movement
# right_counter = 0  # counter for right movement
#
# th_value = 5  # changeable threshold value
#
#
# def thresholding(value):  # function to threshold and give either left or right
#     global left_counter
#     global right_counter
#
#     if (value <= 54):  # check the parameter is less than equal or greater than range to
#         left_counter = left_counter + 1  # increment left counter
#
#         if (left_counter > th_value):  # if left counter is greater than threshold value
#             print('RIGHT')  # the eye is left
#             left_counter = 0  # reset the counter
#
#     elif (value >= 54):  # same procedure for right eye
#         right_counter = right_counter + 1
#
#         if (right_counter > th_value):
#             print('LEFT')
#             right_counter = 0
#
#
# while 1:
#     ret, frame = cap.read()
#     cv2.line(frame, (320, 0), (320, 480), (0, 200, 0), 2)
#     cv2.line(frame, (0, 200), (640, 200), (0, 200, 0), 2)
#     if ret == True:
#         col = frame
#
#         frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
#         pupilFrame = frame
#         clahe = frame
#         blur = frame
#         edges = frame
#         eyes = cv2.CascadeClassifier('haarcascade_eye.xml')
#         detected = eyes.detectMultiScale(frame, 1.3, 5)
#         for (x, y, w, h) in detected:  # similar to face detection
#             cv2.rectangle(frame, (x, y), ((x + w), (y + h)), (0, 0, 255), 1)  # draw rectangle around eyes
#             cv2.line(frame, (x, y), ((x + w, y + h)), (0, 0, 255), 1)  # draw cross
#             cv2.line(frame, (x + w, y), ((x, y + h)), (0, 0, 255), 1)
#             pupilFrame = cv2.equalizeHist(
#                 frame[y + (h * .25):(y + h), x:(x + w)])  # using histogram equalization of better image.
#             cl1 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # set grid size
#             clahe = cl1.apply(pupilFrame)  # clahe
#             blur = cv2.medianBlur(clahe, 7)  # median blur
#             circles = cv2.HoughCircles(blur, cv2.cv.CV_HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=7,
#                                        maxRadius=21)  # houghcircles
#             if circles is not None:  # if atleast 1 is detected
#                 circles = np.round(circles[0, :]).astype("int")  # change float to integer
#                 print
#                 'integer', circles
#                 for (x, y, r) in circles:
#                     cv2.circle(pupilFrame, (x, y), r, (0, 255, 255), 2)
#                     cv2.rectangle(pupilFrame, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
#                     # set thresholds
#                     thresholding(x)
#
#         # frame = cv2.medianBlur(frame,5)
#         cv2.imshow('image', pupilFrame)
#         cv2.imshow('clahe', clahe)
#         cv2.imshow('blur', blur)
#
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#
# cap.release()
# cv2.destroyAllWindows()