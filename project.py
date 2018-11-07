import csv
import numpy
import pandas
from sklearn.cluster import KMeans
from sklearn import preprocessing
import csv
import shutil
import cv2
import pandas
import os,scipy
import subprocess


#------------------------------------ STAGE 1 --------------------------------------#


OPENFACE_LOCATION = '/home/black/OpenFace-master/build/bin/FeatureExtraction'

input_folder_video = './video'
output_folder = './Features'

# files = []
# for f in os.listdir('./video'):
#  	files.append(input_folder_video+'/'+f)
#
# c1 = '"{OPENFACE_LOCATION}" {videos} -out_dir '+ output_folder +' -2Dfp -3Dfp -pose -aus -gaze'
# videoPaths = ""
# for f in range(0,len(files)):
#     videoPaths+='-f "'+files[f]+'" '
#
# com1 = c1.format(OPENFACE_LOCATION = OPENFACE_LOCATION , videos= videoPaths)
# subprocess.call(com1, shell=True)


#------------------------------------ STAGE 2 -------------------------------------#
output_folder_frames = './Frames'
input_folder_features = './Features'

def processCSV_file(input_folder):
    for f in os.listdir(input_folder):
        filePath = os.path.join(input_folder, f)
        if (f[-4:] == '.csv'):
            filePathOrig = os.path.join(input_folder_video, f)
            filePathOrig = filePathOrig.replace('.csv', '.mp4')
            do_further_processing(filePath,filePathOrig)

def do_further_processing(filePath,filePathOrig):
    try:
        print(filePath)
        with open(filePath) as csvfile:
            reader = pandas.read_csv(csvfile)
            landmarkX = ' eye_lmk_x_'
            landmarkY = ' eye_lmk_y_'
            frame = []
            for index, row in reader.iterrows():
                listx = []
                listy = []
                gazelist = []
                for i in range(56):
                    landmarkx = landmarkX + str(i)
                    landmarky = landmarkY + str(i)
                    valuex = row[landmarkx].tolist()
                    listx.append(valuex)
                    valuey = row[landmarky].tolist()
                    listy.append(valuey)
                listx = listx + listy
                gaze_0_x = row[' gaze_0_x'].tolist()
                gaze_0_y = row[' gaze_0_y'].tolist()
                gaze_0_z = row[' gaze_0_z'].tolist()
                gaze_1_x = row[' gaze_1_x'].tolist()
                gaze_1_y = row[' gaze_1_y'].tolist()
                gaze_1_z = row[' gaze_1_z'].tolist()

                gazelist.append(gaze_0_x)
                gazelist.append(gaze_0_y)
                gazelist.append(gaze_0_z)
                gazelist.append(gaze_1_x)
                gazelist.append(gaze_1_y)
                gazelist.append(gaze_1_z)

                gazelist = listx + gazelist
                frame.append(gazelist)
            find_K_cluster(frame,filePathOrig)
            #print(frame)
    except:
        print('File not Found ',filePath)

def find_K_cluster(frame,filePathOrig):
    try:
        print('Processing file ',filePathOrig)
        frame = preprocessing.scale(frame)

        kmeans = KMeans(n_clusters=10, random_state=0).fit(frame)
        frame_number = []
        for cluster in list(kmeans.cluster_centers_):
            values = []
            for framevector in frame:
                values.append(scipy.spatial.distance.euclidean(cluster, framevector))
            frame_number.append(numpy.argmin(values))
        vidcap = cv2.VideoCapture(filePathOrig)
        success,image = vidcap.read()
        count = 0
        framenumberindex=1
        direc = './Frames/{}/'.format(filePathOrig[8:-4])
        if not os.path.exists(direc):
            os.mkdir(direc)
        while success:
            if (framenumberindex in frame_number):
                name = './Frames/{}/frame{}.jpg'.format(filePathOrig[8:-4],count)
                cv2.imwrite(name, image)
            success, image = vidcap.read()
            count += 1
            framenumberindex = framenumberindex + 1
    except:
        print("happed something wrong")

processCSV_file(input_folder_features)

#----------------------------------- STAGE 3 -------------------------------------#


# class eye:
# 	def __init__(self, Ex,Ey,Ew,Eh,image):	
# 		self.Ex = Ex
# 		self.Ey = Ey
# 		self.Ew = Ew
# 		self.Eh = Eh
# 		self.image = image

# list_of_eyes = []

# face_cascade = cv2.CascadeClassifier('/home/black/anaconda3/pkgs/opencv3-3.1.0-py36_0/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
# eye_cascade = cv2.CascadeClassifier('/home/black/anaconda3/pkgs/opencv3-3.1.0-py36_0/share/OpenCV/haarcascades/haarcascade_eye.xml')
# img = cv2.imread('selfie3.jpg')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# faces = face_cascade.detectMultiScale(gray, 1.3, 5)
# for (x,y,w,h) in faces:
# 	roi_gray = gray[y:y+h, x:x+w]
# 	roi_color = img[y:y+h, x:x+w]
# 	eyes = eye_cascade.detectMultiScale(roi_gray)
# 	for (ex,ey,ew,eh) in eyes:
# 	    cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
# 	    list_of_eyes.append(eye(ex,ey,ew,eh,roi_gray))

# cv2.imshow('img',img)

# print('Number of eyes ',len(list_of_eyes))

# for eyes in list_of_eyes:
# 	cropped = eyes.image[eyes.Ey+10:eyes.Ey+eyes.Eh,eyes.Ex:eyes.Ex+eyes.Ew]
# 	th3=cropped
# 	ret,th3 = cv2.threshold(cropped,80,255,cv2.THRESH_BINARY)
# 	print(th3)
# 	cv2.imshow('Image',th3)
# 	cv2.waitKey(0)

# height, width, channels = img.shape
# cv2.countNonZero(img)

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

