import numpy
from sklearn.cluster import KMeans
import cv2
import pandas
import os,scipy
from sklearn import preprocessing

'''
This stage extract the video frames based on clustering
'''

def processCSV_file():
    input_folder = './Features'
    input_folder_video = './video'
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
