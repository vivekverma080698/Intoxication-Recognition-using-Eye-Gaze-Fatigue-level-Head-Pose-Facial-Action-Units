from phogDescriptor import PHogFeatures
import cv2
import pandas
import os

def getGazeFeatures(filePath):
    features = []
    for csv in os.listdir(filePath):
        if csv[-3:] == 'csv':
            with open(os.path.join(filePath,csv)) as csvfile:
                reader = pandas.read_csv(csvfile)
                for index, row in reader.iterrows():
                    features.append(row[' gaze_angle_x'])
                    features.append(row[' gaze_angle_y'])
    return features

def getEyeFatigueValues(filePath):
    feature = []
    for csv in os.listdir(filePath):
        if csv[-3:] == 'csv':
            csvpath = os.path.join(filePath, csv)
            framespath = csvpath.replace('FrameFeature','Frames')
            framespath = framespath.replace('csv', 'jpg')
            with open(csvpath) as csvfile:
                reader = pandas.read_csv(csvfile)
                eye_pupil_X_0 = []
                eye_pupil_y_0 = []
                eye_pupil_X_1 = []
                eye_pupil_y_1 = []

                left_corner_0_x = 0
                right_corner_0_x = 0
                left_corner_1_x = 0
                right_corner_1_x = 0

                for index, row in reader.iterrows():
                    landmark_X_0 = ' eye_lmk_x_'
                    landmark_Y_0 = ' eye_lmk_y_'

                    left_corner_0_x = row[' eye_lmk_x_8']
                    right_corner_0_x = row[' eye_lmk_x_14']
                    left_corner_1_x = row[' eye_lmk_x_36']
                    right_corner_1_x = row[' eye_lmk_x_42']

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
                h1 = abs(int(eye_pupil_y_0[2]) - int(eye_pupil_y_0[6]))
                w1 = abs(int(right_corner_0_x) - int(left_corner_0_x))
                eye1 = image[int(eye_pupil_y_0[2]):int(eye_pupil_y_0[2]) + h1, int(left_corner_0_x):int(left_corner_0_x) + w1]

                h2 = abs(int(eye_pupil_y_1[2]) - int(eye_pupil_y_1[6]))
                w2 = abs(int(right_corner_1_x) - int(left_corner_1_x))
                eye2 = image[int(eye_pupil_y_1[2]):int(eye_pupil_y_1[2]) + h2, int(left_corner_1_x):int(left_corner_1_x) + w2]
                feature.append(EyeCheckUp(eye1, eye2))
    return feature

def getPhogFeatures(filePath):
    feature = []
    phog = PHogFeatures()
    for csv in os.listdir(filePath):
        if csv[-3:] == 'csv':
            csvpath = os.path.join(filePath, csv)
            framespath = csvpath.replace('FrameFeature','CroppedFace')
            framespath = framespath.replace('csv', 'bmp')
            feature.append(phog.get_features(framespath))
    return feature

def ExtractFeatures():
    input_folder = './FrameFeature'
    for f in os.listdir(input_folder):
        filePath = os.path.join(input_folder, f)
        print filePath
        videoFeatures = getGazeFeatures(filePath)
        eyeFatiguresValues = getEyeFatigueValues(filePath)
        phogFeatures = getPhogFeatures(filePath)