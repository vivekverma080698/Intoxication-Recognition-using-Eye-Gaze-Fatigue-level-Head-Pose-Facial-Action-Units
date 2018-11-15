from phogDescriptor import PHogFeatures
import cv2
import pandas
import os

'''
This stage extract all the features ex eyeGaze , eye fatigue , phog features
'''

def EyeCheckUp(eye0,eye1):#BGR
    return 50
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