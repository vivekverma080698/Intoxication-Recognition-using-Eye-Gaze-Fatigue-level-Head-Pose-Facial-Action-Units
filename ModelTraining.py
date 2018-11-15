import csv
import numpy as np
import random
import math
from sklearn import svm

DrunkData = []
if 1:
    FeatureFileDrunk = 'Drunk.csv'
    with open(FeatureFileDrunk) as raw:
        csv_reader = csv.reader(raw, delimiter=',')
        for row in csv_reader:
            DrunkData.append(row)

SoberData = []
if 1:
    FeatureFileSober = 'Sober.csv'
    data = []
    with open(FeatureFileSober) as raw:
        csv_reader = csv.reader(raw, delimiter=',')
        for row in csv_reader:
            SoberData.append(row)

model_1_Drunk_data = []
model_1_Sober_data = []
model_2_Drunk_data = []
model_2_Sober_data = []

for sep in DrunkData:
    model_1_Drunk_data.append(sep[0:500])
    model_2_Drunk_data.append(sep[500:])

for sep in SoberData:
    model_1_Sober_data.append(sep[0:500])
    model_2_Sober_data.append(sep[500:])


model1 = svm.SVC(C=1, kernel='rbf', gamma=0.01, cache_size=800)
model2 = svm.SVC(C=1, kernel='rbf', gamma=0.01, cache_size=800)


if 1:
    X = np.concatenate((model_1_Drunk_data, model_1_Sober_data))
    ones = np.ones(len(model_1_Drunk_data))
    zeros = np.zeros(len(model_1_Sober_data))
    Y = np.concatenate((zeros, ones))
    c = list(zip(X, Y))
    random.shuffle(c)
    X, y = zip(*c)

    fraction = 0.75

    num_of_Train_exp = math.floor(len(X) * fraction)
    X_train = X[0:num_of_Train_exp]
    X_test = X[num_of_Train_exp:-1]
    Y_train = y[0:num_of_Train_exp]
    Y_test = y[num_of_Train_exp:-1]

    model1.fit(X_train, Y_train)


if 1:
    X = np.concatenate((model_2_Drunk_data, model_2_Sober_data))
    ones = np.ones(len(model_2_Drunk_data))
    zeros = np.zeros(len(model_2_Sober_data))
    Y = np.concatenate((zeros, ones))
    c = list(zip(X, Y))
    random.shuffle(c)
    X, y = zip(*c)

    fraction = 0.75

    num_of_Train_exp = math.floor(len(X) * fraction)
    X_train = X[0:num_of_Train_exp]
    X_test = X[num_of_Train_exp:-1]
    Y_train = y[0:num_of_Train_exp]
    Y_test = y[num_of_Train_exp:-1]

    model2.fit(X_train, Y_train)


