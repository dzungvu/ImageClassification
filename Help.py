import numpy as np
import os

def getFeature(imageName, featureType):

    imgFeatureName = imageName
    imgFeatureName = imgFeatureName.replace("stanford_dogs_dataset",str(featureType) + "_features")
    imgFeatureName = imgFeatureName.replace("jpg","npy")

    imgFeatureData = np.load(imgFeatureName)

    return imgFeatureName, imgFeatureData

def getFeatureFromFolder(folderDirect, dataType, featureType):
    
    DB_ROOT = folderDirect
    fileDirect = os.path.join(folderDirect, str(dataType) +".txt")
    Y = []
    X = []

    with open(fileDirect) as fileName:
        names = fileName.read().splitlines()

    for name in names:
        X.append(getFeature(name, featureType)[1])


    X = np.asarray(X)
    X = X.reshape((X.shape[0], X.shape[2]))

    labelFileDirect = os.path.join(DB_ROOT, "lb" + str(dataType) + ".txt")

    with open(labelFileDirect) as labelFile:
        Y = labelFile.read().splitlines()

    Y = np.array(Y)
    Y = Y.reshape((1, Y.shape[0])).T
    Y = np.ravel(Y)
    return X, Y

