import glob
import os
from sklearn.model_selection import train_test_split as trainTestSplit

def split(devSize):

    ROOT = 'stanford_dogs_dataset'
    f = os.listdir(ROOT)

    X = []
    Y = []

    for folder in f:
        direct = os.path.join(ROOT, folder)
        labelDirect = os.path.join(direct,folder + ".txt")

        with open(labelDirect) as labelFile:
            labelList = labelFile.read().splitlines()

        fileList = glob.glob(os.path.join(direct, '*.jpg'))
        for i in range(len(fileList)):
            X.append(fileList[i])
            Y.append(labelList[i])

    XTrain, XDev, YTrain, YDev = trainTestSplit(X, Y, test_size=devSize)

    return XTrain, XDev, YTrain, YDev

def kSplit(kSet, size):

    DB_ROOT = 'db'
    DB_ROOT = os.path.abspath(DB_ROOT)

    for k in range(kSet):

        dbDirect = os.path.join(DB_ROOT, "db" + str(k) + "/")
        if not os.path.exists(dbDirect):
            os.makedirs(dbDirect)
        else:
            tempFile = os.listdir(dbDirect)
            for trash in tempFile:
                os.remove(dbDirect + trash)

        trainFile = open(dbDirect + "train.txt", "w+")
        devFile = open(dbDirect + "dev.txt", "w+")
        lbTrainFile = open(dbDirect + "lbTrain.txt", "w+")
        lbDevFile = open(dbDirect + "lbDev.txt", "w+")

        XTrain, XDev, YTrain, YDev = split(devSize= size)

        for i in range(len(XTrain)):
             trainFile.write(XTrain[i] + "\n")
             lbTrainFile.write(YTrain[i] + "\n")

        for i in range(len(XDev)):
             devFile.write(XDev[i] + "\n")
             lbDevFile.write(YDev[i] + "\n")

        trainFile.close()
        devFile.close()
        lbTrainFile.close()
        lbDevFile.close()

kSplit(5, 0.3)
