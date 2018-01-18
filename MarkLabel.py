import os
import glob


def mark_label():

    ROOT = 'stanford_dogs_dataset\\'
    f = os.listdir('stanford_dogs_dataset')
    label_name = 0
    for folder in f:
        direct = ROOT + folder + '\\'
        if not os.path.exists(direct):
            os.makedirs(direct)
        label_file = open(direct + str(folder) + ".txt", "w+")
        for _ in glob.glob(direct + '*.jpg'):
            label_file.write(str(label_name) + "\n")
        label_name = label_name + 1
        label_file.close()
mark_label()