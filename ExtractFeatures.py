from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input as vgg16_pi
import numpy as np
import os
import glob
from keras.models import Model

def feature_extract_vgg_16():

    ROOT = 'stanford_dogs_dataset\\'
    f = os.listdir('stanford_dogs_dataset')
    base_model = VGG16(weights='imagenet')
    folder_name = 'vgg16_dogs_features\\'

    model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)

    for folder in f:
        direct = ROOT + folder + '\\'
        print(direct)
        for filename in glob.glob(direct + '*.jpg'):
            print(filename)

            img = image.load_img(filename, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = vgg16_pi(x)

            features = model.predict(x)

            names = filename.split('\\')
            count = names[2].split('.')
            newpath = r'' + folder_name + names[1] + '\\'
            if not os.path.exists(newpath):
                os.makedirs(newpath)
            newName = newpath + count[0] + '.npy'
            np.save(newName, features)
            #print(features.shape)
feature_extract_vgg_16()

