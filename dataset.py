"""
This is not a classification task where we need to pass labels txt file. 
In this we actually need to process all our images in POC so we can find their location which we can display again.
"""

import tensorflow
import pickle
from PIL import Image
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

BATCH_SIZE = 64
root_path = r"C:\Users\sudee\OneDrive\Desktop\CapstoneProject\fashion-dataset\images"

datagen = tensorflow.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=preprocess_input)
generator = datagen.flow_from_directory(root_path, 
                                        target_size=(224, 224),
                                        batch_size=BATCH_SIZE,
                                        class_mode=None,
                                        shuffle=False)

filenames = generator.filenames

filenames = [root_path + '/' + s for s in generator.filenames]

pickle.dump(filenames, open('Models/filenames-fashion.pickle', 'wb'))