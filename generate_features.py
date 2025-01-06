import pickle
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from numpy.linalg import norm
import numpy as np
import glob

# Load the finetuned ResNet50 model
finetuned_model = load_model('Models/resnet_model-finetuned.h5')

# Function to extract features
def extract_features(image_path, model):
    img = image.load_img(image_path, target_size=(224, 224))
    image_array = image.img_to_array(img)
    expanded_image_array = np.expand_dims(image_array, axis=0)
    processed_image = preprocess_input(expanded_image_array)
    features = model.predict(processed_image)
    flattened_features = features.flatten()
    normalized_features = flattened_features / norm(flattened_features)
    return normalized_features

# List to hold features
feature_list = []

# Assuming you have a folder with images to extract features from
image_paths = glob.glob(r"C:\Users\sudee\OneDrive\Desktop\CapstoneProject\fashion-dataset\images\*.jpg")

# Extract features from each image and store them in feature_list
for img_path in image_paths:
    img_features = extract_features(img_path, finetuned_model)
    feature_list.append(img_features)

# Save the features to a pickle file
pickle.dump(feature_list, open('Models/features-fashion-resnet50.pickle', 'wb'))

# Optionally, save the image paths to another file if needed for indexing
pickle.dump(image_paths, open('Models/filenames-fashion.pickle', 'wb'))