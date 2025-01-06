import numpy as np
import pickle
from sklearn.neighbors import NearestNeighbors
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from numpy.linalg import norm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Provide the path to the image you want to test
image_to_test = r"C:\Users\sudee\OneDrive\Desktop\CapstoneProject\images\shirt.jpeg"

# Load the necessary files
filenames = pickle.load(open('Models/filenames-fashion.pickle', 'rb'))
feature_list = pickle.load(open('Models/features-fashion-resnet50.pickle', 'rb'))
finetuned_resnet_model = 'Models/resnet_model-finetuned.h5'

# Load the finetuned ResNet50 model
ResNet50_finetuned_model = keras.models.load_model(finetuned_resnet_model)

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

# Extract features from the input image
input_image_features = extract_features(image_to_test, ResNet50_finetuned_model)

# Use NearestNeighbors to find similar images
neighbors = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='euclidean').fit(feature_list)
distances, indices = neighbors.kneighbors([input_image_features])

# Display the input image
print("Input Image:")
plt.imshow(mpimg.imread(image_to_test), interpolation='lanczos')
plt.show()

# Display the retrieved similar images
print("Similar Images:")
for i in range(5):
    plt.imshow(mpimg.imread(filenames[indices[0][i]]), interpolation='lanczos')
    plt.show()
