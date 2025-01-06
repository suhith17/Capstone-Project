import os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input

# Load the pre-trained ResNet50 model without the top layer
base_model = ResNet50(weights='imagenet', include_top=False)

# Add custom layers for fine-tuning
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)

# Define the number of classes
train_dir = r"C:\Users\sudee\OneDrive\Desktop\CapstoneProject\fashion-dataset\train"
num_classes = len(os.listdir(train_dir))  # Count the number of subdirectories in the training folder

predictions = Dense(num_classes, activation='softmax')(x)

# Create the full model
finetuned_model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
finetuned_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Prepare your data
train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
train_generator = train_datagen.flow_from_directory(train_dir, target_size=(224, 224), batch_size=32, class_mode='categorical')

val_dir = r"C:\Users\sudee\OneDrive\Desktop\CapstoneProject\fashion-dataset\val"
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
val_generator = val_datagen.flow_from_directory(val_dir, target_size=(224, 224), batch_size=32, class_mode='categorical')

# Train the model
finetuned_model.fit(train_generator, epochs=10, validation_data=val_generator)

# Save the finetuned model
finetuned_model.save('Models/resnet_model-finetuned.h5')
