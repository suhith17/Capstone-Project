import os
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split

# Paths
image_dir = r"C:\Users\sudee\OneDrive\Desktop\CapstoneProject\fashion-dataset\images"  # Path to the images directory
csv_path = r"C:\Users\sudee\OneDrive\Desktop\CapstoneProject\fashion-dataset\styles.csv"  # Path to the styles.csv file
output_dir = r"C:\Users\sudee\OneDrive\Desktop\CapstoneProject\fashion-dataset"  # Path where training and validation folders will be created

# Load the CSV file with proper handling of commas in the description column
df = pd.read_csv(csv_path, delimiter=',', quotechar='"', encoding='utf-8')

# Check class distribution
class_counts = df['articleType'].value_counts()
print("Class distribution:")
print(class_counts)

# Filter out classes with fewer than 2 samples
classes_to_keep = class_counts[class_counts > 1].index
df = df[df['articleType'].isin(classes_to_keep)]

# Create training and validation directories
train_dir = os.path.join(output_dir, 'train')
val_dir = os.path.join(output_dir, 'val')
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Split the dataframe into training and validation sets
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['articleType'], random_state=42)

# Function to move images to respective directories
def move_images(df, base_dir):
    for _, row in df.iterrows():
        image_id = row['id']
        article_type = row['articleType']
        src_image_path = os.path.join(image_dir, f"{image_id}.jpg")
        dest_dir = os.path.join(base_dir, article_type)
        os.makedirs(dest_dir, exist_ok=True)
        dest_image_path = os.path.join(dest_dir, f"{image_id}.jpg")

        # Check if the image exists and move it
        if os.path.exists(src_image_path):
            shutil.copy(src_image_path, dest_image_path)
        else:
            print(f"Image {image_id}.jpg not found.")

# Move training images
move_images(train_df, train_dir)

# Move validation images
move_images(val_df, val_dir)

print("Images have been split into training and validation sets.")
