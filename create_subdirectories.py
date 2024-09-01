import os
import shutil
import random
import pandas as pd

# Define the source directory containing the classes of the Animals-10 dataset
source_directory = "Project/dataset/"

# Define the destination directory where the split data will be stored
destination_directory = "Project/split_dataset/"

# Define the ratio for splitting (80% training, 20% validation)
split_ratio = 0.8

# Create the destination directory if it doesn't exist
if not os.path.exists(destination_directory):
    os.makedirs(destination_directory)

# Function to count the number of files in a directory
def count_files(directory):
    return sum(len(files) for _, _, files in os.walk(directory))

# Dictionary to store the counts
counts = {'Class': [], 'Train': [], 'Validation': []}

# Iterate through each class directory in the source directory
for class_name in os.listdir(source_directory):
    class_directory = os.path.join(source_directory, class_name)
    
    # Create subdirectories for training and validation
    train_directory = os.path.join(destination_directory, "train", class_name)
    val_directory = os.path.join(destination_directory, "val", class_name)
    os.makedirs(train_directory, exist_ok=True)
    os.makedirs(val_directory, exist_ok=True)
    
    # Get a list of image files in the class directory
    image_files = [f for f in os.listdir(class_directory) if os.path.isfile(os.path.join(class_directory, f))]
    
    # Shuffle the list of image files randomly
    random.shuffle(image_files)
    
    # Calculate the number of images for training and validation based on the split ratio
    num_train_images = int(len(image_files) * split_ratio)
    num_val_images = len(image_files) - num_train_images
    
    # Split the image files into training and validation sets
    train_images = image_files[:num_train_images]
    val_images = image_files[num_train_images:]
    
    # Move training images to the training directory
    for image in train_images:
        source_path = os.path.join(class_directory, image)
        destination_path = os.path.join(train_directory, image)
        shutil.copy(source_path, destination_path)
    
    # Move validation images to the validation directory
    for image in val_images:
        source_path = os.path.join(class_directory, image)
        destination_path = os.path.join(val_directory, image)
        shutil.copy(source_path, destination_path)
    
    # Update the counts dictionary
    counts['Class'].append(class_name)
    counts['Train'].append(len(train_images))
    counts['Validation'].append(len(val_images))

# Create a DataFrame from the counts dictionary
df_counts = pd.DataFrame(counts)

# Display the DataFrame as a table
print(df_counts)
