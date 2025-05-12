# Import required libraries
import numpy as np
import math
import os 
from random import shuffle
import constants as CONST 
import cv2

def label_img(name):
    """
    Convert image filename to one-hot encoded label.
    Args:
        name (str): Image filename (e.g., 'cat.123.jpg' or 'dog.456.jpg')
    Returns:
        numpy.ndarray: One-hot encoded label array [1,0] for cat or [0,1] for dog
        None: If the image name doesn't match expected format
    """
    word_label = name.split('.')[0]  # Extract the class name (cat or dog)
    if word_label not in CONST.LABEL_MAP:
        return None
    label = CONST.LABEL_MAP[word_label]  # Get numeric label (0 for cat, 1 for dog)
    label_arr = np.zeros(2)  # Create one-hot encoded array
    label_arr[label] = 1  # Set the appropriate index to 1
    return label_arr


def prep_and_load_data():
    """
    Load and preprocess the training images and their labels.
    Returns:
        tuple: (images, labels) where:
            - images: numpy array of preprocessed images
            - labels: numpy array of one-hot encoded labels
    """
    DIR = CONST.TRAIN_DIR
    images = []
    labels = []
    image_paths = os.listdir(DIR)
    shuffle(image_paths)  # Randomize the order of images
    count = 0
    
    # Process each image in the directory
    for img_path in image_paths:
        try:
            # Get label for the image
            label = label_img(img_path)
            if label is None:
                continue
                
            # Load and preprocess the image
            path = os.path.join(DIR, img_path)
            image = cv2.imread(path)
            if image is None:
                print(f"Could not load image: {img_path}")
                continue
                
            # Resize image to standard size and normalize pixel values
            image = cv2.resize(image, (CONST.IMG_SIZE, CONST.IMG_SIZE))
            image = image.astype('float') / 255.0  # Normalize to [0,1] range
            
            # Add processed image and label to lists
            images.append(image)
            labels.append(label)
            count += 1
            
            # Print progress every 100 images
            if count % 100 == 0:
                print(f"Processed {count} images")
                
            # Stop if we've reached the desired dataset size
            if count == CONST.DATA_SIZE:
                break
                
        except Exception as e:
            print(f"Error processing image {img_path}: {str(e)}")
            continue

    # Convert lists to numpy arrays
    images = np.array(images)
    labels = np.array(labels)
    
    # Shuffle the dataset
    indices = np.random.permutation(len(images))
    images = images[indices]
    labels = labels[indices]
    
    print(f"Total images processed: {len(images)}")
    return images, labels


if __name__ == "__main__":
    # Test the data loading function
    prep_and_load_data()
    



