import numpy as np
import math
import os 
from random import shuffle
import constants as CONST 
import cv2
import pickle

def get_size_statistics():
    heights = []
    widths = []
    img_count = 0
    DIR = CONST.TRAIN_DIR
    for img in os.listdir(CONST.TRAIN_DIR):
        path = os.path.join(DIR, img)
        data = cv2.imread(path)
        if data is not None:  # Check if image was loaded successfully
            heights.append(data.shape[0])
            widths.append(data.shape[1])
            img_count += 1
    avg_height = sum(heights) / len(heights)
    avg_width = sum(widths) / len(widths)
    print("Average Height: " + str(avg_height))
    print("Max Height: " + str(max(heights)))
    print("Min Height: " + str(min(heights)))
    print('\n')
    print("Average Width: " + str(avg_width))
    print("Max Width: " + str(max(widths)))
    print("Min Width: " + str(min(widths)))

#get_size_statistics()


def label_img(name):
    word_label = name.split('.')[0]
    if word_label not in CONST.LABEL_MAP:
        return None
    label = CONST.LABEL_MAP[word_label]
    label_arr = np.zeros(2)
    label_arr[label] = 1
    return label_arr


def prep_and_load_data():
    DIR = CONST.TRAIN_DIR
    images = []
    labels = []
    image_paths = os.listdir(DIR)
    shuffle(image_paths)
    count = 0
    for img_path in image_paths:
        try:
            label = label_img(img_path)
            if label is None:
                continue
                
            path = os.path.join(DIR, img_path)
            image = cv2.imread(path)
            if image is None:
                print(f"Could not load image: {img_path}")
                continue
                
            image = cv2.resize(image, (CONST.IMG_SIZE, CONST.IMG_SIZE))
            image = image.astype('float') / 255.0 
            images.append(image)
            labels.append(label)
            count += 1
            if count % 100 == 0:
                print(f"Processed {count} images")
            if count == CONST.DATA_SIZE:
                break
        except Exception as e:
            print(f"Error processing image {img_path}: {str(e)}")
            continue

    # Convert to numpy arrays with proper shapes
    images = np.array(images)
    labels = np.array(labels)
    
    # Create a random permutation
    indices = np.random.permutation(len(images))
    images = images[indices]
    labels = labels[indices]
    
    print(f"Total images processed: {len(images)}")
    return images, labels


if __name__ == "__main__":
    prep_and_load_data()
    



