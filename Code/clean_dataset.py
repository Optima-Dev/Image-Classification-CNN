import os
import shutil
import random

def clean_dataset(train_dir, num_images_per_class=1500):
    # Get all image files
    cat_images = [f for f in os.listdir(train_dir) if f.startswith('cat.')]
    dog_images = [f for f in os.listdir(train_dir) if f.startswith('dog.')]

    # Sort images to ensure consistent selection
    cat_images.sort()
    dog_images.sort()

    # Select first 1500 images of each class
    cat_images_to_keep = cat_images[:num_images_per_class]
    dog_images_to_keep = dog_images[:num_images_per_class]

    # Create a set of images to keep
    images_to_keep = set(cat_images_to_keep + dog_images_to_keep)

    # Remove images that are not in the keep set
    removed_count = 0
    for img in os.listdir(train_dir):
        if img not in images_to_keep:
            os.remove(os.path.join(train_dir, img))
            removed_count += 1

    print(f"Kept {len(cat_images_to_keep)} cat images and {len(dog_images_to_keep)} dog images")
    print(f"Removed {removed_count} images")

if __name__ == "__main__":
    train_dir = "dogs-vs-cats/train"
    clean_dataset(train_dir) 