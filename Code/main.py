import numpy as np
import math
import os
import cv2
from data_prep import prep_and_load_data
from model import get_model
import constants as CONST
import pickle
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
import tensorflow as tf
import argparse

from matplotlib import pyplot as plt
import copy

def plotter(history_file):
    with open(history_file, 'rb') as file:
        history = pickle.load(file)
    
    plt.figure(1)
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('18_000_15epoch_accuracy.png')
    plt.close()

    plt.figure(2)
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('18_000_15epoch_loss.png')
    plt.close()


def video_write(model):
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter("./prediction.mp4", fourcc, 1.0, (400,400))
    val_map = {1: 'Dog', 0: 'Cat'}

    font = cv2.FONT_HERSHEY_SIMPLEX
    location = (20,20)
    fontScale = 0.5
    fontColor = (255,255,255)
    lineType  = 2

    DIR = CONST.TEST_DIR
    image_paths = os.listdir(DIR)
    image_paths = image_paths[:200]
    count = 0
    
    for img_path in image_paths:
        try:
            image, image_std = process_image(DIR, img_path)
            if image is None or image_std is None:
                continue
                
            image_std = image_std.reshape(-1, CONST.IMG_SIZE, CONST.IMG_SIZE, 3)
            pred = model.predict(image_std, verbose=0)
            arg_max = np.argmax(pred, axis=1)
            max_val = np.max(pred, axis=1)
            s = val_map[arg_max[0]] + ' - ' + str(round(max_val[0]*100, 2)) + '%'
            cv2.putText(image, s, 
                location, 
                font, 
                fontScale,
                fontColor,
                lineType)
            
            frame = cv2.resize(image, (400, 400))
            out.write(frame)
            
            count += 1
            if count % 10 == 0:
                print(f"Processed {count} test images")
        except Exception as e:
            print(f"Error processing test image {img_path}: {str(e)}")
            continue
            
    out.release()
    print("Video processing complete")


def process_image(directory, img_path):
    try:
        path = os.path.join(directory, img_path)
        image = cv2.imread(path)
        if image is None:
            print(f"Could not load image: {img_path}")
            return None, None
            
        image_copy = copy.deepcopy(image)
        image = cv2.resize(image, (CONST.IMG_SIZE, CONST.IMG_SIZE))
        image_std = image.astype('float32') / 255.0
        return image_copy, image_std
    except Exception as e:
        print(f"Error in process_image for {img_path}: {str(e)}")
        return None, None


def classify_single_image(model, image_path):
    """
    Classify a single image and return the prediction.
    Args:
        model: The trained model
        image_path: Path to the image to classify
    Returns:
        tuple: (prediction, confidence)
    """
    try:
        # Read and preprocess the image
        image = cv2.imread(image_path)
        if image is None:
            return None, "Could not load image"
            
        # Make a copy for display
        display_image = copy.deepcopy(image)
        
        # Preprocess the image
        image = cv2.resize(image, (CONST.IMG_SIZE, CONST.IMG_SIZE))
        image = image.astype('float32') / 255.0
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        
        # Make prediction
        pred = model.predict(image, verbose=0)
        class_idx = np.argmax(pred, axis=1)[0]
        confidence = float(pred[0][class_idx])
        
        # Map prediction to class name
        class_names = {0: 'Cat', 1: 'Dog'}
        prediction = class_names[class_idx]
        
        # Add text to image
        font = cv2.FONT_HERSHEY_SIMPLEX
        location = (20, 30)
        fontScale = 0.8
        fontColor = (0, 255, 0)  # Green
        lineType = 2
        
        text = f"{prediction}: {confidence:.2%}"
        cv2.putText(display_image, text, location, font, fontScale, fontColor, lineType)
        
        # Save the annotated image
        output_path = "prediction_result.jpg"
        cv2.imwrite(output_path, display_image)
        
        return prediction, confidence, output_path
        
    except Exception as e:
        return None, f"Error during classification: {str(e)}"


if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(description='Cat vs Dog Classifier')
    parser.add_argument('--mode', choices=['train', 'predict'], required=True, help='Mode: train or predict')
    parser.add_argument('--image', help='Path to image for prediction (required in predict mode)')
    args = parser.parse_args()

    if args.mode == 'train':
        # Training mode
        # Enable memory growth for GPU if available
        physical_devices = tf.config.list_physical_devices('GPU')
        try:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
        except:
            pass

        # Load and preprocess data
        images, labels = prep_and_load_data()
        train_size = int(len(images) * CONST.SPLIT_RATIO)
        print('Total images:', len(images), 'Train size:', train_size)

        # Split into train and test sets
        train_images = images[:train_size]
        train_labels = labels[:train_size]
        print('Train data shape:', train_images.shape)

        test_images = images[train_size:]
        test_labels = labels[train_size:]
        print('Test data shape:', test_images.shape)

        # Create callbacks
        tensorboard = TensorBoard(log_dir='logs')
        checkpoint = ModelCheckpoint(
            'best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
        
        # Get and train model
        model = get_model()
        print('Training started...')
        
        try:
            history = model.fit(
                train_images, 
                train_labels,
                batch_size=CONST.BATCH_SIZE,
                epochs=15,
                verbose=1,
                validation_data=(test_images, test_labels),
                callbacks=[tensorboard, checkpoint]
            )
            print('Training completed successfully')
            
            # Save final model
            model.save('final_model.h5')
            
            # Save and plot history
            history_file = 'training_history.pickle'
            with open(history_file, 'wb') as file:
                pickle.dump(history.history, file)
            
            plotter(history_file)
            video_write(model)
            
        except Exception as e:
            print(f"Error during training: {str(e)}")
            
    elif args.mode == 'predict':
        # Prediction mode
        if not args.image:
            print("Error: --image argument is required in predict mode")
            exit(1)
            
        # Load the trained model
        try:
            model = tf.keras.models.load_model('final_model.h5')
            print("Model loaded successfully")
        except:
            print("Error: Could not load model. Make sure you have trained the model first.")
            exit(1)
            
        # Classify the image
        prediction, confidence, output_path = classify_single_image(model, args.image)
        if prediction:
            print(f"\nPrediction: {prediction}")
            print(f"Confidence: {confidence:.2%}")
            print(f"Annotated image saved as: {output_path}")
        else:
            print(f"Error: {confidence}")




