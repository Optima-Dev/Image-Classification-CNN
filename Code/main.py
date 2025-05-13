# Import required libraries
import numpy as np
import cv2
from data_prep import prep_and_load_data
from model import get_model
import constants as CONST
import pickle
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
import tensorflow as tf
import argparse
import copy

def classify_single_image(model, image_path):
    """
    Classify a single image and return the prediction.
    Args:
        model: The trained model
        image_path: Path to the image to classify
    Returns:
        tuple: (prediction, confidence, output_path) where:
            - prediction: Class name ('Cat' or 'Dog')
            - confidence: Prediction confidence (0-1)
            - output_path: Path to the saved annotated image
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
        image = image.astype('float32') / 255.0  # Normalize to [0,1] range
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
    # Create argument parser for command-line interface
    parser = argparse.ArgumentParser(description='Cat vs Dog Classifier')
    parser.add_argument('--mode', choices=['train', 'predict'], required=True, help='Mode: train or predict')
    parser.add_argument('--image', help='Path to image for prediction (required in predict mode)')
    parser.add_argument('--epochs', type=int, default=15, help='Number of training epochs (default: 15)')
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

        # Create callbacks for training
        tensorboard = TensorBoard(log_dir='logs')  # For monitoring training progress
        checkpoint = ModelCheckpoint(  # Save best model during training
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
            # Train the model
            history = model.fit(
                train_images, 
                train_labels,
                batch_size=CONST.BATCH_SIZE,
                epochs=args.epochs,
                verbose=1,
                validation_data=(test_images, test_labels),
                callbacks=[tensorboard, checkpoint]
            )
            print('Training completed successfully')
            
            # Save final model
            model.save('final_model.h5')
            
            # Save training history
            history_file = 'training_history.pickle'
            with open(history_file, 'wb') as file:
                pickle.dump(history.history, file)
            
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




