# Import required libraries
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import constants as CONST

def get_model():
    """
    Create and return a Convolutional Neural Network (CNN) model for cat/dog classification.
    
    Architecture:
    1. Input Layer: Takes images of size (IMG_SIZE x IMG_SIZE x 3)
    2. Convolutional Blocks: 5 blocks, each containing:
       - Conv2D layer with increasing filters (32->64->96->96->64)
       - MaxPooling2D for dimensionality reduction
       - BatchNormalization for training stability
       - Dropout in later layers to prevent overfitting
    3. Dense Layers: 3 fully connected layers (256->128->2)
    
    Returns:
        tf.keras.Model: Compiled CNN model ready for training
    """
    # Initialize sequential model
    model = tf.keras.Sequential()
    
    # First Convolutional Block
    # Input shape: (IMG_SIZE x IMG_SIZE x 3)
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', 
                     input_shape=(CONST.IMG_SIZE, CONST.IMG_SIZE, 3)))
    model.add(MaxPooling2D(pool_size=(2,2)))  # Reduce spatial dimensions by half
    model.add(BatchNormalization())  # Normalize activations
    
    # Second Convolutional Block
    # Increase filters to capture more complex features
    model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    
    # Third Convolutional Block
    # Further increase filters for higher-level features
    model.add(Conv2D(96, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    
    # Fourth Convolutional Block
    # Maintain filter count but add dropout for regularization
    model.add(Conv2D(96, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))  # Drop 20% of neurons to prevent overfitting
    
    # Fifth Convolutional Block
    # Reduce filters and add dropout
    model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    # Flatten layer to convert 2D feature maps to 1D vector
    model.add(Flatten())
    
    # Dense (Fully Connected) Layers
    # First dense layer with 256 neurons
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    
    # Second dense layer with 128 neurons
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))  # Higher dropout rate for final layers
    
    # Output layer with 2 neurons (one for each class: cat and dog)
    # Softmax activation for probability distribution
    model.add(Dense(2, activation='softmax'))

    # Compile the model
    # - binary_crossentropy: Loss function for binary classification
    # - adam: Optimizer for training
    # - accuracy: Metric to monitor during training
    model.compile(loss='binary_crossentropy', 
                 optimizer='adam', 
                 metrics=['accuracy'])
    
    print('Model architecture prepared...')
    return model
