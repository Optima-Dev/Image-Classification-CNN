# Import required libraries
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import os
from main import get_model, prep_and_load_data, classify_single_image
import constants as CONST
from tensorflow.keras.callbacks import TensorBoard
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image, ImageTk
import numpy as np
import time

class CatDogClassifierGUI:
    """
    GUI application for training and using a Cat vs Dog classifier.
    Provides interface for:
    - Training the model
    - Classifying individual images
    - Viewing training progress and results
    """
    def __init__(self, root):
        """
        Initialize the GUI application.
        Args:
            root: The root Tkinter window
        """
        self.root = root
        self.root.title("Cat vs Dog Classifier")
        self.root.geometry("1000x700")
        self.root.configure(bg='#f0f0f0')
        
        # Configure style for GUI elements
        self.style = ttk.Style()
        self.style.configure('TFrame', background='#f0f0f0')
        self.style.configure('TLabel', background='#f0f0f0', font=('Arial', 10))
        self.style.configure('TButton', font=('Arial', 10))
        self.style.configure('TLabelframe', background='#f0f0f0')
        self.style.configure('TLabelframe.Label', font=('Arial', 11, 'bold'))
        
        # Create main layout frames
        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Left frame for training and classification
        self.left_frame = ttk.Frame(self.main_frame)
        self.left_frame.grid(row=0, column=0, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Right frame for logs and statistics
        self.right_frame = ttk.Frame(self.main_frame)
        self.right_frame.grid(row=0, column=1, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Initialize training section
        self._init_training_section()
        
        # Initialize classification section
        self._init_classification_section()
        
        # Initialize log section
        self._init_log_section()
        
        # Initialize statistics section
        self._init_stats_section()
        
        # Initialize model and training variables
        self.model = None
        self.training_thread = None
        self.current_epoch = 0
        self.total_epochs = 15
        self.training_start_time = None

    def _init_training_section(self):
        """Initialize the training section of the GUI"""
        # Training frame
        self.training_frame = ttk.LabelFrame(self.left_frame, text="Training", padding="10")
        self.training_frame.grid(row=0, column=0, padx=5, pady=5, sticky=(tk.W, tk.E))
        
        # Training button
        self.train_button = ttk.Button(self.training_frame, text="Start Training", command=self.start_training)
        self.train_button.grid(row=0, column=0, padx=5, pady=5)
        
        # Progress bar
        self.progress_bar = ttk.Progressbar(self.training_frame, orient=tk.HORIZONTAL, length=300, mode='determinate')
        self.progress_bar.grid(row=1, column=0, padx=5, pady=5)
        
        # Status label
        self.status_label = ttk.Label(self.training_frame, text="Ready to train")
        self.status_label.grid(row=2, column=0, padx=5, pady=5)
        
        # Training parameters frame
        self.params_frame = ttk.LabelFrame(self.training_frame, text="Training Parameters", padding="5")
        self.params_frame.grid(row=3, column=0, padx=5, pady=5, sticky=(tk.W, tk.E))
        
        # Epochs input
        ttk.Label(self.params_frame, text="Epochs:").grid(row=0, column=0, padx=5, pady=2)
        self.epochs_var = tk.StringVar(value="15")
        self.epochs_entry = ttk.Entry(self.params_frame, textvariable=self.epochs_var, width=5)
        self.epochs_entry.grid(row=0, column=1, padx=5, pady=2)

    def _init_classification_section(self):
        """Initialize the classification section of the GUI"""
        # Classification frame
        self.classification_frame = ttk.LabelFrame(self.left_frame, text="Classification", padding="10")
        self.classification_frame.grid(row=1, column=0, padx=5, pady=5, sticky=(tk.W, tk.E))
        
        # Image display area
        self.image_frame = ttk.Frame(self.classification_frame)
        self.image_frame.grid(row=0, column=0, padx=5, pady=5)
        
        self.image_label = ttk.Label(self.image_frame, text="No image selected")
        self.image_label.grid(row=0, column=0, padx=5, pady=5)
        
        # Select image button
        self.select_button = ttk.Button(self.classification_frame, text="Select Image", command=self.select_image)
        self.select_button.grid(row=1, column=0, padx=5, pady=5)
        
        # Result label
        self.result_label = ttk.Label(self.classification_frame, text="No image selected")
        self.result_label.grid(row=2, column=0, padx=5, pady=5)

    def _init_log_section(self):
        """Initialize the log section of the GUI"""
        # Log frame
        self.log_frame = ttk.LabelFrame(self.right_frame, text="Training Log", padding="10")
        self.log_frame.grid(row=0, column=0, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Log text area
        self.log_text = scrolledtext.ScrolledText(self.log_frame, width=50, height=20)
        self.log_text.grid(row=0, column=0, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.log_text.config(state='disabled')

    def _init_stats_section(self):
        """Initialize the statistics section of the GUI"""
        # Statistics frame
        self.stats_frame = ttk.LabelFrame(self.right_frame, text="Model Statistics", padding="10")
        self.stats_frame.grid(row=1, column=0, padx=5, pady=5, sticky=(tk.W, tk.E))
        
        # Accuracy label
        self.accuracy_label = ttk.Label(self.stats_frame, text="Accuracy: --")
        self.accuracy_label.grid(row=0, column=0, padx=5, pady=2)
        
        # Loss label
        self.loss_label = ttk.Label(self.stats_frame, text="Loss: --")
        self.loss_label.grid(row=1, column=0, padx=5, pady=2)
        
    def log_message(self, message):
        """
        Add a message to the log text area.
        Args:
            message (str): Message to add to the log
        """
        self.log_text.config(state='normal')
        self.log_text.insert(tk.END, f"{time.strftime('%H:%M:%S')} - {message}\n")
        self.log_text.see(tk.END)
        self.log_text.config(state='disabled')
        
    def start_training(self):
        """Start the model training process in a separate thread"""
        if self.training_thread and self.training_thread.is_alive():
            messagebox.showwarning("Warning", "Training is already in progress!")
            return
            
        try:
            self.total_epochs = int(self.epochs_var.get())
            CONST.BATCH_SIZE = 32  # Default batch size
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numbers for epochs")
            return
            
        self.train_button.config(state='disabled')
        self.progress_bar['value'] = 0
        self.status_label.config(text="Training in progress...")
        self.training_start_time = time.time()
        self.log_message("Starting training...")
        
        # Start training in a separate thread
        self.training_thread = threading.Thread(target=self.train_model)
        self.training_thread.daemon = True
        self.training_thread.start()
        
    def update_progress(self, epoch, logs=None):
        """
        Update the training progress display.
        Args:
            epoch (int): Current epoch number
            logs (dict): Training metrics for the current epoch
        """
        self.current_epoch = epoch
        progress = (epoch / self.total_epochs) * 100
        self.progress_bar['value'] = progress
        self.status_label.config(text=f"Training in progress... Epoch {epoch}/{self.total_epochs}")
        
        if logs:
            self.accuracy_label.config(text=f"Accuracy: {logs.get('accuracy', 0):.4f}")
            self.loss_label.config(text=f"Loss: {logs.get('loss', 0):.4f}")
            self.log_message(f"Epoch {epoch}/{self.total_epochs} - Loss: {logs.get('loss', 0):.4f} - Accuracy: {logs.get('accuracy', 0):.4f}")
        
    def train_model(self):
        """Train the model and update the GUI with progress"""
        try:
            # Enable memory growth for GPU if available
            physical_devices = tf.config.list_physical_devices('GPU')
            try:
                for device in physical_devices:
                    tf.config.experimental.set_memory_growth(device, True)
            except:
                pass

            # Load and preprocess data
            self.log_message("Loading and preprocessing data...")
            images, labels = prep_and_load_data()
            train_size = int(len(images) * CONST.SPLIT_RATIO)

            # Split into train and test sets
            train_images = images[:train_size]
            train_labels = labels[:train_size]
            test_images = images[train_size:]
            test_labels = labels[train_size:]
            
            self.log_message(f"Training set size: {len(train_images)}")
            self.log_message(f"Test set size: {len(test_images)}")

            # Custom callback for progress updates
            class ProgressCallback(tf.keras.callbacks.Callback):
                def __init__(self, gui):
                    self.gui = gui
                
                def on_epoch_end(self, epoch, logs=None):
                    self.gui.root.after(0, lambda: self.gui.update_progress(epoch + 1, logs))
            
            # Get and train model
            self.log_message("Creating model...")
            self.model = get_model()
            
            self.log_message("Starting training...")
            history = self.model.fit(
                train_images, 
                train_labels,
                batch_size=CONST.BATCH_SIZE,
                epochs=self.total_epochs,
                verbose=1,
                validation_data=(test_images, test_labels),
                callbacks=[ProgressCallback(self)]
            )
            
            # Calculate and display training time
            training_time = time.time() - self.training_start_time
            self.log_message(f"Training completed in {training_time/60:.2f} minutes")
            
            # Update GUI
            self.root.after(0, self.training_complete)
            
        except Exception as e:
            error_message = str(e)
            self.log_message(f"Error during training: {error_message}")
            self.root.after(0, lambda: self.training_error(error_message))
            
    def training_complete(self):
        """Handle completion of training"""
        self.progress_bar['value'] = 100
        self.status_label.config(text="Training completed successfully!")
        self.train_button.config(state='normal')
        self.log_message("Training completed successfully!")
        messagebox.showinfo("Success", "Model training completed successfully!")
        
    def training_error(self, error_message):
        """Handle training errors"""
        self.progress_bar['value'] = 0
        self.status_label.config(text="Training failed!")
        self.train_button.config(state='normal')
        self.log_message(f"Training failed: {error_message}")
        messagebox.showerror("Error", f"Training failed: {error_message}")
        
    def display_image(self, image_path):
        """
        Display the selected image in the GUI.
        Args:
            image_path (str): Path to the image to display
        """
        try:
            # Load and resize image
            image = Image.open(image_path)
            image = image.resize((300, 300), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image)
            
            # Update image label
            self.image_label.configure(image=photo)
            self.image_label.image = photo  # Keep a reference
            
        except Exception as e:
            self.image_label.configure(text=f"Error loading image: {str(e)}")
        
    def select_image(self):
        """Handle image selection and classification"""
        if self.model is None:
            messagebox.showerror("Error", "Please train the model first!")
            return
            
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        
        if file_path:
            try:
                # Display selected image
                self.display_image(file_path)
                
                # Use the current model instead of loading from file
                prediction, confidence, output_path = classify_single_image(self.model, file_path)
                if prediction:
                    self.result_label.config(
                        text=f"Prediction: {prediction}\nConfidence: {confidence:.2%}"
                    )
                    self.log_message(f"Classified image as {prediction} with {confidence:.2%} confidence")
                else:
                    self.result_label.config(text=f"Error: {confidence}")
                    self.log_message(f"Classification error: {confidence}")
                    
            except Exception as e:
                self.result_label.config(text=f"Error during classification: {str(e)}")
                self.log_message(f"Classification error: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = CatDogClassifierGUI(root)
    root.mainloop() 