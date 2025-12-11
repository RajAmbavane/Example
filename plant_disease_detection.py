"""
Smart Plant Health Monitoring System
DATA-602 Final Project
Team: Raj Prasad Ambavane, Dhruv Dubey, Vansh Pradeep Jain, Shantanu Ramavat
"""

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd
import os
from datetime import datetime

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class PlantDiseaseDetector:
    """
    A comprehensive plant disease detection system using transfer learning
    with EfficientNetB0 architecture.
    """
    
    def __init__(self, train_dir, valid_dir, test_dir=None, img_size=224, batch_size=32):
        """
        Initialize the plant disease detector.
        
        Args:
            train_dir (str): Path to training data directory
            valid_dir (str): Path to validation data directory
            test_dir (str): Path to test data directory (optional, will use valid_dir if None)
            img_size (int): Size to resize images (default: 224)
            batch_size (int): Batch size for training (default: 32)
        """
        self.train_dir = train_dir
        self.valid_dir = valid_dir
        self.test_dir = test_dir if test_dir is not None else valid_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.model = None
        self.history = None
        self.class_names = None
        
        print(f"Initializing Plant Disease Detector")
        print(f"Image size: {img_size}x{img_size}")
        print(f"Batch size: {batch_size}")
    
    def prepare_data(self):
        """
        Prepare data generators with augmentation for training and validation.
        """
        print("\n" + "="*50)
        print("PREPARING DATA")
        print("="*50)
        
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            shear_range=0.2,
            fill_mode='nearest'
        )
        
        # Only rescaling for validation/test
        valid_datagen = ImageDataGenerator(rescale=1./255)
        
        # Load training data
        self.train_generator = train_datagen.flow_from_directory(
            self.train_dir,
            target_size=(self.img_size, self.img_size),
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=True
        )
        
        # Load validation data
        self.valid_generator = valid_datagen.flow_from_directory(
            self.valid_dir,
            target_size=(self.img_size, self.img_size),
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        # Load test data
        self.test_generator = valid_datagen.flow_from_directory(
            self.test_dir,
            target_size=(self.img_size, self.img_size),
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        # Store class names
        self.class_names = list(self.train_generator.class_indices.keys())
        self.num_classes = len(self.class_names)
        
        print(f"\nTraining samples: {self.train_generator.samples}")
        print(f"Validation samples: {self.valid_generator.samples}")
        print(f"Test samples: {self.test_generator.samples}")
        if self.test_dir == self.valid_dir:
            print("Note: Using validation set for testing (no separate test set)")
        print(f"Number of classes: {self.num_classes}")
        print(f"\nSample classes: {self.class_names[:5]}...")
        
    def build_model(self):
        """
        Build the model using transfer learning with EfficientNetB0.
        """
        print("\n" + "="*50)
        print("BUILDING MODEL")
        print("="*50)
        
        # Load pre-trained EfficientNetB0
        base_model = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=(self.img_size, self.img_size, 3)
        )
        
        # Freeze base model
        base_model.trainable = False
        
        # Add custom classification head
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu', name='dense_1')(x)
        x = Dropout(0.5, name='dropout_1')(x)
        x = Dense(256, activation='relu', name='dense_2')(x)
        x = Dropout(0.3, name='dropout_2')(x)
        predictions = Dense(self.num_classes, activation='softmax', name='output')(x)
        
        # Create model
        self.model = Model(inputs=base_model.input, outputs=predictions)
        
        # Compile model
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top_5_accuracy')]
        )
        
        print(f"\nModel built successfully!")
        print(f"Total parameters: {self.model.count_params():,}")
        print(f"Trainable parameters: {sum([tf.size(w).numpy() for w in self.model.trainable_weights]):,}")
        
    def train(self, epochs=20, fine_tune=True):
        """
        Train the model with callbacks and optional fine-tuning.
        
        Args:
            epochs (int): Number of epochs to train
            fine_tune (bool): Whether to fine-tune the base model after initial training
        """
        print("\n" + "="*50)
        print("TRAINING MODEL")
        print("="*50)
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                'best_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Initial training
        print("\n--- Phase 1: Training with frozen base model ---")
        self.history = self.model.fit(
            self.train_generator,
            epochs=epochs,
            validation_data=self.valid_generator,
            callbacks=callbacks,
            verbose=1
        )
        
        # Fine-tuning
        if fine_tune:
            print("\n--- Phase 2: Fine-tuning ---")
            
            # Unfreeze the base model
            base_model = self.model.layers[0]
            base_model.trainable = True
            
            # Freeze first 100 layers
            for layer in base_model.layers[:100]:
                layer.trainable = False
            
            # Recompile with lower learning rate
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                loss='categorical_crossentropy',
                metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top_5_accuracy')]
            )
            
            print(f"Trainable parameters after unfreezing: {sum([tf.size(w).numpy() for w in self.model.trainable_weights]):,}")
            
            # Continue training
            history_fine = self.model.fit(
                self.train_generator,
                epochs=10,
                validation_data=self.valid_generator,
                callbacks=callbacks,
                verbose=1
            )
            
            # Combine histories
            for key in self.history.history.keys():
                self.history.history[key].extend(history_fine.history[key])
    
    def evaluate(self):
        """
        Evaluate the model on test data and generate comprehensive metrics.
        """
        print("\n" + "="*50)
        print("EVALUATING MODEL")
        print("="*50)
        
        # Get predictions
        print("\nGenerating predictions...")
        predictions = self.model.predict(self.test_generator, verbose=1)
        y_pred = np.argmax(predictions, axis=1)
        y_true = self.test_generator.classes
        
        # Calculate metrics
        test_loss, test_acc, test_top5 = self.model.evaluate(self.test_generator, verbose=0)
        
        print(f"\n{'Metric':<20} {'Value':<10}")
        print("-" * 30)
        print(f"{'Test Loss:':<20} {test_loss:.4f}")
        print(f"{'Test Accuracy:':<20} {test_acc:.4f}")
        print(f"{'Top-5 Accuracy:':<20} {test_top5:.4f}")
        
        # Classification report
        print("\n" + "="*50)
        print("CLASSIFICATION REPORT")
        print("="*50)
        print(classification_report(y_true, y_pred, target_names=self.class_names))
        
        # Save confusion matrix
        self.confusion_matrix = confusion_matrix(y_true, y_pred)
        
        return {
            'test_loss': test_loss,
            'test_accuracy': test_acc,
            'top_5_accuracy': test_top5,
            'predictions': predictions,
            'y_true': y_true,
            'y_pred': y_pred
        }
    
    def plot_training_history(self, save_path='training_history.png'):
        """
        Plot training and validation accuracy/loss over epochs.
        """
        if self.history is None:
            print("No training history available. Train the model first.")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Accuracy plot
        ax1.plot(self.history.history['accuracy'], label='Train Accuracy', linewidth=2)
        ax1.plot(self.history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
        ax1.set_title('Model Accuracy Over Epochs', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Loss plot
        ax2.plot(self.history.history['loss'], label='Train Loss', linewidth=2)
        ax2.plot(self.history.history['val_loss'], label='Val Loss', linewidth=2)
        ax2.set_title('Model Loss Over Epochs', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Loss', fontsize=12)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nTraining history plot saved to {save_path}")
        plt.show()
    
    def plot_confusion_matrix(self, save_path='confusion_matrix.png', figsize=(20, 18)):
        """
        Plot confusion matrix heatmap.
        """
        if not hasattr(self, 'confusion_matrix'):
            print("No confusion matrix available. Evaluate the model first.")
            return
        
        plt.figure(figsize=figsize)
        sns.heatmap(
            self.confusion_matrix,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={'label': 'Count'}
        )
        plt.title('Confusion Matrix - Plant Disease Classification', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Predicted Label', fontsize=14)
        plt.ylabel('True Label', fontsize=14)
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nConfusion matrix saved to {save_path}")
        plt.show()
    
    def predict_image(self, image_path, top_k=5):
        """
        Predict disease for a single image and visualize results.
        
        Args:
            image_path (str): Path to image file
            top_k (int): Number of top predictions to show
        """
        # Load and preprocess image
        img = tf.keras.preprocessing.image.load_img(
            image_path, 
            target_size=(self.img_size, self.img_size)
        )
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        
        # Predict
        predictions = self.model.predict(img_array, verbose=0)
        predicted_class_idx = np.argmax(predictions)
        predicted_class = self.class_names[predicted_class_idx]
        confidence = predictions[0][predicted_class_idx] * 100
        
        # Get top K predictions
        top_k_idx = np.argsort(predictions[0])[-top_k:][::-1]
        top_k_classes = [self.class_names[i] for i in top_k_idx]
        top_k_probs = [predictions[0][i] * 100 for i in top_k_idx]
        
        # Visualize
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Show image
        ax1.imshow(img)
        ax1.set_title(f'Predicted: {predicted_class}\nConfidence: {confidence:.2f}%', 
                     fontsize=12, fontweight='bold')
        ax1.axis('off')
        
        # Show top predictions
        colors = ['green' if i == 0 else 'lightgreen' for i in range(top_k)]
        ax2.barh(range(top_k), top_k_probs, color=colors, alpha=0.7)
        ax2.set_yticks(range(top_k))
        ax2.set_yticklabels(top_k_classes)
        ax2.set_xlabel('Confidence (%)', fontsize=12)
        ax2.set_title(f'Top {top_k} Predictions', fontsize=12, fontweight='bold')
        ax2.set_xlim(0, 100)
        ax2.invert_yaxis()
        ax2.grid(True, axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return predicted_class, confidence, dict(zip(top_k_classes, top_k_probs))
    
    def save_model(self, filepath='plant_disease_model.h5'):
        """Save the trained model."""
        self.model.save(filepath)
        print(f"\nModel saved to {filepath}")
    
    def load_saved_model(self, filepath='plant_disease_model.h5'):
        """Load a saved model."""
        self.model = load_model(filepath)
        print(f"\nModel loaded from {filepath}")


def main():
    """
    Main function to run the complete pipeline.
    """
    print("="*70)
    print(" "*15 + "SMART PLANT HEALTH MONITORING SYSTEM")
    print("="*70)
    print("\nDATA-602 Final Project")
    print("Team: Raj Prasad Ambavane, Dhruv Dubey, Vansh Pradeep Jain, Shantanu Ramavat")
    print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Set paths - Dataset only has train and valid folders
    TRAIN_DIR = 'dataset/train'
    VALID_DIR = 'dataset/valid'
    # No separate test folder, will use validation for testing
    
    # Initialize detector
    detector = PlantDiseaseDetector(
        train_dir=TRAIN_DIR,
        valid_dir=VALID_DIR,
        test_dir=None,  # Will use valid_dir for testing
        img_size=224,
        batch_size=32
    )
    
    # Prepare data
    detector.prepare_data()
    
    # Build model
    detector.build_model()
    
    # Train model
    detector.train(epochs=20, fine_tune=True)
    
    # Evaluate
    results = detector.evaluate()
    
    # Visualizations
    detector.plot_training_history()
    detector.plot_confusion_matrix()
    
    # Save model
    detector.save_model('plant_disease_model.h5')
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"\nFinal Test Accuracy: {results['test_accuracy']*100:.2f}%")
    print(f"Top-5 Accuracy: {results['top_5_accuracy']*100:.2f}%")
    print(f"\nFinished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    

if __name__ == "__main__":
    main()