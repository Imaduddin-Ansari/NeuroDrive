import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from pathlib import Path

# Configuration
DATASET_PATH = "/content/Pakistan/BalancedData"  # Update this path
IMG_SIZE = 64
BATCH_SIZE = 64
EPOCHS = 50
VAL_SPLIT = 0.3

def load_dataset(dataset_path, img_size=IMG_SIZE):
    """Load images and labels from dataset directory"""
    images = []
    labels = []
    class_names = []

    # Get class folders
    class_folders = sorted([d for d in os.listdir(dataset_path)
                           if os.path.isdir(os.path.join(dataset_path, d))])

    print(f"Found {len(class_folders)} classes: {class_folders}\n")

    for class_idx, class_name in enumerate(class_folders):
        class_path = os.path.join(dataset_path, class_name)
        image_files = [f for f in os.listdir(class_path)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

        print(f"Loading class '{class_name}': {len(image_files)} images")
        class_names.append(class_name)

        for img_file in image_files:
            try:
                img_path = os.path.join(class_path, img_file)
                image = cv2.imread(img_path)

                if image is None:
                    continue

                # Resize image
                image = cv2.resize(image, (img_size, img_size))
                # Normalize pixel values
                image = image / 255.0

                images.append(image)
                labels.append(class_idx)
            except Exception as e:
                print(f"  Error loading {img_file}: {e}")
                continue

    return np.array(images), np.array(labels), class_names

def build_model(num_classes):
    """Build CNN model"""
    model = keras.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),

        # First block
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(2),
        layers.Dropout(0.25),

        # Second block
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(2),
        layers.Dropout(0.25),

        # Third block
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(2),
        layers.Dropout(0.25),

        # Dense layers
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    return model

def train_model():
    """Main training function"""
    print("=" * 50)
    print("Traffic Sign Classification Training")
    print("=" * 50 + "\n")

    # Load dataset
    print("Loading dataset...")
    X, y, class_names = load_dataset(DATASET_PATH)
    num_classes = len(class_names)

    print(f"\nDataset shape: {X.shape}")
    print(f"Number of classes: {num_classes}")
    print(f"Total images: {len(X)}\n")

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}\n")

    # Build model
    print("Building model...")
    model = build_model(num_classes)
    model.summary()

    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Callbacks
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=15, restore_best_weights=True
    )

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7
    )

    # Train model
    print("\nTraining model...")
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=VAL_SPLIT,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )

    # Evaluate on test set
    print("\n" + "=" * 50)
    print("Evaluating on test set...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}")

    # Plot training history
    plot_history(history)

    # Save model
    model_path = "traffic_sign_model.h5"
    model.save(model_path)
    print(f"\n✓ Model saved to {model_path}")

    # Save class names
    np.save("class_names.npy", np.array(class_names))
    print("✓ Class names saved")

    return model, history, class_names

def plot_history(history):
    """Plot training history"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    axes[0].plot(history.history['accuracy'], label='Train Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)

    # Loss
    axes[1].plot(history.history['loss'], label='Train Loss')
    axes[1].plot(history.history['val_loss'], label='Val Loss')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=100)
    print("✓ Training history plot saved as 'training_history.png'")
    plt.show()

if __name__ == "__main__":
    model, history, class_names = train_model()
    print("\n" + "=" * 50)
    print("✓ Training completed!")
    print("=" * 50)