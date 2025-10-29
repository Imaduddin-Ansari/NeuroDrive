"""
OPTIMIZED PIPELINE FOR GOOGLE COLAB
Pakistani Traffic Signs with GTSRB Transfer Learning
FASTER TRAINING - 5-10 min per epoch instead of 20-40 min
"""

import os
import shutil
import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json

print("="*70)
print("PAKISTANI TRAFFIC SIGN CLASSIFIER - OPTIMIZED")
print("="*70)

# ============================================================
# STEP 1: ORGANIZE PAKISTANI DATASET
# ============================================================

def organize_pakistani_dataset(source_path, output_path='OrganizedDataset', train_ratio=0.8):
    """Organize Pakistani dataset into train/val"""

    print("\n" + "="*70)
    print("STEP 1: ORGANIZING PAKISTANI DATASET")
    print("="*70)

    if not os.path.exists(source_path):
        print(f"❌ ERROR: Dataset not found at {source_path}")
        return False

    class_folders = [d for d in os.listdir(source_path)
                     if os.path.isdir(os.path.join(source_path, d))
                     and not d.startswith('.')]

    if len(class_folders) == 0:
        print(f"❌ No class folders found in {source_path}")
        return False

    print(f"\n✓ Found {len(class_folders)} classes")

    train_path = os.path.join(output_path, 'train')
    val_path = os.path.join(output_path, 'val')

    if os.path.exists(output_path):
        shutil.rmtree(output_path)

    os.makedirs(train_path, exist_ok=True)
    os.makedirs(val_path, exist_ok=True)

    total_train = 0
    total_val = 0

    for class_name in sorted(class_folders):
        class_source = os.path.join(source_path, class_name)
        class_train = os.path.join(train_path, class_name)
        class_val = os.path.join(val_path, class_name)

        os.makedirs(class_train, exist_ok=True)
        os.makedirs(class_val, exist_ok=True)

        images = [f for f in os.listdir(class_source)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

        if len(images) == 0:
            continue

        random.seed(42)
        random.shuffle(images)

        n_train = max(1, int(len(images) * train_ratio))
        train_images = images[:n_train]
        val_images = images[n_train:]

        for img in train_images:
            shutil.copy2(os.path.join(class_source, img),
                        os.path.join(class_train, img))

        for img in val_images:
            shutil.copy2(os.path.join(class_source, img),
                        os.path.join(class_val, img))

        total_train += len(train_images)
        total_val += len(val_images)

    print(f"✓ Dataset organized: {total_train} train, {total_val} val")
    return True


# ============================================================
# STEP 2: TRAIN ON GTSRB (OPTIMIZED)
# ============================================================

def train_gtsrb_model(gtsrb_path, resume_from='/content/gtsrb_model_checkpoint.keras'):
    """Train model on GTSRB - FAST VERSION"""

    print("\n" + "="*70)
    print("STEP 2: TRAINING ON GTSRB (43 classes) - OPTIMIZED")
    print("="*70)
    print("Using MobileNetV2 + smaller image size = 5-10 min per epoch\n")

    gtsrb_train = os.path.join(gtsrb_path, 'Train')
    if not os.path.exists(gtsrb_train):
        gtsrb_train = os.path.join(gtsrb_path, 'train')

    if not os.path.exists(gtsrb_train):
        print(f"❌ GTSRB Train folder not found")
        return None

    IMG_SIZE = 96

    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=25,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.25,
        brightness_range=[0.7, 1.3],
        shear_range=0.15,
        fill_mode='nearest',
        horizontal_flip=False,
        validation_split=0.2
    )

    train_gen = datagen.flow_from_directory(
        gtsrb_train,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=128,
        class_mode='categorical',
        subset='training'
    )

    val_gen = datagen.flow_from_directory(
        gtsrb_train,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=128,
        class_mode='categorical',
        subset='validation'
    )

    print(f"✓ GTSRB: {train_gen.samples} train, {val_gen.samples} val")
    print(f"  Image size: {IMG_SIZE}x{IMG_SIZE}, Batch size: 128\n")

    start_epoch = 0
    if os.path.exists(resume_from):
        print(f"✓ Found saved model: {resume_from}")
        model = keras.models.load_model(resume_from)
        metadata_file = 'gtsrb_training_state.json'
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                start_epoch = metadata.get('last_epoch', 0) + 1
                print(f"Resuming from epoch {start_epoch}\n")
        else:
          start_epoch=8
    else:
        print("Building new model (MobileNetV2 - lightweight & fast)...\n")

        base_model = MobileNetV2(
            include_top=False,
            weights='imagenet',
            input_shape=(IMG_SIZE, IMG_SIZE, 3),
            pooling='avg'
        )

        base_model.trainable = True

        model = keras.Sequential([
            base_model,
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(43, activation='softmax')
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

    history = model.fit(
        train_gen,
        epochs=10,
        initial_epoch=start_epoch,
        validation_data=val_gen,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.5),
            keras.callbacks.ModelCheckpoint(
                resume_from,
                save_best_only=False,
                save_freq='epoch',
                verbose=0
            )
        ],
        verbose=1
    )

    final_epoch = start_epoch + len(history.history['val_accuracy']) - 1
    with open('/content/gtsrb_training_state.json', 'w') as f:
        json.dump({'last_epoch': final_epoch}, f)

    best_acc = max(history.history['val_accuracy'])
    print(f"\n✓ GTSRB training complete! Best accuracy: {best_acc*100:.1f}%")

    return model


# ============================================================
# STEP 3: TRANSFER TO PAKISTANI SIGNS (OPTIMIZED)
# ============================================================

def transfer_to_pakistani(gtsrb_model, pakistani_path):
    """Transfer learned features to Pakistani traffic signs - FAST"""

    print("\n" + "="*70)
    print("STEP 3: TRANSFER TO PAKISTANI SIGNS - OPTIMIZED")
    print("="*70 + "\n")

    pak_train = os.path.join(pakistani_path, 'train')
    pak_val = os.path.join(pakistani_path, 'val')

    num_classes = len([d for d in os.listdir(pak_train)
                      if os.path.isdir(os.path.join(pak_train, d))])

    IMG_SIZE = 96

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=45,
        width_shift_range=0.35,
        height_shift_range=0.35,
        zoom_range=0.5,
        brightness_range=[0.4, 1.6],
        shear_range=0.25,
        channel_shift_range=50,
        fill_mode='reflect',
        horizontal_flip=False
    )

    val_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        pak_train,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=32,
        class_mode='categorical'
    )

    val_gen = val_datagen.flow_from_directory(
        pak_val,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=32,
        class_mode='categorical'
    )

    print(f"✓ Pakistani: {train_gen.samples} train, {val_gen.samples} val, {num_classes} classes")
    print(f"  Random guess: {100.0/num_classes:.2f}%\n")

    # Extract feature extractor from GTSRB model
    feature_extractor = keras.Sequential(gtsrb_model.layers[:-2])
    feature_extractor.trainable = False

    for layer in feature_extractor.layers:
        layer.trainable = False

    pakistani_model = keras.Sequential([
        feature_extractor,
        layers.Dropout(0.4),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])

    print(f"✓ Feature extractor frozen: {not feature_extractor.trainable}")
    print(f"✓ Total trainable params: {pakistani_model.count_params()}\n")

    pakistani_model.compile(
        optimizer=keras.optimizers.Adam(0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print("="*70)
    print("PHASE 1: Training classifier head")
    print("="*70 + "\n")

    history1 = pakistani_model.fit(
        train_gen,
        epochs=15,
        validation_data=val_gen,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5),
            keras.callbacks.ModelCheckpoint('pakistani_best.keras',
                                          save_best_only=True, monitor='val_accuracy')
        ],
        verbose=1
    )

    print("\n" + "="*70)
    print("PHASE 2: Fine-tuning entire model")
    print("="*70 + "\n")

    feature_extractor.trainable = True

    pakistani_model.compile(
        optimizer=keras.optimizers.Adam(0.00005),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    history2 = pakistani_model.fit(
        train_gen,
        epochs=20,
        validation_data=val_gen,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(patience=4, factor=0.5),
            keras.callbacks.ModelCheckpoint('pakistani_best.keras',
                                          save_best_only=True, monitor='val_accuracy')
        ],
        verbose=1
    )

    class_labels = {v: k for k, v in train_gen.class_indices.items()}
    with open('class_labels.json', 'w') as f:
        json.dump(class_labels, f, indent=2)

    best_acc = max(history2.history['val_accuracy'])

    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"\nValidation Accuracy: {best_acc*100:.1f}%")
    print(f"Random Guess: {100.0/num_classes:.2f}%")
    print(f"Improvement: {(best_acc - 1.0/num_classes)*100:.1f}% above random\n")
    print(f"Saved: pakistani_best.keras, class_labels.json")

    return pakistani_model, class_labels


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    print("\n🚀 Starting OPTIMIZED training pipeline...")
    print("Estimated time: 1-2 hours total\n")

    PAKISTANI_DATASET_PATH = '/content/Pakistan/BalancedData'
    GTSRB_PATH = '/content/GTSRB/archive'
    ORGANIZED_PATH = 'OrganizedDataset'

    print(f"Paths:")
    print(f"  Pakistani dataset: {PAKISTANI_DATASET_PATH}")
    print(f"  GTSRB dataset: {GTSRB_PATH}\n")

    if not os.path.exists(PAKISTANI_DATASET_PATH):
        print(f"Pakistani dataset not found at {PAKISTANI_DATASET_PATH}")
        return

    if not os.path.exists(GTSRB_PATH):
        print("Downloading GTSRB...")
        os.system('pip install -q gdown')
        os.system('gdown --id 1nKj8K6SXTt8aYD_SN0xSYmr00Z5xHETv -O gtsrb.zip')
        os.system(f'unzip -q gtsrb.zip -d {GTSRB_PATH}')

    if not organize_pakistani_dataset(PAKISTANI_DATASET_PATH, ORGANIZED_PATH):
        print("Failed to organize dataset")
        return

    gtsrb_model = train_gtsrb_model(GTSRB_PATH)
    if gtsrb_model is None:
        print("Failed to train GTSRB model")
        return

    pakistani_model, labels = transfer_to_pakistani(gtsrb_model, ORGANIZED_PATH)

    print("\n" + "="*70)
    print("ALL DONE!")
    print("="*70)


if __name__ == "__main__":
    main()