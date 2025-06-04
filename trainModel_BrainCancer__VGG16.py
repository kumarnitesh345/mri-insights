import numpy as np
from tqdm import tqdm
import cv2
import os
import shutil
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns

from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16, preprocess_input
from keras import layers
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.utils import to_categorical

# Create necessary directories
os.makedirs('TRAIN/YES', exist_ok=True)
os.makedirs('TRAIN/NO', exist_ok=True)
os.makedirs('TEST/YES', exist_ok=True)
os.makedirs('TEST/NO', exist_ok=True)
os.makedirs('VAL/YES', exist_ok=True)
os.makedirs('VAL/NO', exist_ok=True)

# Use the current directory's yes/no folders
IMG_PATH = './'
# split the data by train/val/test
for CLASS in ['yes', 'no']:
    if os.path.exists(os.path.join(IMG_PATH, CLASS)):
        IMG_NUM = len(os.listdir(os.path.join(IMG_PATH, CLASS)))
        print(f"Processing {CLASS}: {IMG_NUM} images")
        for (n, FILE_NAME) in enumerate(os.listdir(os.path.join(IMG_PATH, CLASS))):
            img = os.path.join(IMG_PATH, CLASS, FILE_NAME)
            if n < 5:
                shutil.copy(img, f'TEST/{CLASS.upper()}/{FILE_NAME}')
            elif n < 0.8*IMG_NUM:
                shutil.copy(img, f'TRAIN/{CLASS.upper()}/{FILE_NAME}')
            else:
                shutil.copy(img, f'VAL/{CLASS.upper()}/{FILE_NAME}')

def load_data(dir_path, img_size=(224, 224)):
    """Load resized images as np.arrays to workspace"""
    X = []
    y = []
    i = 0
    labels = dict()
    for path in tqdm(sorted(os.listdir(dir_path))):
        if not path.startswith('.'):
            labels[i] = path
            for file in os.listdir(os.path.join(dir_path, path)):
                if not file.startswith('.'):
                    img = cv2.imread(os.path.join(dir_path, path, file))
                    if img is not None:
                        img = cv2.resize(img, img_size)
                        X.append(img)
                        y.append(i)
            i += 1
    X = np.array(X)
    y = np.array(y)
    print(f'{len(X)} images loaded from {dir_path} directory.')
    return X, y, labels

def preprocess_imgs(set_name, img_size):
    """
    Resize and apply VGG-16 preprocessing
    """
    set_new = []
    for img in set_name:
        img = cv2.resize(img, dsize=img_size, interpolation=cv2.INTER_CUBIC)
        set_new.append(preprocess_input(img))
    return np.array(set_new)

# Load and preprocess the data
TRAIN_DIR = 'TRAIN/'
TEST_DIR = 'TEST/'
VAL_DIR = 'VAL/'
IMG_SIZE = (224, 224)

print("Loading training data...")
X_train, y_train, labels = load_data(TRAIN_DIR, IMG_SIZE)
print("Loading test data...")
X_test, y_test, _ = load_data(TEST_DIR, IMG_SIZE)
print("Loading validation data...")
X_val, y_val, _ = load_data(VAL_DIR, IMG_SIZE)

# Convert labels to categorical
y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)
y_val = to_categorical(y_val, num_classes=2)

# Preprocess the images
print("Preprocessing images...")
X_train = preprocess_imgs(X_train, IMG_SIZE)
X_test = preprocess_imgs(X_test, IMG_SIZE)
X_val = preprocess_imgs(X_val, IMG_SIZE)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Create and train the model
print("Creating model...")
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add custom layers
x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(2048, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(1024, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(512, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)
predictions = layers.Dense(2, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True)

# Train the model
print("Training model...")
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    validation_data=(X_val, y_val),
    epochs=50,
    callbacks=[early_stopping, reduce_lr, checkpoint]
)

# Evaluate the model
print("Evaluating model...")
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")

# Get predictions
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Print classification report
print("\nClassification Report:")
print(classification_report(y_true_classes, y_pred_classes, target_names=['No Tumor', 'Tumor']))

# Plot confusion matrix
cm = confusion_matrix(y_true_classes, y_pred_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('confusion_matrix.png')
plt.close()

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('training_history.png')
plt.close()

# Save the model
print("Saving model...")
model.save('brain_tumor_model.h5')
print("Model saved as 'brain_tumor_model.h5'")
