import numpy as np
import cv2
import os
from tqdm import tqdm
from keras.models import load_model
from keras.applications.vgg16 import preprocess_input
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

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
    """Resize and apply VGG-16 preprocessing"""
    set_new = []
    for img in set_name:
        img = cv2.resize(img, dsize=img_size, interpolation=cv2.INTER_CUBIC)
        set_new.append(preprocess_input(img))
    return np.array(set_new)

# Load the saved model
print("Loading model...")
model = load_model('brain_tumor_model.h5')

# Load and preprocess validation data
VAL_DIR = 'VAL/'
IMG_SIZE = (224, 224)

print("Loading validation data...")
X_val, y_val, labels = load_data(VAL_DIR, IMG_SIZE)

# Preprocess the images
print("Preprocessing images...")
X_val = preprocess_imgs(X_val, IMG_SIZE)

# Get predictions
print("Making predictions...")
y_pred = model.predict(X_val)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = y_val  # Already in the correct format

# Calculate metrics
accuracy = accuracy_score(y_true_classes, y_pred_classes)
precision = precision_score(y_true_classes, y_pred_classes, average='binary')
recall = recall_score(y_true_classes, y_pred_classes, average='binary')
f1 = f1_score(y_true_classes, y_pred_classes, average='binary')

# Print metrics
print("\nModel Performance Metrics on Validation Set:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# Print detailed classification report
print("\nClassification Report:")
print(classification_report(y_true_classes, y_pred_classes, target_names=['No Tumor', 'Tumor']))

# Create and plot confusion matrix
cm = confusion_matrix(y_true_classes, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Tumor', 'Tumor'],
            yticklabels=['No Tumor', 'Tumor'])
plt.title('Confusion Matrix (Validation Set)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix_validation.png')
plt.close()

print("\nConfusion matrix has been saved as 'confusion_matrix_validation.png'") 