import numpy as np
import cv2
import os
from tqdm import tqdm
from keras.models import load_model
from keras.applications.vgg16 import preprocess_input
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

def create_evaluation_directory():
    """Create a directory to store evaluation results"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_dir = f'evaluation_results/evaluation_{timestamp}'
    os.makedirs(eval_dir, exist_ok=True)
    return eval_dir

def load_test_cases(dir_path, img_size=(224, 224)):
    """Load and preprocess test case images"""
    X = []
    y = []
    file_names = []
    i = 0
    labels = dict()
    
    print(f"\nProcessing images from {dir_path}...")
    for path in tqdm(sorted(os.listdir(dir_path))):
        if not path.startswith('.'):
            labels[i] = path
            for file in os.listdir(os.path.join(dir_path, path)):
                if not file.startswith('.'):
                    img_path = os.path.join(dir_path, path, file)
                    img = cv2.imread(img_path)
                    if img is not None:
                        img = cv2.resize(img, img_size)
                        X.append(img)
                        y.append(i)
                        file_names.append(file)
            i += 1
    
    X = np.array(X)
    y = np.array(y)
    print(f'{len(X)} test case images loaded.')
    return X, y, labels, file_names

def preprocess_imgs(set_name, img_size):
    """Resize and apply VGG-16 preprocessing"""
    set_new = []
    for img in set_name:
        img = cv2.resize(img, dsize=img_size, interpolation=cv2.INTER_CUBIC)
        set_new.append(preprocess_input(img))
    return np.array(set_new)

def evaluate_model(model, X, y_true, labels, file_names, eval_dir):
    """Evaluate model and generate comprehensive reports"""
    # Get predictions
    print("\nMaking predictions...")
    y_pred = model.predict(X)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = y_true

    # Calculate metrics
    accuracy = accuracy_score(y_true_classes, y_pred_classes)
    precision = precision_score(y_true_classes, y_pred_classes, average='binary')
    recall = recall_score(y_true_classes, y_pred_classes, average='binary')
    f1 = f1_score(y_true_classes, y_pred_classes, average='binary')

    # Print metrics
    print("\nModel Performance Metrics on Test Cases:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

    # Generate detailed classification report
    print("\nClassification Report:")
    report = classification_report(y_true_classes, y_pred_classes, 
                                 target_names=['No Tumor', 'Tumor'],
                                 output_dict=True)
    print(classification_report(y_true_classes, y_pred_classes, 
                              target_names=['No Tumor', 'Tumor']))

    # Create confusion matrix
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Tumor', 'Tumor'],
                yticklabels=['No Tumor', 'Tumor'])
    plt.title('Confusion Matrix (Test Cases)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(eval_dir, 'confusion_matrix_test_cases.png'))
    plt.close()

    # Create detailed results DataFrame
    results_df = pd.DataFrame({
        'File Name': file_names,
        'True Label': [labels[y] for y in y_true_classes],
        'Predicted Label': [labels[y] for y in y_pred_classes],
        'Confidence': np.max(y_pred, axis=1),
        'Correct': y_true_classes == y_pred_classes
    })

    # Save results to CSV
    results_df.to_csv(os.path.join(eval_dir, 'detailed_results.csv'), index=False)

    # Save metrics to text file
    with open(os.path.join(eval_dir, 'evaluation_metrics.txt'), 'w') as f:
        f.write("Model Evaluation Metrics\n")
        f.write("======================\n\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1-Score: {f1:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(y_true_classes, y_pred_classes, 
                                    target_names=['No Tumor', 'Tumor']))

    # Print incorrect predictions
    incorrect_predictions = results_df[~results_df['Correct']]
    if not incorrect_predictions.empty:
        print("\nIncorrect Predictions:")
        print(incorrect_predictions[['File Name', 'True Label', 'Predicted Label', 'Confidence']])

    return results_df, report

def main():
    # Create evaluation directory
    eval_dir = create_evaluation_directory()
    print(f"Evaluation results will be saved in: {eval_dir}")

    # Load the model
    print("\nLoading model...")
    model = load_model('brain_tumor_model.h5')

    # Load and preprocess test cases
    TEST_CASES_DIR = 'Test Case image/'
    IMG_SIZE = (224, 224)

    X_test, y_test, labels, file_names = load_test_cases(TEST_CASES_DIR, IMG_SIZE)
    X_test = preprocess_imgs(X_test, IMG_SIZE)

    # Evaluate model
    results_df, report = evaluate_model(model, X_test, y_test, labels, file_names, eval_dir)

    print(f"\nEvaluation complete! Results saved in: {eval_dir}")
    print("\nFiles generated:")
    print(f"1. confusion_matrix_test_cases.png - Confusion matrix visualization")
    print(f"2. detailed_results.csv - Detailed results for each image")
    print(f"3. evaluation_metrics.txt - Complete evaluation metrics and report")

if __name__ == "__main__":
    main() 