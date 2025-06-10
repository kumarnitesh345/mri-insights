import shutil
import os

# List of folders/files to remove
to_remove = [
    'venv', 'TRAIN', 'VAL', 'TEST', 'Test Case image', 'yes', 'no', '.git',
    'confusion_matrix.png', 'confusion_matrix_evaluation.png', 'confusion_matrix_validation.png',
    'training_history.png'
]

for item in to_remove:
    if os.path.exists(item):
        if os.path.isdir(item):
            shutil.rmtree(item)
            print(f"Deleted folder: {item}")
        else:
            os.remove(item)
            print(f"Deleted file: {item}")