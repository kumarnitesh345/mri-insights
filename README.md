# Brain Tumor Detection Web Application

This is a web-based application for detecting brain tumors in MRI images using deep learning. The application uses a pre-trained VGG16-based CNN model to classify brain MRI images as either containing a tumor or not.

## Features

- Modern, responsive web interface
- Drag-and-drop image upload
- Real-time image preview
- Instant tumor detection results with confidence scores
- Support for JPG, JPEG, and PNG image formats

## Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd <repository-directory>
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Make sure you have the pre-trained model file (`brain_tumor_model.h5`) in the root directory.

2. Start the Flask application:
```bash
python app.py
```

3. Open your web browser and navigate to:
```
http://localhost:5000
```

4. Upload a brain MRI image by either:
   - Clicking the upload area and selecting a file
   - Dragging and dropping an image file onto the upload area

5. Wait for the analysis to complete. The result will show:
   - The uploaded image
   - Whether a tumor was detected
   - The confidence score of the prediction

## Model Information

The application uses a VGG16-based CNN model that has been trained on a dataset of brain MRI images. The model performs the following preprocessing steps:

1. Image resizing to 224x224 pixels
2. Grayscale conversion
3. Gaussian blur application
4. Brain region extraction using contour detection
5. Image normalization

## Directory Structure

```
.
├── app.py              # Flask application
├── requirements.txt    # Python dependencies
├── brain_tumor_model.h5 # Pre-trained model
├── static/
│   └── uploads/       # Uploaded images storage
└── templates/
    └── index.html     # Web interface
```

## Notes

- The application is for educational and research purposes only
- Always consult medical professionals for actual medical diagnosis
- The model's predictions should not be used as a substitute for professional medical advice

## License

This project is licensed under the MIT License - see the LICENSE file for details. 