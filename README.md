# MediScan AI - Medical Image Classification

MediScan AI is a powerful web application built with FastAPI and TensorFlow that provides automated analysis of medical images. It utilizes deep learning models to assist in diagnosing three distinct medical conditions from MRI scans and Chest X-Rays: Brain Tumors, Tuberculosis, and Pneumonia.

> **Disclaimer:** This application is for educational and demonstrative purposes only. It is not intended to be a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of a qualified health provider with any questions you may have regarding a medical condition.

## Features

*   **Brain Tumor Detection:** Analyzes MRI/CT scans to classify brain conditions into four categories: Glioma, Meningioma, Pituitary Tumor, or No Tumor.
*   **Tuberculosis Detection:** Analyzes Chest X-Rays to identify signs of Tuberculosis infection.
*   **Pneumonia Detection:** Analyzes Chest X-Rays to detect pneumonia patterns.
*   **Interactive Web Interface:** User-friendly frontend for seamless image uploading and result visualization.
*   **Detailed Diagnostic Reports:** Provides prediction confidence scores, severity assessments, medical descriptions, and recommended next steps based on the findings.

## Technologies Used

*   **Backend:** FastAPI, Python
*   **Machine Learning:** TensorFlow, Keras, NumPy
*   **Image Processing:** Pillow (PIL)
*   **Frontend:** HTML, CSS, Jinja2 Templates
*   **Server:** Uvicorn

## Project Structure

```text
.
├── main.py                     # Main FastAPI application script
├── requirements.txt            # Python dependencies
├── braintumor.h5               # Trained model for Brain Tumor detection
├── Tuberculosis_model.h5       # Trained model for TB detection
├── pneumonia_model.h5          # Trained model for Pneumonia detection
├── static/                     # Static assets (CSS, JS, Images)
│   └── css/
│       ├── style.css
│       └── result.css
├── templates/                  # Jinja2 HTML templates
│   ├── index.html              # Home/Upload page
│   └── result.html             # Prediction results page
└── uploads/                    # Temporary storage for uploaded images
```

## Setup & Installation

### Prerequisites

*   Python 3.8+ 
*   pip (Python package installer)

### Installation Steps

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/mediscan-ai.git
    cd mediscan-ai
    ```

2.  **Create a virtual environment (optional but recommended):**
    ```bash
    python -m venv venv
    
    # On Windows
    venv\Scripts\activate
    
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Model Files:**
    Ensure the following pre-trained `.h5` model files are present in the root directory:
    *   `braintumor.h5`
    *   `Tuberculosis_model.h5`
    *   `pneumonia_model.h5`

## Usage

1.  **Start the FastAPI server:**
    ```bash
    python main.py
    # or using uvicorn directly:
    # uvicorn main:app --host 0.0.0.0 --port 8000
    ```

2.  **Access the application:**
    Open your web browser and navigate to:
    ```
    http://localhost:8000
    ```

3.  **Make a prediction:**
    *   Select the type of scan you want to perform (Brain Tumor, Tuberculosis, or Pneumonia).
    *   Upload the corresponding medical image (MRI or Chest X-Ray).
    *   Click "Analyze" to view the diagnostic results.

## API Documentation

FastAPI automatically generates interactive API documentation. Once the server is running, you can explore the endpoints:

*   **Swagger UI:** `http://localhost:8000/docs`
*   **ReDoc:** `http://localhost:8000/redoc`

## How it Works

1.  The user uploads an image via the web interface.
2.  The image is sent to the `/api/predict` endpoint along with the selected diagnostic model.
3.  The backend preprocesses the image (resizing, normalization) to match the input shape expected by the specific model.
4.  The appropriate TensorFlow model processes the image and outputs probability scores.
5.  The system determines the most likely classification, calculates the confidence score, and retrieves relevant medical context.
6.  A JSON response is sent back and rendered on the results page.
