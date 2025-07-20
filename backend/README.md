# Adverse Weather Classification API

A FastAPI-based image classification API for detecting weather conditions in images.

## Features

- Image classification for weather conditions: Cloudy, Fog, Rainy, Sand, Shine, Snow, Sunrise
- Single image and batch prediction endpoints
- TensorFlow/Keras model integration
- CORS support for web applications

## API Endpoints

- `GET /` - Health check
- `GET /model/info` - Get model information
- `POST /predict` - Predict single image
- `POST /predict/batch` - Predict multiple images (max 10)

## Local Development

1. Create virtual environment:
```bash
python -m venv myenv
source myenv/bin/activate  # On Windows: myenv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
uvicorn main:app --reload
```

## Deployment on Render

### Method 1: Using render.yaml (Recommended)

1. Push your code to GitHub
2. Connect your GitHub repository to Render
3. The `render.yaml` file will automatically configure your deployment

### Method 2: Manual Setup

1. Create a new Web Service on Render
2. Connect your GitHub repository
3. Use the following settings:
   - **Environment**: Python
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Python Version**: 3.12.0

### Environment Variables

Set the following environment variables in Render:
- `PYTHON_VERSION`: 3.12.0
- `TF_CPP_MIN_LOG_LEVEL`: 2
- `CUDA_VISIBLE_DEVICES`: "" (empty string to disable GPU)

## Model Requirements

Ensure your `my_model.keras` file is included in your repository. The model should:
- Accept input shape compatible with (224, 224, 3) images
- Output 7 classes for weather conditions

## Usage Example

```python
import requests

# Single prediction
with open('image.jpg', 'rb') as f:
    response = requests.post('https://your-app.onrender.com/predict', 
                           files={'file': f})
    print(response.json())
```

## File Structure

```
├── main.py              # FastAPI application
├── my_model.keras       # Trained TensorFlow model
├── requirements.txt     # Python dependencies
├── render.yaml          # Render deployment configuration
├── Dockerfile           # Docker configuration (optional)
├── .dockerignore        # Docker ignore file
└── sample_test_images/  # Test images
```
