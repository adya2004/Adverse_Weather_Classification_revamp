import os
# Set environment variables before importing TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TF logging
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use only first GPU if available

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io
from typing import List

# Configure TensorFlow GPU memory growth to prevent conflicts
try:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
except RuntimeError as e:
    print(f"GPU configuration error: {e}")

# Initialize FastAPI app
app = FastAPI(title="Image Classification API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variable to store the model
model = None

# Load the model when the app starts
@app.on_event("startup")
async def load_model():
    global model
    try:
        # Additional TensorFlow configuration
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
        
        # Check if model file exists
        model_path = "my_model.keras"
        if not os.path.exists(model_path):
            print(f"Model file {model_path} not found!")
            raise FileNotFoundError(f"Model file {model_path} not found")
        
        model = tf.keras.models.load_model(model_path, compile=False)
        
        # Compile the model after loading to avoid issues
        model.compile()
        
        print("Model loaded successfully!")
        print(f"Model input shape: {model.input_shape}")
        print(f"Model output shape: {model.output_shape}")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise e

def preprocess_image(image: Image.Image, target_size: tuple = (224, 224)) -> np.ndarray:
    """
    Minimal preprocessing - just resize and add batch dimension
    
    Args:
        image: PIL Image object
        target_size: Target size for resizing (width, height)
    
    Returns:
        Preprocessed image array
    """
    # Only resize the image
    image = image.resize(target_size)
    
    # Convert to numpy array
    img_array = np.array(image)
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    # Keep original pixel values (0-255) - no normalization
    return img_array

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Image Classification API is running!"}

@app.get("/model/info")
async def model_info():
    """Get information about the loaded model"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    return {
        "model_loaded": True,
        "input_shape": model.input_shape,
        "output_shape": model.output_shape,
        "total_params": model.count_params()
    }

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """
    Predict the class of an uploaded image
    
    Args:
        file: Uploaded image file
    
    Returns:
        JSON response with predicted class and confidence scores
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image file
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Get target size from model input shape (assuming NHWC format)
        if len(model.input_shape) == 4:  # (batch, height, width, channels)
            target_size = (model.input_shape[2], model.input_shape[1])  # (width, height)
        else:
            target_size = (224, 224)  # default size
        
        # Preprocess image
        processed_image = preprocess_image(image, target_size)
        
        # Make prediction
        predictions = model.predict(processed_image)
        
        # Get the predicted class (highest probability)
        predicted_class = int(np.argmax(predictions[0]))

        mp = {
            0:"Cloudy",
            1:"Fog",
            2:"Rainy",
            3:"Sand",
            4:"Shine",
            5:"Snow",
            6:"Sunrise"
        }

        descriptions = {
            0:"Maintain normal driving parameters but increase sensor sensitivity and reduce following distances slightly. Monitor for rapid weather changes and ensure all lighting systems are functional for potential visibility reduction.",
            1:" Immediately reduce speed by 30-50% and increase following distance to 5-8 seconds. Activate fog lights, rely heavily on LiDAR sensors, and consider route diversion to avoid areas with known fog accumulation.",
            2:"Reduce speed by 20-30% and increase following distance to 4-6 seconds. Activate windshield wipers, monitor tire traction continuously, and avoid sudden acceleration or braking maneuvers to prevent hydroplaning.",
            3:"Slow down significantly and increase following distance to avoid dust clouds from other vehicles. Close air intake vents, rely on sealed sensors, and consider stopping safely if visibility drops below safe thresholds.",
            4:"Activate automatic headlight dimming and sun visors, monitor camera sensors for glare interference. Adjust route timing to minimize direct sun exposure during critical driving phases like lane changes or turns.",
            5:"Reduce speed by 40-60% and increase following distance to 8-10 seconds. Engage traction control systems, monitor tire grip continuously, and prioritize cleared roadways while avoiding sudden steering inputs.",
            6:"Adjust camera exposure settings and activate anti-glare protocols. Monitor for sun glare interference with sensors and reduce speed when driving directly toward sunrise until lighting conditions stabilize."
        }

        prediction = mp[predicted_class]
        description = descriptions[predicted_class]

        confidence = float(np.max(predictions[0]))
        
        # Get all class probabilities
        class_probabilities = predictions[0].tolist()
        
        return JSONResponse(content={
            "prediction": prediction,
            "descripton": description,
            "predicted_class":predicted_class,
            "confidence": confidence,
            "all_probabilities": class_probabilities,
            "image_shape": processed_image.shape[1:],
            "filename": file.filename
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)