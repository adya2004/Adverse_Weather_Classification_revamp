# Render Deployment Guide

## Files Created for Render Deployment

1. **render.yaml** - Infrastructure as Code configuration
2. **Dockerfile** - Container configuration (optional)
3. **Procfile** - Process file for deployment
4. **runtime.txt** - Python version specification
5. **README.md** - Documentation
6. **.dockerignore** - Docker ignore patterns
7. **Updated .gitignore** - Git ignore patterns
8. **Updated main.py** - Port configuration for Render

## Deployment Steps

### Option 1: Automatic Deployment (Recommended)

1. **Push to GitHub:**
   ```bash
   git add .
   git commit -m "Add Render deployment configuration"
   git push origin main
   ```

2. **Connect to Render:**
   - Go to [render.com](https://render.com)
   - Sign up/Login with GitHub
   - Click "New" → "Blueprint"
   - Connect your GitHub repository
   - Select this repository
   - Render will automatically detect `render.yaml` and deploy

### Option 2: Manual Web Service

1. **Create Web Service:**
   - Go to Render Dashboard
   - Click "New" → "Web Service"
   - Connect your GitHub repository

2. **Configure Settings:**
   - **Name:** adverse-weather-classifier
   - **Environment:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`

3. **Environment Variables:**
   - `PYTHON_VERSION`: 3.12.0
   - `TF_CPP_MIN_LOG_LEVEL`: 2
   - `CUDA_VISIBLE_DEVICES`: "" (empty)

## Important Notes

- **Model File:** Ensure `my_model.keras` is in your repository (not in .gitignore)
- **Memory:** The free tier has limited memory. TensorFlow models can be memory-intensive
- **GPU:** Disabled GPU usage for Render deployment (CPU only)
- **Cold Starts:** First request may take longer due to model loading

## Testing After Deployment

Your API will be available at: `https://your-service-name.onrender.com`

Test endpoints:
- Health check: `GET /`
- Model info: `GET /model/info`
- Prediction: `POST /predict` (with image file)

## Troubleshooting

- **Model loading errors:** Check if `my_model.keras` is included in the repository
- **Memory issues:** Consider using a smaller model or upgrading to a paid plan
- **Timeout issues:** Optimize model loading and inference time

## Next Steps

1. Test locally: `uvicorn main:app --reload`
2. Commit and push changes to GitHub
3. Deploy on Render using the blueprint method
4. Test the deployed API endpoints
