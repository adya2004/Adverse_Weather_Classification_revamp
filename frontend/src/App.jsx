import React, { useState } from "react";
import { Upload, Cloud, Eye, Loader } from "lucide-react";

export default function WeatherClassifier() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleImageChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedImage(file);
      const reader = new FileReader();
      reader.onload = (e) => {
        setImagePreview(e.target.result);
      };
      reader.readAsDataURL(file);
      setResult(null);
      setError(null);
    }
  };

  const handleSubmit = async () => {
    if (!selectedImage) {
      setError("Please select an image first");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append("file", selectedImage);

      // Replace this URL with your actual AI backend endpoint
      const response = await fetch(
        "https://2f7570c146c0.ngrok-free.app/predict",
        {
          method: "POST",
          body: formData,
        }
      );

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(`Classification failed: ${err.message}`);
      // For demo purposes, show a mock response
      setResult({
        weather_condition: "Sunny",
        confidence: 0.87,
        description: "Clear sky with good visibility",
        vehicle_recommendation:
          "Normal driving conditions. No special precautions needed.",
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-cyan-50">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="text-center mb-12">
          <div className="flex items-center justify-center mb-4">
            <Cloud className="w-12 h-12 text-blue-600 mr-3" />
            <h1 className="text-4xl font-bold text-gray-800">
              Weather Classification System
            </h1>
          </div>
          <p className="text-lg text-gray-600 max-w-3xl mx-auto">
            Empowering autonomous vehicles with intelligent weather recognition.
            Upload an image to classify weather conditions and receive
            navigation recommendations for safer, adaptive driving in any
            environment.
          </p>
        </div>

        <div className="max-w-4xl mx-auto">
          <div className="grid md:grid-cols-2 gap-8">
            {/* Input Section */}
            <div className="bg-white rounded-xl shadow-lg p-6">
              <h2 className="text-2xl font-semibold text-gray-800 mb-6 flex items-center">
                <Upload className="w-6 h-6 mr-2 text-blue-600" />
                Image Input
              </h2>

              <div className="space-y-6">
                {/* File Upload */}
                <div className="border-2 border-dashed border-blue-300 rounded-lg p-8 text-center hover:border-blue-400 transition-colors">
                  <input
                    type="file"
                    accept="image/*"
                    onChange={handleImageChange}
                    className="hidden"
                    id="image-upload"
                  />
                  <label htmlFor="image-upload" className="cursor-pointer">
                    <Upload className="w-12 h-12 text-blue-400 mx-auto mb-4" />
                    <p className="text-gray-600">
                      Click to upload or drag and drop an image
                    </p>
                    <p className="text-sm text-gray-400 mt-2">
                      Supports JPG, PNG, GIF formats
                    </p>
                  </label>
                </div>

                {/* Image Preview */}
                {imagePreview && (
                  <div className="border rounded-lg p-4">
                    <h3 className="font-medium text-gray-700 mb-2">Preview:</h3>
                    <img
                      src={imagePreview}
                      alt="Preview"
                      className="w-full h-48 object-cover rounded-lg"
                    />
                  </div>
                )}

                {/* Submit Button */}
                <button
                  onClick={handleSubmit}
                  disabled={!selectedImage || loading}
                  className="w-full bg-blue-600 text-white py-3 px-6 rounded-lg font-semibold 
                           hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed 
                           transition-colors flex items-center justify-center"
                >
                  {loading ? (
                    <>
                      <Loader className="w-5 h-5 mr-2 animate-spin" />
                      Classifying...
                    </>
                  ) : (
                    <>
                      <Eye className="w-5 h-5 mr-2" />
                      Classify Weather
                    </>
                  )}
                </button>
              </div>
            </div>

            {/* Output Section */}
            <div className="bg-white rounded-xl shadow-lg p-6">
              <h2 className="text-2xl font-semibold text-gray-800 mb-6 flex items-center">
                <Cloud className="w-6 h-6 mr-2 text-green-600" />
                Classification Results
              </h2>

              <div className="space-y-4">
                {error && (
                  <div className="bg-red-50 border border-red-200 rounded-lg p-4">
                    <p className="text-red-800">{error}</p>
                  </div>
                )}

                {result ? (
                  <div className="space-y-4">
                    <div className="bg-green-50 border border-green-200 rounded-lg p-4">
                      <h3 className="font-semibold text-green-800 text-lg mb-2">
                        Weather Detected: {result.prediction}
                      </h3>
                      <div className="space-y-2 text-sm">
                        <p>
                          <span className="font-medium">Confidence:</span>{" "}
                          {(result.confidence * 100).toFixed(1)}%
                        </p>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="text-center py-12">
                    <Cloud className="w-16 h-16 text-gray-300 mx-auto mb-4" />
                    <p className="text-gray-500">
                      Upload an image to see weather classification results
                    </p>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="text-center mt-12 text-gray-500">
          <p>Designed for autonomous vehicle weather adaptation and safety</p>
        </div>
      </div>
    </div>
  );
}
