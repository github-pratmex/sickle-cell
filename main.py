from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
from fastapi.middleware.cors import CORSMiddleware

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress info and warning messages from TensorFlow

# Force TensorFlow to use CPU (if a GPU is available, it will be ignored)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # This will disable GPU

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to your frontend URL for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).resize((224, 224))
    image = np.array(image) / 255.0
    if image.ndim == 2:
        image = np.stack((image,) * 3, axis=-1)  # Grayscale to RGB
    image = np.expand_dims(image.astype(np.float32), axis=0)

    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])

    result = "positive" if prediction[0][0] > 0.5 else "negative"
    return {"result": result, "confidence": float(prediction[0][0])}
