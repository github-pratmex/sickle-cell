from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI()

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
