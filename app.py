from flask import Flask,render_template,request
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model


model = load_model("vegetable_classifier.keras")
class_names = [
    "Bean",
    "Bitter_Gourd",
    "Bottle_Gourd",
    "Brinjal",
    "Broccoli",
    "Cabbage",
    "Capsicum",
    "Carrot",
    "Cauliflower",
    "Cucumber",
    "Papaya",
    "Potato",
    "Pumpkin",
    "Radish",
    "Tomato"
]
app = Flask(__name__)
@app.route("/")
def home():
    return  render_template("index.html")
@app.route("/predict", methods=["POST"])
def predict():
    image = request.files["image"]
    image_path = os.path.join(
    "static",
    "uploads",
    image.filename
)

    image.save(image_path)
    read_image = cv2.imread(image_path)
    prediction_image = cv2.resize(read_image, (224, 224))
    prediction_image = prediction_image.astype(np.float32) / 255.0
    prediction_image = np.expand_dims(prediction_image, axis=0)
    prediction = model.predict(prediction_image)
    predicted_index = np.argmax(prediction)
    predicted_class = class_names[predicted_index]
    confidence = prediction[0][predicted_index] * 100
    
    return render_template(
    "index.html",
    prediction=predicted_class,
    confidence=round(confidence, 2),image_file = image.filename)

    

if __name__ == "__main__":   
    app.run(debug=True)