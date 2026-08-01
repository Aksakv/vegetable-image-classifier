import tensorflow as tf
import cv2
import numpy as  np
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model

from tkinter import Tk
from tkinter import filedialog

model = load_model("vegetable_classifier.keras")
print("model sucessful")
# image = cv2.imread("veg.jpg")
root =Tk()
root.withdraw()

image = filedialog.askopenfilename(title="select a veg image",filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
print(image)
image = cv2.imread(image)
plt.show()
image = cv2.resize(image, (224, 224))
image = image.astype(np.float32) / 255.0
image = np.expand_dims(image, axis=0)
print(image.shape)
prediction = model.predict(image)
# print(prediction)
predicted_index = np.argmax(prediction)
confidence = np.max(prediction)*100
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
plt.title(f"Predicted Vegetable: {class_names[predicted_index]}\n confidence : {confidence:.2f}%")
plt.imshow(cv2.cvtColor(image[0], cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()