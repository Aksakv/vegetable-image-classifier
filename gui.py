import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import ImageTk,Image

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
# print("Model Loaded Successfully")
window =tk.Tk()
window.title("🥕 Vegetable Image Classifier")
window.geometry("800x700")
window.configure(bg="#f5f5f5")
def open_image():
    
    file_path = filedialog.askopenfilename(
        title="Select a Vegetable Image",
        filetypes=[
            ("Image Files", "*.jpg *.jpeg *.png")
        ]
    )
    image = cv2.imread(file_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    pil_image = pil_image.resize((300, 300))
    tk_image = ImageTk.PhotoImage(pil_image)
    image_label.config(image=tk_image)
    image_label.config(text="")
    image_label.image = tk_image
    
    prediction_image = cv2.resize(image, (224, 224))

    prediction_image = prediction_image.astype(np.float32) / 255.0

    prediction_image = np.expand_dims(prediction_image, axis=0)
        
    prediction = model.predict(prediction_image)
    predicted_index = np.argmax(prediction)
   
    # print(class_names[predicted_index])
    confidence = prediction[0][predicted_index] * 100
    # print(f"Confidence: {confidence:.2f}%")
    result_label.config(
    text=f"Predicted: {class_names[predicted_index]}\nConfidence: {confidence:.2f}%"
)
    
heading = tk.Label(window,text="🥗 Vegetable Image Classifier", font=("Times New Roman", 26, "bold"),fg="darkgreen",
    bg="#F5F5F5")
heading.pack(padx=20)
button = tk.Button(window,
                text="📁 Select Image",
                font=("Arial", 14),
                command=open_image,
                bg="#B8B7D0",
                fg="white",
                padx=20,
                pady=10,
                cursor="hand2")
button.pack(pady=20)
image_frame = tk.Frame(
                window,
                width=320,
                height=320,
                bg="white",
                bd=2,
                relief="solid"
            )
image_frame.pack(pady=20)
image_frame.pack_propagate(False)
image_label = tk.Label(
    image_frame ,
    text="📷\n\nNo Image Selected",
    font=("Arial",16),
    bg="white",
    fg="gray",
    justify="center")
image_label.pack(expand=True)
result_label = tk.Label(
    window,
    text="",
    font=("Arial", 16, "bold"),
    fg="blue"
)

result_label.pack(pady=20)
# image_frame.pack_propagate(False)
window.mainloop()


