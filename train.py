import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
# from tensorflow.keras.models import load_model

dataset_folder = "dataset"


# model = load_model("vegetable_classifier.keras")

main_folder = os.listdir(dataset_folder)
main_folder_path = os.path.join(dataset_folder,main_folder[0])
main_folder_sub_div = os.listdir(main_folder_path)
classes_path = os.path.join(main_folder_path,main_folder_sub_div[1])
class_image =os.listdir(classes_path)
print(class_image)
# print(classes_path)
stored_image =[]
labels =[]


for label,class_name in  enumerate(class_image):
    
    
    bean = os.path.join(classes_path,class_name)
    print(bean)
    bean_files = os.listdir(bean)
    # print(bean_files)
    for image_name in  bean_files:
        # print(image_name)
        image_path = os.path.join(bean,image_name)
        # print(image_path)
    
        read_image = cv2.imread(image_path)
# print(read_image.shape,read_image.size)
        resized_img = cv2.resize(read_image,(224,224))
    # color = cv2.cvtColor(resized_img,cv2.COLOR_BGR2RGB)
        stored_image.append(resized_img)
        labels.append(label)
stored_image= np.array(stored_image)
labels = np.array(labels)

print(stored_image.shape)
print(labels.shape)

plt.imshow(cv2.cvtColor(stored_image[0], cv2.COLOR_BGR2RGB))
plt.show()
# ==========================
# Validation Dataset
# ==========================

validation_images =[]
validation_labels = []
validation_path = os.path.join(main_folder_path,main_folder_sub_div[2])
print(validation_path)
validation_class = os.listdir(validation_path)
print(validation_class)

for label, class_name in enumerate(validation_class):
    class_path_val = os.path.join(validation_path,class_name)
    # print(class_path_val)
    full_validatio_path = os.listdir(class_path_val)
    # print(full_validatio_path)
    for image_name in full_validatio_path:
        image = os.path.join(class_path_val,image_name)
        # print(image)
        read_image_vali = cv2.imread(image)
        read_image_vali = cv2.resize(read_image_vali,(224,224))
        validation_images.append(read_image_vali)
        validation_labels.append(label)
validation_images = np.array(validation_images)
validation_labels = np.array(validation_labels)
print(validation_images.shape)
print(validation_labels.shape)

test_image =[]
test_label = []

test_path = os.path.join(main_folder_path,main_folder_sub_div[0])
test_list = os.listdir(test_path)
# print(test_list)

for label,image in enumerate(test_list):
    test_detail = os.path.join(test_path,image)
    test_detail_list = os.listdir(test_detail)
    # print(test_detail_list)
    for test_image_list in test_detail_list:
        tests_image_path = os.path.join(test_detail,test_image_list)
    # print(tests_image_path)
        read_image_test = cv2.imread(tests_image_path)
        read_image_test =  cv2.resize(read_image_test,(224,224))
        test_image.append(read_image_test)
        test_label.append(label)
test_image = np.array(test_image)
test_label =np.array(test_label)
print(test_label.shape)
print(test_image.shape)

stored_image = stored_image .astype(np.float32)/255.0

validation_images = validation_images.astype(np.float32)/255.0

test_image = test_image.astype(np.float32)/255.0
print(stored_image.min(),stored_image.max())
print(validation_images.min(),validation_images.max())
print(test_image.min(),test_image.max())


print(tf.__version__)
model = Sequential()
model.add(Input(shape=(224, 224, 3)))
model.add(Conv2D(filters=32,kernel_size=(3,3),activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(filters=64,kernel_size=(3,3),activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(128,(3,3),activation="relu"))
model.add(MaxPooling2D((2,2)))
model.add(Flatten())
# model.add(Dense(128,activation="relu"))
model.add(Dense(256,activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(15,activation="softmax"))
model.summary()
model.compile(optimizer = "adam",loss = "sparse_categorical_crossentropy",metrics = ["accuracy"])
history = model.fit(stored_image,labels,validation_data =(validation_images,validation_labels),epochs = 50,batch_size =32)


test_loss,test_accuracy = model.evaluate(test_image,test_label)
print("test loss",test_loss)
print("test_accuracy",test_accuracy)
model.save("vegetable_classifier.keras")
