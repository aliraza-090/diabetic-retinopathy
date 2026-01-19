print("Prediction started...")

import tensorflow as tf
import numpy as np
import cv2
import os

# Paths
MODEL_PATH = "../saved_models/model_trained.h5"
IMAGE_PATH = "../data/test_images"

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Pick one image from test_images
img_name = os.listdir(IMAGE_PATH)[0]
img_path = os.path.join(IMAGE_PATH, img_name)

# Read and preprocess image
img = cv2.imread(img_path)
img = cv2.resize(img, (224, 224))
img = img / 255.0
img = np.expand_dims(img, axis=0)

# Predict
prediction = model.predict(img)
predicted_class = np.argmax(prediction)

print("Image name:", img_name)
print("Predicted disease level:", predicted_class)
