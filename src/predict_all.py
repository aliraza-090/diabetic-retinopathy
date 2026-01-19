print("Batch prediction started...")

import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Paths
MODEL_PATH = "../saved_models/model_trained.h5"
IMG_DIR = "../data/test_images"

# Load trained model
model = load_model(MODEL_PATH)

# Class labels
class_names = {
    0: "No Diabetic Retinopathy",
    1: "Mild Diabetic Retinopathy",
    2: "Moderate Diabetic Retinopathy",
    3: "Severe Diabetic Retinopathy",
    4: "Proliferative Diabetic Retinopathy"
}

THRESHOLD = 60  # minimum confidence %

print("\n🔍 Predicting images...\n")

# Loop through images
for img_name in os.listdir(IMG_DIR):
    img_path = os.path.join(IMG_DIR, img_name)

    # Load & preprocess image
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    prediction = model.predict(img, verbose=0)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    # 🔑 THRESHOLD LOGIC (THIS IS THE ANSWER)
    print(f"🖼 Image: {img_name}")

    if confidence < THRESHOLD:
        print("❌ Image not clear or not a retinal image")
    else:
        print(f"➡ Disease: {class_names[predicted_class]}")
        print(f"➡ Confidence: {confidence:.2f}%")

    print("-" * 50)

print("✅ Batch prediction completed.")
