import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from sklearn.metrics import confusion_matrix, classification_report

# Load model (FIXED PATH)
model = load_model("../saved_models/model_v2.h5")

# Class names
class_names = [
    "No DR",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferative DR"
]

# Load CSV

df = pd.read_csv("../data/train.csv")

y_true = []
y_pred = []

print("🔍 Evaluating test images...")

for _, row in df.iterrows():
    img_path = f"../data/test_images/{row['id_code']}.png"

    if not os.path.exists(img_path):
        continue

    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img, verbose=0)
    y_pred.append(np.argmax(pred))
    y_true.append(row["diagnosis"])

# SAFETY CHECK
if len(y_true) == 0:
    print("❌ No images found for evaluation")
    exit()

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8,6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

print("\n📊 Classification Report:\n")
print(classification_report(y_true, y_pred, target_names=class_names))
