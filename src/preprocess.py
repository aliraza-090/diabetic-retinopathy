import os
import cv2
import pandas as pd

DATA_DIR = "../data"
IMG_DIR = os.path.join(DATA_DIR, "train_images")
CSV_PATH = os.path.join(DATA_DIR, "train.csv")

IMG_SIZE = 224

df = pd.read_csv(CSV_PATH)

def preprocess_image(img_name):
    img_path = os.path.join(IMG_DIR, img_name + ".png")
    img = cv2.imread(img_path)
    if img is None:
        return None
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    return img

print("Preprocessing check started...")

sample = df.iloc[0]["id_code"]
img = preprocess_image(sample)

if img is not None:
    print("Image preprocessing successful")
    print("Image shape:", img.shape)
else:
    print("Image preprocessing failed")
