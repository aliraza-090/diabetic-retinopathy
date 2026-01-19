print("Training started...")

import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

# Paths
CSV_PATH = "../data/train.csv"
IMG_DIR = "../data/train_images"
MODEL_PATH = "../saved_models/model.h5"

# Load CSV
df = pd.read_csv(CSV_PATH)
df["id_code"] = df["id_code"].astype(str) + ".png"

# Image generator
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    validation_split=0.2
)

train_gen = datagen.flow_from_dataframe(
    df,
    directory=IMG_DIR,
    x_col="id_code",
    y_col="diagnosis",
    target_size=(224, 224),
    batch_size=8,
    class_mode="raw",
    subset="training"
)

val_gen = datagen.flow_from_dataframe(
    df,
    directory=IMG_DIR,
    x_col="id_code",
    y_col="diagnosis",
    target_size=(224, 224),
    batch_size=8,
    class_mode="raw",
    subset="validation"
)

# Load model
model = load_model(MODEL_PATH)
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train model
model.fit(train_gen, validation_data=val_gen, epochs=3)

# Save trained model
model.save("../saved_models/model_trained.h5")

print("Training completed and model saved!")

import pickle

with open("../saved_models/history.pkl", "wb") as f:
    pickle.dump(history.history, f)
