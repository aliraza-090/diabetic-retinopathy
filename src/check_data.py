import os
import pandas as pd

train_csv = "../data/train.csv"
train_images = "../data/train_images"

print("CSV exists:", os.path.exists(train_csv))
print("Images folder exists:", os.path.exists(train_images))

df = pd.read_csv(train_csv)
print("CSV loaded successfully")
print(df.head())

# check one image example
sample_id = df.iloc[0]["id_code"]
image_path = os.path.join(train_images, sample_id + ".png")
print("Sample image exists:", os.path.exists(image_path))
