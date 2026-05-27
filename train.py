import os
import cv2
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

DATASET_DIR = "dataset"

CLASSES = ["heo", "meo", "chuot", "cho", "voi"]

IMG_SIZE = 28

X = []
y = []

for label, cls in enumerate(CLASSES):

    folder = os.path.join(DATASET_DIR, cls)

    if not os.path.isdir(folder):
        print("Missing folder:", folder)
        continue

    for filename in os.listdir(folder):

        path = os.path.join(folder, filename)

        img = cv2.imread(path)

        if img is None:
            continue

        # GRAYSCALE
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # RESIZE
        gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))

        # INVERT COLOR
        gray = 255 - gray

        # NORMALIZE
        gray = gray.astype("float32") / 255.0

        # FLATTEN
        gray = gray.reshape(784)

        X.append(gray)
        y.append(label)

X = np.array(X, dtype=np.float32)
y = np.array(y)

print("Dataset shape:", X.shape)

# ONE HOT
y = to_categorical(y, num_classes=5)

# ANN MODEL
model = Sequential()

model.add(Dense(256, activation="relu", input_shape=(784,)))

model.add(Dense(128, activation="relu"))

model.add(Dense(64, activation="relu"))

model.add(Dense(5, activation="softmax"))

# COMPILE
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# TRAIN
model.fit(
    X,
    y,
    epochs=50,
    batch_size=8,
    validation_split=0.2,
    shuffle=True
)

# SAVE
model.save("animal_ann.h5")

print("Saved model: animal_ann.h5")