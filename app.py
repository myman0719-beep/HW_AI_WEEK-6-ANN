from flask import Flask, render_template, request, jsonify
import numpy as np
from keras.models import load_model
from PIL import Image
import base64
import io

app = Flask(__name__)

# LOAD MODEL
model = load_model("animal_ann.h5")

# CLASS NAMES
classes = ["heo", "meo", "chuot", "cho", "voi"]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    try:
        data = request.json["image"]

        # REMOVE BASE64 HEADER
        image_data = data.split(",")[1]

        # OPEN IMAGE
        image = Image.open(
            io.BytesIO(base64.b64decode(image_data))
        ).convert("L")

        # RESIZE
        image = image.resize((28, 28))

        # CONVERT TO ARRAY
        img_array = np.array(image)

        # INVERT COLOR
        img_array = 255 - img_array

        # NORMALIZE
        img_array = img_array.astype("float32") / 255.0

        # FLATTEN FOR ANN
        img_array = img_array.reshape(1, 784)

        # PREDICT
        prediction = model.predict(img_array)

        predicted_index = np.argmax(prediction)

        result = classes[predicted_index]

        confidence = float(np.max(prediction))

        return jsonify({
            "prediction": result,
            "confidence": round(confidence * 100, 2)
        })

    except Exception as e:
        return jsonify({
            "prediction": "error",
            "confidence": 0,
            "message": str(e)
        })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)