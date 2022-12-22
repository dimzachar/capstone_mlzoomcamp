from io import BytesIO
from pathlib import Path
from urllib import request

import numpy as np
import tflite_runtime.interpreter as tflite
from PIL import Image

# from keras_image_helper import create_preprocessor

model = Path("model.tflite")
interpreter = tflite.Interpreter(model_path=model.name)
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]
target_size = tuple(interpreter.get_input_details()[0]["shape"][1:3])


def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def prepare_image(img, target_size):
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize(target_size, Image.Resampling.NEAREST)
    return img

classes = [
    'Pebbles',
    'Shells'
]

def predict(url):
    img_orig = download_image(url)
    img = prepare_image(img_orig, target_size)

    x = np.array(img)
    X = np.array([x / 255], dtype=np.float32)

    interpreter.set_tensor(input_index, X)
    interpreter.invoke()

    preds = interpreter.get_tensor(output_index)

    return float(preds[0, 0])


def lambda_handler(event, context):
    url = event['url']
    pred = predict(url)
    class_label = "Shells" if pred >= 0.5 else "Pebbles"
    prediction_value = pred if pred >= 0.5 else 1 - pred
    result = {
        f"{class_label} - prediction": prediction_value
    }
    return result