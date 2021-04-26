import tensorflow as tf
import requests
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess
import base64
import json
from tf_tools import hide_WARN

os.environ["CUDA_VISIBLE_DEVICES"] = ""
hide_WARN()

np.random.seed(571024)

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
test_images = test_images / 255.0
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

data = json.dumps({"signature_name": "serve", "instances": test_images[0:10].tolist()})
headers = {"content-type": "application/json"}
json_response = requests.post('http://localhost:8510/v1/models/random_model/versions/1:predict', data=data, headers=headers)
predictions = json.loads(json_response.text)['predictions']

print("\nResult")
for i in range(0,10):
    if np.argmax(predictions[i]) == test_labels[i]:
        result = "OK"
    else:
        result = "NG"
    print('Predict: {:<10} (class {}), Ground Truth: {:<10} (class {}) {}'.format(
         class_names[np.argmax(predictions[i])], np.argmax(predictions[i]), class_names[test_labels[i]], test_labels[i], result))