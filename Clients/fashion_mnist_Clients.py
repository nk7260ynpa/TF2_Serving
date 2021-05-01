import tensorflow as tf
import requests
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import base64
import json
from tf_tools import hide_WARN

os.environ["CUDA_VISIBLE_DEVICES"] = ""
hide_WARN()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_amounts", default=10, type=int, help="Data Validation Amount")
    parser.add_argument("--random_seed", default=1048596, type=int, help="This is the choice of Steins Gate.")
    opt = parser.parse_args()
    
    DATA_VALIDATION_AMOUNTS = opt.data_amounts
    RANDOM_SEED = opt.random_seed
    
    np.random.seed(571024)
    tf.random.set_seed(RANDOM_SEED)

    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    test_images = test_images / 255.0
    test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    data = json.dumps({"signature_name": "serving", "instances": test_images[0:DATA_VALIDATION_AMOUNTS].tolist()})
    headers = {"content-type": "application/json"}
    json_response = requests.post('http://localhost:8510/v1/models/fashion_mnist_model/versions/1:predict', data=data, headers=headers)
    predictions = json.loads(json_response.text)['predictions']

    print("\nResult")
    for i in range(0,DATA_VALIDATION_AMOUNTS):
        if np.argmax(predictions[i]) == test_labels[i]:
            result = "OK"
        else:
            result = "NG"
        print('Predict: {:<12} (class {}), Ground Truth: {:<12} (class {}) {}'.format(
             class_names[np.argmax(predictions[i])], np.argmax(predictions[i]), class_names[test_labels[i]], test_labels[i], result))