import numpy as np
import matplotlib.pyplot as plt
import tempfile
import os 
import tensorflow as tf
import argparse
from tf_tools import gpu_growth
from tf_tools import hide_WARN

gpu_growth()
hide_WARN()

class TF_Serving_Model(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    @tf.function(input_signature=[tf.TensorSpec([None, 28, 28, 1], tf.float32)])
    def serving(self, inputs):
        return self.call(inputs)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", default="TF_Model/weights/fashion_mnist", type=str, help='Model Save path')
    parser.add_argument("--epochs", default=5, type=int, help="Training epochs")
    parser.add_argument("--version", default=1, type=int, help="Model Version")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch Size")
    parser.add_argument("--random_seed", default=1048596, type=int, help="This is the choice of Steins Gate.")
    
    opt = parser.parse_args()
    
    MODEL_DIR = opt.save_path 
    EPOCHS = opt.epochs
    VERSION = opt.version
    BATCH_SIZE = opt.batch_size
    RANDOM_SEED = opt.random_seed 
    
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)
    
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
    test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)
    
    inputs = tf.keras.Input(shape=(28, 28, 1), name="inputs")
    x = tf.keras.layers.Conv2D(filters=8, kernel_size=3, strides=2)(inputs)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(10)(x)
    outputs = tf.keras.layers.Softmax()(x)
    model = TF_Serving_Model(inputs, outputs)

    model.compile(optimizer='adam', 
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    model.fit(train_images, train_labels, epochs=EPOCHS, batch_size=BATCH_SIZE)

    print("\nTesting Set Result:")
    test_loss, test_acc = model.evaluate(test_images, test_labels)

    export_path = os.path.join(MODEL_DIR, str(VERSION))
    print("export_path = {}".format(export_path))

    signatures={"serving": model.serving}

    tf.keras.models.save_model(model,
                               export_path,
                               overwrite=True,
                               save_format=None,
                               signatures=signatures,
                               options=None)
    
    print("======Save complete!=======")
    