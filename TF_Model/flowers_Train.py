import os
import pathlib2
import numpy as np
import tensorflow as tf
from tf_tools import gpu_growth, hide_WARN
import base64
import argparse

gpu_growth()
hide_WARN()

def load_img(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img)
    img = tf.image.resize(img, (224, 224))
    img /= 255.
    return img

class TF_Serving_Model(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    @tf.function(input_signature=[tf.TensorSpec([], tf.string, name="images")])
    def serving(self, inputs):
        img = tf.io.decode_base64(inputs, name="Decode_Base64")
        img = tf.reshape(img, [])
        img = tf.image.decode_jpeg(img, channels=3, name="Decode_jpg")
        img = tf.image.resize(img, (224, 224), name="Resize_img")
        img /= 255.
        img = tf.expand_dims(img, axis=0)
        return self.call(img)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="TF_Model/data/flowers/", type=str, help='Data Path')
    parser.add_argument("--save_path", default="TF_Model/weights/flowers/", type=str, help='Model Save Path')
    parser.add_argument("--epochs", default=5, type=int, help="Training epochs")
    parser.add_argument("--version", default=1, type=int, help="Model Version")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch Size")
    parser.add_argument("--random_seed", default=1048596, type=int, help="This is the choice of Steins Gate.")
    
    opt = parser.parse_args()
    
    DATA_DIR = opt.data_path
    MODEL_DIR = opt.save_path
    EPOCHS = opt.epochs
    VERSION = opt.version
    BATCH_SIZE = opt.batch_size
    RANDOM_SEED = opt.random_seed
    
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)
    
    data_root = pathlib2.Path(DATA_DIR)
    class_list = {path.name:i for i, path in enumerate(data_root.glob("./*"))}
    img_path_list = [str(path) for path in data_root.glob("./*/*.jpg")]
    label_list = [float(class_list[str(path.parent.name)]) for path in data_root.glob("./*/*.jpg")]

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    path_ds = tf.data.Dataset.from_tensor_slices(img_path_list)
    image_ds = path_ds.map(load_img, num_parallel_calls=AUTOTUNE)
    label_ds = tf.data.Dataset.from_tensor_slices(label_list)
    image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))
    image_label_ds = image_label_ds.shuffle(buffer_size=len(label_list)).batch(BATCH_SIZE)
    image_label_ds = image_label_ds.prefetch(buffer_size=AUTOTUNE)
    
    base_model = tf.keras.applications.ResNet50(input_shape=(224,224,3), weights='imagenet', include_top=False)
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(5)(x)
    predictions = tf.keras.layers.Softmax()(x)
    model = TF_Serving_Model(inputs=base_model.input, outputs=predictions)
    model.compile(loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True), 
                  optimizer=tf.optimizers.Adam(),
                  metrics=["acc"])

    model.fit(image_label_ds, epochs=EPOCHS, verbose=1, shuffle=True)

    export_path = os.path.join(MODEL_DIR, str(VERSION))
    print("\nexport_path = {}".format(export_path))

    signatures={"serving": model.serving}

    tf.keras.models.save_model(model,
                               export_path,
                               overwrite=True,
                               save_format=None,
                               signatures=signatures,
                               options=None)
    print("======Save complete!=======")
    