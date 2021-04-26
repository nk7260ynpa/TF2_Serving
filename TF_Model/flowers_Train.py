import os
import pathlib2
import numpy as np
import tensorflow as tf
from tf_tools import gpu_growth, hide_WARN

gpu_growth()
hide_WARN()

BATCH_SIZE = 64
MODEL_DIR = "TF_Model/weights/flowers/"
VERSION = 1

def load_img(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img)
    img = tf.image.resize(img, (224, 224))
    img /= 255.
    return img

class TF_Serving_Model(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    @tf.function(input_signature=[tf.TensorSpec([None, 224, 224, 3], tf.float32)])
    def serve(self, inputs):
        return self.call(inputs)

data_root = pathlib2.Path("TF_Model/data/flowers/")
class_list = {path.name:i for i, path in enumerate(data_root.glob("./*"))}
img_path_list = [str(path) for path in data_root.glob("./*/*.jpg")]
label_list = [float(class_list[str(path.parent.name)]) for path in data_root.glob("./*/*.jpg")]

AUTOTUNE = tf.data.experimental.AUTOTUNE
path_ds = tf.data.Dataset.from_tensor_slices(img_path_list)
image_ds = path_ds.map(load_img, num_parallel_calls=AUTOTUNE)
label_ds = tf.data.Dataset.from_tensor_slices(label_list)
image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))
image_label_ds = image_label_ds.shuffle(buffer_size=len(label_list)).batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)

base_model = tf.keras.applications.ResNet50(input_shape=(224,224,3), weights='imagenet', include_top=False)
x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(5)(x)
predictions = tf.keras.layers.Softmax()(x)
model = TF_Serving_Model(inputs=base_model.input, outputs=predictions)
model.compile(loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True), 
              optimizer=tf.optimizers.Adam(),
              metrics=["acc"])

model.fit(image_label_ds, epochs=3, verbose=1, shuffle=True)

export_path = os.path.join(MODEL_DIR, str(VERSION))
print("export_path = {}".format(export_path))

signatures={"serve": model.serve}

tf.keras.models.save_model(model,
                           export_path,
                           overwrite=True,
                           save_format=None,
                           signatures=signatures,
                           options=None)