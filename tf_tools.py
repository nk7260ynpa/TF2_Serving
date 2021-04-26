import os
import tensorflow as tf
from tensorflow.keras.layers import Activation
from tensorflow.keras.utils import get_custom_objects
import logging


def gpu_growth():
	gpus = tf.config.experimental.list_physical_devices("GPU")
	for i in range(len(gpus)):
		tf.config.experimental.set_memory_growth(gpus[i], True)
	print("Total {} GPUS".format(len(gpus)))
	return len(gpus)

def hide_WARN():
    tf.get_logger().setLevel(logging.ERROR)


