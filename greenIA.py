import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D

# Set mixed precision policy
tf.keras.mixed_precision.set_global_policy('mixed_float16')
# Load data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# Normalize pixel values
x_train, x_test = x_train / 32.0, x_test / 32.0
# Define model
'''model = Sequential([
  Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
  MaxPooling2D((2, 2)),
  #Conv2D(64, (3, 3), activation='relu'),
  #MaxPooling2D((2, 2)),
  #Conv2D(32, (2, 2), activation='relu'),
  Flatten(),
  Dense(64, activation='relu'),
  Dense(10, dtype='float32')
])
'''
num_classes = 5

model = tf.keras.Sequential([
  #tf.keras.layers.Rescaling(1./255),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10, dtype='float32')
])

# Compile and train model
model.compile(optimizer='adam',
       loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
       metrics=['accuracy'])
model.fit(x_train, y_train, epochs=6)