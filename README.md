# GreenIA


# TensorFlow Machine Learning Optimization Experiment

This project is an experiment on how to optimise TensorFlow machine learning.

The outcomes of the tests are the following:
- Reducing the number of pixels improved the accuracy (see Test 1)
- Adjusting Conv2D parameters improved the accuracy (see Test 3)
- Applying the suggested config of the TensorFlow documentation improved the running time (see Test 4)

## Original Model

Epoch 1/10
1563/1563 [==============================] - 282s 180ms/step - loss: 1.5492 - accuracy: 0.4339
Epoch 2/10
1563/1563 [==============================] - 326s 209ms/step - loss: 1.1873 - accuracy: 0.5766
Epoch 3/10
1563/1563 [==============================] - 338s 216ms/step - loss: 1.0382 - accuracy: 0.6348

Epoch 4/10
 716/1563 [============>.................] - ETA: 3:06 - loss: 0.9539 - accuracy: 0.6623
/usr/bin/python3 "/Users/nora/VSCodeProjects/Experiment projects/GreenIA/code/greenIA.py"
1563/1563 [==============================] - 350s 224ms/step - loss: 0.9399 - accuracy: 0.6723

## Test 1: 

Reduce Pixel Values

250 -> 25
x_train, x_test = x_train / 25.0, x_test / 25.0


Epoch 1/3
1563/1563 [==============================] - 326s 208ms/step - loss: 1.4701 - accuracy: 0.4674
Epoch 2/3
1563/1563 [==============================] - 383s 245ms/step - loss: 1.0955 - accuracy: 0.6150
Epoch 3/3
1563/1563 [==============================] - 391s 250ms/step - loss: 0.9445 - accuracy: 0.6696

## TEST2
remove sequence 

#MaxPooling2D((2, 2)),
#Conv2D(64, (3, 3), activation='relu'),

1563/1563 [==============================] - 361s 231ms/step - loss: 1.4305 - accuracy: 0.4856
Epoch 2/3
1563/1563 [==============================] - 404s 259ms/step - loss: 1.0822 - accuracy: 0.6227
Epoch 3/3
1563/1563 [==============================] - 349s 223ms/step - loss: 0.9283 - accuracy: 0.6754


## TEST3 * * * (best outcome)

adjust params

Conv2D(64, (3, 3), activation='relu') ->  Conv2D(32, (2, 2), activation='relu'),

Epoch 1/3
1563/1563 [==============================] - 332s 212ms/step - loss: 1.4279 - accuracy: 0.4913
Epoch 2/3
1563/1563 [==============================] - 333s 213ms/step - loss: 1.0706 - accuracy: 0.6247
Epoch 3/3
1563/1563 [==============================] - 340s 218ms/step - loss: 0.9190 - accuracy: 0.6800


## TEST4

remove sequence items

  Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
  MaxPooling2D((2, 2)),
  #Conv2D(64, (3, 3), activation='relu'),
  #MaxPooling2D((2, 2)),
  #Conv2D(32, (2, 2), activation='relu'),
  Flatten(),

Epoch 1/3
1563/1563 [==============================] - 320s 204ms/step - loss: 1.4465 - accuracy: 0.4792
Epoch 2/3
1563/1563 [==============================] - 351s 224ms/step - loss: 1.0824 - accuracy: 0.6179
Epoch 3/3
1563/1563 [==============================] - 372s 238ms/step - loss: 0.9366 - accuracy: 0.6704


## TEST5

adapt the image-optimized training model in the TensorFlow documentation https://www.tensorflow.org/tutorials/load_data/images

model = tf.keras.Sequential([
  tf.keras.layers.Rescaling(1./255),
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

Epoch 1/6
1563/1563 [==============================] - 162s 103ms/step - loss: 1.5630 - accuracy: 0.4317
Epoch 2/6
1563/1563 [==============================] - 198s 127ms/step - loss: 1.2131 - accuracy: 0.5686
Epoch 3/6
1563/1563 [==============================] - 194s 124ms/step - loss: 1.0818 - accuracy: 0.6193



