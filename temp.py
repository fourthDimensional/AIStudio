import pandas as pd
import numpy as np
import itertools
import random

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf
import tensorflow.keras as ks

inputs = ks.Input(shape=(10,))
dense = ks.layers.Dense(17, activation="relu")
x = dense(inputs)
x = ks.layers.Dense(20, activation="relu")(x)
x = ks.layers.Dense(17, activation="sigmoid")(x)
outputs = ks.layers.Dense(10)(x)

model = ks.Model(inputs=inputs, outputs=outputs, name="test_model")

model.compile(
    loss='binary_crossentropy',
    optimizer=ks.optimizers.Adam(),
    metrics=['accuracy']
)

def generate_permutations(x):
    return [list(p) for p in itertools.product([0, 1], repeat=x)]

def generate_unique_arrays(x):
    permutations = generate_permutations(x)
    return [permutations[i % len(permutations)] for i in range(len(permutations))]

def shuffle_arrays(arrays):
    random.shuffle(arrays)

def invert_arrays(arrays):
    return [[1 - num for num in arr] for arr in arrays]

x = 10
arrays = generate_unique_arrays(x)
shuffle_arrays(arrays)
inverted_arrays = invert_arrays(arrays)

x_train, y_train = arrays, inverted_arrays

arrays = generate_unique_arrays(x)
shuffle_arrays(arrays)
inverted_arrays = invert_arrays(arrays)

x_test, y_test = arrays, inverted_arrays

print(x_test)

history = model.fit(x_train, y_train, batch_size=2, epochs=50)

test_scores = model.evaluate(x_test, y_test, verbose=2)
print("accuracy:", test_scores[1])
model.summary()

# Provide inputs for prediction
input_data = np.array([[1, 0, 1, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 0, 1, 0, 1, 1, 0, 1], [0, 1, 0, 0, 1, 1, 1, 0, 1, 0], [0, 0, 1, 0, 1, 0, 0, 1, 0, 1], [0, 1, 1, 0, 0, 0, 0, 0, 1, 0], [0, 1, 1, 1, 0, 0, 1, 0, 0, 1], [1, 1, 0, 1, 1, 1, 1, 0, 1, 0], [1, 0, 0, 1, 1, 0, 1, 1, 0, 1], [0, 0, 1, 0, 1, 0, 0, 0, 1, 1], [0, 1, 1, 0, 0, 1, 1, 1, 1, 1], [0, 0, 1, 1, 1, 0, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 0, 0, 0, 1], [0, 1, 1, 1, 0, 0, 0, 1, 1, 0], [1, 1, 0, 1, 0, 0, 1, 0, 1, 1], [0, 0, 1, 1, 0, 0, 1, 0, 0, 1], [0, 0, 0, 1, 1, 0, 1, 0, 0, 0], [0, 1, 0, 1, 0, 1, 0, 0, 0, 1], [1, 0, 1, 0, 1, 1, 1, 0, 0, 0], [0, 1, 0, 1, 1, 1, 0, 0, 1, 1], [1, 0, 1, 1, 1, 1, 0, 1, 1, 0], [1, 0, 1, 1, 1, 0, 0, 0, 0, 1], [0, 1, 0, 0, 0, 0, 0, 1, 1, 1], [0, 1, 0, 0, 1, 1, 0, 1, 1, 0], [0, 0, 0, 1, 1, 0, 1, 1, 0, 0], [1, 0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 1, 1, 1, 1, 0, 0, 0, 1, 1], [0, 1, 1, 0, 1, 0, 0, 0, 0, 0], [0, 1, 1, 0, 0, 0, 0, 1, 0, 1], [0, 0, 1, 0, 1, 0, 1, 0, 1, 0], [1, 0, 0, 1, 1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 0, 1, 1, 1, 0, 1], [1, 0, 0, 0, 1, 0, 0, 0, 0, 1], [1, 1, 1, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 1, 1, 1, 1, 0, 0, 1], [1, 0, 0, 0, 1, 0, 0, 0, 1, 1], [0, 0, 1, 0, 0, 0, 0, 1, 0, 1], [1, 0, 1, 1, 1, 0, 1, 0, 1, 1], [1, 0, 0, 0, 1, 1, 1, 1, 0, 0], [0, 1, 1, 1, 1, 0, 1, 0, 1, 1], [0, 1, 0, 1, 0, 0, 0, 0, 1, 0], [0, 1, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 1, 1, 0, 0, 1, 0], [1, 0, 1, 1, 1, 1, 1, 1, 0, 0], [0, 0, 0, 0, 1, 0, 1, 0, 1, 0], [1, 0, 0, 0, 0, 1, 0, 1, 1, 0], [0, 1, 1, 0, 0, 0, 1, 0, 1, 0]])  # Example input
threshold = 0.5
predictions = model.predict(input_data)
class_labels = (predictions > threshold).astype(int)
print("Predictions:", class_labels)