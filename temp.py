import pandas as pd
import numpy as np
import itertools
import random
import tensorflow as tf
import os

dataframe_csv = None
encodings = ["utf-8", "utf-8-sig", "iso-8859-1", "latin1", "cp1252"]
for encoding in encodings:
    try:
        with open('/Users/Sam/Documents/HRL-DRL System Network Testing/AIStudio/2018.csv', 'r', encoding=encoding, errors='replace') as f:
            dataframe_csv = pd.read_csv(f)
        break
    except Exception as e:
        raise FileExistsError
        pass

dataframe_csv.drop(labels="Overall rank", axis=1)
dataframe_csv.drop(labels="Country or region", axis=1)

print(dataframe_csv.head())

features = dataframe_csv.copy()
labels = features.pop('Score')

inputs = {}

for name, column in features.items():
    dtype = column.dtype
    if dtype == object:
        dtype = tf.string
    else:
        dtype = tf.float32

    inputs[name] = tf.keras.Input(shape=(1,), name=name, dtype=dtype)

print(inputs)
