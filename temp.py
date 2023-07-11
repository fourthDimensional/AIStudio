import pandas as pd
import numpy as np
import tensorflow as tf

dataframe_csv = None
encodings = ["utf-8", "utf-8-sig", "iso-8859-1", "latin1", "cp1252"]
for encoding in encodings:
    try:
        with open('/Users/Sam/Documents/HRL-DRL System Network Testing/AIStudio/2018.csv', 'r', encoding=encoding,
                  errors='replace') as f:
            dataframe_csv = pd.read_csv(f)
        break
    except Exception as e:
        raise FileExistsError
        pass

dataframe_csv = dataframe_csv.drop(labels='Overall rank', axis=1)
dataframe_csv = dataframe_csv.drop(labels='Country or region', axis=1)

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

numeric_inputs = {name: input for name, input in inputs.items()
                  if input.dtype == tf.float32}

x = tf.keras.layers.Concatenate()(list(numeric_inputs.values()))
norm = tf.keras.layers.Normalization()

norm.adapt(np.array(dataframe_csv[numeric_inputs.keys()]))
all_numeric_inputs = norm(x)
preprocessed_inputs = [all_numeric_inputs]

dataframe_preprocessing = tf.keras.Model(inputs, preprocessed_inputs)

dataframe_features_dict = {name: np.array(value) for name, value in features.items()}

features_dict = {name: values for name, values in dataframe_features_dict.items()}

print(features_dict)


def dataframe_model(preprocessing_head, inputs):
    body = tf.keras.Sequential([
        tf.keras.layers.Dense(6),
        tf.keras.layers.Dense(3),
        tf.keras.layers.Dense(1)
    ])

    preprocessed_inputs_two = preprocessing_head(inputs)
    result = body(preprocessed_inputs_two)
    model = tf.keras.Model(inputs, result)

    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                  optimizer=tf.keras.optimizers.Adam())
    return model


dataframe_model_instance = dataframe_model(dataframe_preprocessing, inputs)

dataframe_model_instance.fit(x=features_dict, y=labels, epochs=100)

test_case = {
    'GDP per capita': [1.27],
    'Social support': [1.525],
    'Healthy life expectancy': [0.884],
    'Freedom to make life choices': [0.645],
    'Generosity': [0.376],
    'Perceptions of corruption': [0.142]
}

test_case_dict = {name: np.array(value) for name, value in test_case.items()}

print(test_case_dict)

print(dataframe_model_instance.predict(test_case_dict))


