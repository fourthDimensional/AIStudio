from redis import Redis
from routes.helpers.submodules.storage import RedisFileStorage
from routes.helpers.submodules.storage import StorageInterface
from routes.helpers.compiler import ModelCompiler
from routes.helpers.training import TrainingConfigPackager
import routes.helpers.model as model
import routes.helpers.submodules.layers_registry as layers
import routes.helpers.submodules.data_proc as data_proc

from keras.optimizers import Adam
from keras.losses import MeanSquaredError
from keras.metrics import BinaryAccuracy

import os

import pandas as pd
import keras
from io import BytesIO

# Configuration for Redis connection
redis_host: str = 'localhost'
redis_port: int = 6379
redis_db: int = 0

REDIS_CONNECTION_INFO = {
    'host': os.getenv('REDIS_HOST', redis_host),
    'port': int(os.getenv('REDIS_PORT', str(redis_port))),
    'decode_responses': True
}

dataset_storage = StorageInterface(RedisFileStorage(Redis(**REDIS_CONNECTION_INFO)))

api_key = '0e88f732d5f4d145130de7e210cd9a03'
dataset_key = 'rainfall_amount_regression'

# data = dataset_storage.get_file(f"{api_key}:{dataset_key}")
# csv_buffer = BytesIO(data)
# dataframe = pd.read_csv(csv_buffer)

dataframe = pd.read_csv('static/datasets/aids_classification_small.csv')

layer_manipulator = model.LayerManipulator()
hyperparameter_manager = model.HyperparameterManager()
dataprocessing_engine = model.DataProcessingEngine()

dataprocessing_engine.set_input_fields(dataframe)
dataprocessing_engine.add_label_column('infected')

x, y = dataprocessing_engine.separate_labels(dataframe)

input_layer = layers.InputLayer(input_size=22)
dense_layer = layers.DenseLayer(units=50)

layer_manipulator.add_layer(input_layer, 0, 0)
layer_manipulator.forward_layer(0, 0)
layer_manipulator.add_layer(layers.BatchNormalizationLayer(), 1, 0)
layer_manipulator.forward_layer(1, 0)
layer_manipulator.add_layer(dense_layer, 2, 0)
layer_manipulator.forward_layer(2, 0)
layer_manipulator.add_layer(dense_layer, 3, 0)
layer_manipulator.forward_layer(3, 0)
layer_manipulator.add_layer(layers.DenseLayer(units=1), 4, 0)
# layer_manipulator.point_layer(1, 0, 2, 0, 0)
# layer_manipulator.point_layer(1, 0, 2, 1, 0)
# layer_manipulator.add_layer(dense_layer, 2, 0)
# layer_manipulator.forward_layer(2, 0)
# layer_manipulator.add_layer(dense_layer, 2, 1)
# layer_manipulator.point_layer(2, 1, 4, 0, 0)
# layer_manipulator.add_layer(dense_layer, 3, 0)
# layer_manipulator.forward_layer(3, 0)
# layer_manipulator.add_layer(dense_layer, 4, 0)


model_compiler = ModelCompiler()
config_packager = TrainingConfigPackager()

new_model = model.ModelWrapper(dataprocessing_engine, hyperparameter_manager, layer_manipulator)

model = model_compiler.compile_model(layer_manipulator.get_layers())

model.compile(optimizer=Adam(), loss=MeanSquaredError(), metrics=[BinaryAccuracy()])
model.fit(x, y, epochs=100, batch_size=64)

print(model_compiler.input_storage)

model.summary()

keras.utils.plot_model(model, "test.png", rankdir='LR', show_shapes=True)
