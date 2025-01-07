from redis import Redis
from routes.helpers.submodules.storage import RedisFileStorage
from routes.helpers.submodules.storage import StorageInterface
from routes.helpers.compiler import ModelCompiler
import routes.helpers.jobs as jobs
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
import time

# Configuration for Redis connection
redis_host: str = 'localhost'
redis_port: int = 6379
redis_db: int = 0

REDIS_CONNECTION_INFO = {
    'host': os.getenv('REDIS_HOST', redis_host),
    'port': int(os.getenv('REDIS_PORT', str(redis_port))),
    'decode_responses': True
}

REDIS_WORKER_CONNECTION_INFO = {
    'host': os.getenv('REDIS_HOST', redis_host),
    'port': int(os.getenv('REDIS_PORT', str(redis_port))),
}

# jobs.kill_all_workers(Redis(**REDIS_CONNECTION_INFO))

job_manager = jobs.JobManager(REDIS_WORKER_CONNECTION_INFO)

dataset_storage = StorageInterface(RedisFileStorage(Redis(**REDIS_CONNECTION_INFO)))

api_key = '0e88f732d5f4d145130de7e210cd9a03'
dataset_key = 'rainfall_amount_regression'

print(os.environ.get('KERAS_BACKEND'))

# data = dataset_storage.get_file(f"{api_key}:{dataset_key}")
# csv_buffer = BytesIO(data)
# dataframe = pd.read_csv(csv_buffer)

dataframe = pd.read_csv('static/datasets/rainfall_amount_regression.csv')

layer_manipulator = model.LayerManipulator()
dataprocessing_engine = model.DataProcessingEngine()

dataprocessing_engine.add_modification(data_proc.DateFeatureExtraction('date'))
dataprocessing_engine.add_modification(data_proc.StringLookup('weather_condition'))

# TODO Column deletion of date_day also deletes date_dayofyear because of the multi-column regex checking

dataprocessing_engine.add_modification(data_proc.ColumnDeletion(['date_month', 'date_day', 'date_year']))
dataprocessing_engine.set_input_fields(dataframe)
dataprocessing_engine.add_label_column('weather_condition')

x, y = dataprocessing_engine.separate_labels(dataframe)

pd.DataFrame(x).to_csv('x.csv')
pd.DataFrame(y).to_csv('y.csv')

input_layer = layers.InputLayer(input_size=5)
dense_layer = layers.DenseLayer(units=10)

layer_manipulator.add_layer(input_layer, 0, 0)
layer_manipulator.forward_layer(0, 0)
layer_manipulator.add_layer(layers.BatchNormalizationLayer(), 1, 0)
layer_manipulator.forward_layer(1, 0)
layer_manipulator.add_layer(layers.ReshapeLayer(target_shape=(1, 5)), 2, 0)
layer_manipulator.forward_layer(2, 0)
layer_manipulator.add_layer(layers.GRULayer(units=5), 3, 0)
layer_manipulator.forward_layer(3, 0)
layer_manipulator.add_layer(layers.FlattenLayer(), 4, 0)
layer_manipulator.forward_layer(4, 0)
layer_manipulator.add_layer(layers.DenseLayer(units=3),5, 0)


print(layer_manipulator.get_layer_hyperparameters(5, 0))


model_compiler = ModelCompiler()
config_packager = jobs.JobConfigPackager()

new_model = model.ModelWrapper(dataprocessing_engine, layer_manipulator)

job = job_manager.queue_train_job(new_model, model_compiler, config_packager)

# _, compile_job = job_manager.queue_compile_job(new_model, model_compiler)
#
# time.sleep(5)
#
# model = compile_job.return_value()

# model.compile(optimizer=Adam(), loss=MeanSquaredError(), metrics=[BinaryAccuracy()])
# model.fit(x, y, epochs=100, batch_size=11)
#
# model.summary()

# keras.utils.plot_model(model, "test.png", rankdir='LR', show_shapes=True)