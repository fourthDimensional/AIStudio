from redis import Redis
from routes.helpers.submodules.storage import RedisFileStorage
from routes.helpers.submodules.storage import StorageInterface
from routes.helpers.compiler import TensorflowModelCompiler
from routes.helpers.training import TrainingConfigPackager
import routes.helpers.model as model
import routes.helpers.submodules.layers_registry as layers
import routes.helpers.submodules.data_proc as data_proc

import os

import pandas as pd
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

data = dataset_storage.get_file(f"{api_key}:{dataset_key}")
csv_buffer = BytesIO(data)
dataframe = pd.read_csv(csv_buffer)

layer_manipulator = model.LayerManipulator()
hyperparameter_manager = model.HyperparameterManager()
dataprocessing_engine = model.DataProcessingEngine()

column_deletion = data_proc.ColumnDeletion('date')
dataprocessing_engine.add_modification(column_deletion)
print(dataprocessing_engine.process_data(dataframe))
dataprocessing_engine.set_input_fields(dataframe)
print(dataprocessing_engine.input_fields)
dataprocessing_engine.add_label_column('rainfall')
print(dataprocessing_engine.separate_labels(dataframe))

input_layer = layers.InputLayer()
dense_layer = layers.DenseLayer()
layer_manipulator.add_layer(input_layer, 0, 0)
layer_manipulator.add_layer(dense_layer, 0, 0)
print(layer_manipulator.get_layers())

model_compiler = TensorflowModelCompiler()
config_packager = TrainingConfigPackager()

new_model = model.ModelWrapper(dataprocessing_engine, hyperparameter_manager, layer_manipulator)
