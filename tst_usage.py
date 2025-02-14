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

import matplotlib.pyplot as plt

import os

import pandas as pd
import keras
from io import BytesIO
import time
import json

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

dataframe = pd.read_csv('static/datasets/rainfall_amount_regression.csv')

print(os.environ.get('KERAS_BACKEND'))

csv_buffer = BytesIO()
dataframe.to_csv(csv_buffer)
csv_buffer.seek(0)  # Move the cursor to the beginning of the buffer

dataset_storage.store_file(dataset_key, csv_buffer.read())

layer_manipulator = model.LayerManipulator()
data_processing_engine = model.DataProcessingEngine()

data_processing_engine.add_modification(data_proc.DateFeatureExtraction('date'))
data_processing_engine.add_modification(data_proc.StringLookup('weather_condition'))

# TODO Column deletion of date_day also deletes date_dayofyear because of the multi-column regex checking
data_processing_engine.set_input_fields(dataframe)
data_processing_engine.add_label_column('weather_condition')

x, y = data_processing_engine.separate_labels(dataframe)

pd.DataFrame(x).to_csv('x.csv')
pd.DataFrame(y).to_csv('y.csv')

input_layer = layers.InputLayer(input_size=8)
dense_layer = layers.DenseLayer(units=10)

layer_manipulator.add_layer(input_layer, 0, 0)
layer_manipulator.forward_layer(0, 0)
layer_manipulator.add_layer(layers.BatchNormalizationLayer(), 1, 0)
layer_manipulator.forward_layer(1, 0)
layer_manipulator.add_layer(layers.ReshapeLayer(target_shape=(1, 8)), 2, 0)
layer_manipulator.forward_layer(2, 0)
layer_manipulator.add_layer(layers.GRULayer(units=5), 3, 0)
layer_manipulator.forward_layer(3, 0)
layer_manipulator.add_layer(layers.FlattenLayer(), 4, 0)
layer_manipulator.forward_layer(4, 0)
layer_manipulator.add_layer(layers.DenseLayer(units=3),5, 0)


print(layer_manipulator.get_layer_hyperparameters(5, 0))


model_compiler = ModelCompiler(optimizer=Adam(), loss=MeanSquaredError(), metrics=[BinaryAccuracy()])
# config_packager = jobs.TrainingConfigPackager()

new_model = model.ModelWrapper(data_processing_engine, layer_manipulator, model_compiler)


job = job_manager.queue_train_job(new_model, None, 'rainfall_amount_regression', dataset_storage)

while not job.is_finished:
    print("Job is not finished")
    time.sleep(0.1)

model = job.return_value()

model.summary()

new_model.deregister(Redis(**REDIS_CONNECTION_INFO))
# keras.utils.plot_model(model, "test.png", rankdir='LR', show_shapes=True)

logs_history = job.get_meta()['logs_history']

def plot_loss_accuracy_history(logs_history):
    epochs, losses, accuracies = [], [], []

    for entry in logs_history:
        epochs.append(entry['epoch'])
        losses.append(entry.get('loss', 0))  # Ensure a default value if missing
        accuracies.append(entry.get('binary_accuracy', 0))

    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color='tab:blue')
    ax1.plot(epochs, losses, marker='o', linestyle='-', color='tab:blue', label="Loss")
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel("Accuracy", color='tab:orange')
    ax2.plot(epochs, accuracies, marker='s', linestyle='--', color='tab:orange', label="Accuracy")
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    fig.suptitle("Loss and Accuracy Over Time")
    fig.tight_layout()
    plt.grid()
    plt.show()

# save job.meta to a file
with open('job_meta.json', 'w') as f:
    json.dump(job.get_meta(), f)

plot_loss_accuracy_history(logs_history)