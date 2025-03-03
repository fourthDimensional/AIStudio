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
from keras.metrics import BinaryAccuracy, Accuracy

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
dataset_key = 'classification_small'

dataframe = pd.read_csv(f'static/datasets/{dataset_key}.csv')

csv_buffer = BytesIO()
dataframe.to_csv(csv_buffer)
csv_buffer.seek(0)  # Move the cursor to the beginning of the buffer

dataset_storage.store_file(dataset_key, csv_buffer.read())

layer_manipulator = model.LayerManipulator()
data_processing_engine = model.DataProcessingEngine()

data_processing_engine.add_modification(data_proc.ColumnDeletion(['time']))

data_processing_engine.set_input_fields(dataframe)
data_processing_engine.add_label_column('infected')

x, y = data_processing_engine.separate_labels(dataframe)

input_layer = layers.InputLayer(input_size=21)
dense_layer = layers.DenseLayer(units=20)

layer_manipulator.add_layer(input_layer, 0, 0)
layer_manipulator.forward_layer(0, 0)
layer_manipulator.add_layer(layers.BatchNormalizationLayer(), 1, 0)
layer_manipulator.forward_layer(1, 0)
layer_manipulator.add_layer(dense_layer, 2, 0)
layer_manipulator.forward_layer(2, 0)
layer_manipulator.add_layer(layers.DenseLayer(units=1), 3, 0)


model_compiler = ModelCompiler(optimizer=Adam(), loss=MeanSquaredError(), metrics=[BinaryAccuracy()])

new_model = model.ModelWrapper(data_processing_engine, layer_manipulator, model_compiler)


job = job_manager.queue_train_job(new_model, None, dataset_key, dataset_storage)

while not job.is_finished:
    time.sleep(0.1)

trained_model = job.return_value()
eval_job = job_manager.queue_evaluation_job(new_model, dataset_key, dataset_storage, trained_model)

trained_model.summary()

new_model.deregister(Redis(**REDIS_CONNECTION_INFO))
# keras.utils.plot_model(trained_model, "test.png", rankdir='LR', show_shapes=True)

logs_history = job.get_meta()['logs_history']


def plot_loss_accuracy_history(logs_history, plateau_epoch=None):
    epochs, losses, accuracies = [], [], []

    for entry in logs_history:
        epochs.append(entry['epoch'])
        losses.append(entry.get('loss', 0))  # Ensure a default value if missing
        accuracies.append(entry.get('binary_accuracy', 0))

    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color='tab:blue')
    loss_line, = ax1.plot(epochs, losses, marker='o', linestyle='-', color='tab:blue', label="Loss")
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # Add vertical line if plateau_epoch is provided
    if plateau_epoch is not None:
        ax1.axvline(x=plateau_epoch, color='red', linestyle='--', label=f'Plateau (epoch {plateau_epoch})')

    ax2 = ax1.twinx()
    ax2.set_ylabel("Accuracy", color='tab:orange')
    acc_line, = ax2.plot(epochs, accuracies, marker='s', linestyle='--', color='tab:orange', label="Accuracy")
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    fig.suptitle("Loss and Accuracy Over Time")
    fig.tight_layout()
    plt.grid()

    # Combine legends from both axes
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

    plt.show()


# Save job.meta to a file for reference
with open('job_meta.json', 'w') as f:
    json.dump(job.get_meta(), f)

# Extract plateau epoch from the job metadata if it exists
job_meta = job.get_meta()
plateau_epoch = job_meta.get('plateau_epoch')

# Plot loss and accuracy history including a vertical line for the plateau epoch
plot_loss_accuracy_history(logs_history, plateau_epoch)
