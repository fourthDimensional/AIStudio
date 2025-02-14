from redis import Redis
from rq import Queue, get_current_job
from rq.worker import Worker, WorkerStatus
from rq.command import send_shutdown_command, send_kill_horse_command
from rq.serializers import DefaultSerializer

from routes.helpers.submodules.storage import StorageInterface
from routes.helpers.compiler import ModelCompiler
from routes.helpers.model import ModelWrapper

from keras.callbacks import EarlyStopping, ModelCheckpoint, LambdaCallback
from io import BytesIO

import time
import pickle

# TEMp
import pandas as pd

# speed up the process
import tensorflow

"""
WIP Up-to-date Training Class Code

Currently being written.

Needs to be integrated into the training process across the codebase.

Will be used to train models and manage training agents and package configs.
"""

def kill_all_workers(redis_connection: Redis):
    workers = Worker.all(connection=redis_connection)

    for worker in workers:
        send_shutdown_command(redis_connection, worker.name)

class TrainingConfigPackager:
    pass

class TrainingConfig:
    def __init__(self, epochs, batch_size):
        self.epochs = epochs
        self.batch_size = batch_size


# class WorkerAgent:
#     def __init__(self, redis_connection: Redis, name: str, queues: list[Queue]):
#         self.redis_connection = redis_connection
#
#         self.name = name
#         self.queues = queues
#
#         self.worker = Worker(self.queues, connection=self.redis_connection, name=self.name)
#
#     def start(self):
#         self.worker.work()
#
#     def kill(self):
#         send_shutdown_command(self.redis_connection, self.name)


class JobManager:
    def __init__(self, redis_connection: dict):
        self.redis_connection = redis_connection

        self.agents = {}

        connection = Redis(**self.redis_connection)

        self.train_queue = Queue('training', connection=connection)
        self.evaluation_queue = Queue('evaluation', connection=connection)
        self.inference_queue = Queue('inference', connection=connection)
        self.data_queue = Queue('data', connection=connection)

    def _queue_data_job(self, model: ModelWrapper, path, storage_interface: StorageInterface):
        raw_data = storage_interface.get_file(path)
        cache_key = f'processed_data:{path}:version:{model.data_processing_engine.get_version()}'

        return self.data_queue.enqueue(process_data, model, raw_data, self.redis_connection, cache_key)

    def queue_train_job(self, model: ModelWrapper, training_config: TrainingConfig, path, storage_interface: StorageInterface):
        training_config = TrainingConfig(epochs=100, batch_size=10)
        # above is a placeholder for now

        data_compilation_job = self._queue_data_job(model, path, storage_interface)

        job = self.train_queue.enqueue(train_model, training_config, model, depends_on=data_compilation_job)

        return job

    def queue_evaluation_job(self, model: ModelWrapper, path, storage_interface: StorageInterface, trained_model):
        data_compilation_job = self._queue_data_job(model, path, storage_interface)

        job = self.evaluation_queue.enqueue(evaluate_model, trained_model, depends_on=data_compilation_job)

        return job


def evaluate_model(trained_model):
    job = get_current_job()

    x, y = job.dependency.return_value()

    trained_model.evaluate(x, y)

    return trained_model


def process_data(model, raw_data, redis_connection_info, cache_key: str):
    redis_connection = Redis(**redis_connection_info)
    # Check if the processed data is already cached
    cached_data = redis_connection.get(cache_key)
    if cached_data:
        # Load the cached data
        x, y = pickle.loads(cached_data)
    else:
        # Process the raw data
        buffer = BytesIO(raw_data)
        dataframe = pd.read_csv(buffer, index_col=0)
        dataframe.head()
        x, y = model.data_processing_engine.separate_labels(dataframe)

        # Cache the processed data
        redis_connection.set(cache_key, pickle.dumps((x, y)), ex=3600)  # 1 hour

    return x, y

def train_model(training_config: TrainingConfig, model: ModelWrapper):
    job = get_current_job()

    def update_logs(epoch, logs):
        if 'logs_history' not in job.meta:
            job.meta['logs_history'] = []

        timestamp = time.time()
        job.meta['logs_history'].append({'epoch': epoch, 'timestamp': timestamp, **logs})
        job.save_meta()

    callbacks = [LambdaCallback(on_epoch_end=update_logs)]

    x, y = job.dependency.return_value()

    compiled_model = model.compiler.compile_model(model)

    print("Compiled model", compiled_model)

    compiled_model.fit(x, y, epochs=training_config.epochs, batch_size=training_config.batch_size, callbacks=callbacks)

    return compiled_model