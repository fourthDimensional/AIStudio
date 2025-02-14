from redis import Redis
from rq import Queue, get_current_job
from rq.worker import Worker, WorkerStatus
from rq.command import send_shutdown_command, send_kill_horse_command
from rq.serializers import DefaultSerializer

from routes.helpers.compiler import ModelCompiler
from routes.helpers.model import ModelWrapper

from keras.callbacks import EarlyStopping, ModelCheckpoint, LambdaCallback

import time


# TEMp
import pandas as pd

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

        self.compile_queue = Queue('compiling', connection=connection)
        self.train_queue = Queue('training', connection=connection)
        self.inference_queue = Queue('inference', connection=connection)
        self.data_queue = Queue('data', connection=connection)

    def queue_train_job(self, model: ModelWrapper, training_config: TrainingConfig):
        training_config = TrainingConfig(epochs=100, batch_size=11)
        # above is a placeholder for now

        job = self.train_queue.enqueue(train_model, training_config, model)

        return job


def train_model(training_config: TrainingConfig, model: ModelWrapper):
    job = get_current_job()

    def update_logs(epoch, logs):
        if 'logs_history' not in job.meta:
            job.meta['logs_history'] = []

        timestamp = time.time()
        job.meta['logs_history'].append({'epoch': epoch, 'timestamp': timestamp, **logs})
        job.save_meta()

    callbacks = [LambdaCallback(on_epoch_end=update_logs)]

    dataframe = pd.read_csv('static/datasets/rainfall_amount_regression.csv')

    x, y = model.data_processing_engine.separate_labels(dataframe)

    compiled_model = model.compiler.compile_model(model)

    print("Compiled model", compiled_model)

    compiled_model.fit(x, y, epochs=training_config.epochs, batch_size=training_config.batch_size, callbacks=callbacks)

    return compiled_model