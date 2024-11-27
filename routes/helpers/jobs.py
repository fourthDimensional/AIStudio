from redis import Redis
from rq import Queue
from rq.worker import Worker, WorkerStatus
from rq.command import send_shutdown_command, send_kill_horse_command
from rq.serializers import DefaultSerializer

from routes.helpers.compiler import ModelCompiler
from routes.helpers.model import ModelWrapper

from keras.callbacks import EarlyStopping, ModelCheckpoint

import time

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

class JobConfigPackager:
    pass

class TrainingConfig:
    def __init__(self, optimizer, loss, metrics, epochs, batch_size):
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.epochs = epochs
        self.batch_size = batch_size

        self.compiled_model_uuid = None

    def set_compiled_model_uuid(self, compiled_model_uuid):
        self.compiled_model_uuid = compiled_model_uuid

    def define_early_stopping(self, monitor, min_delta, patience, mode):
        self.early_stopping = {
            'monitor': monitor,
            'min_delta': min_delta,
            'patience': patience,
            'mode': mode
        }

    def define_checkpointing(self, monitor, mode, save_best_only):
        self.checkpointing = {
            'monitor': monitor,
            'mode': mode,
            'save_best_only': save_best_only
        }


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

    def queue_compile_job(self, model_wrapper: ModelWrapper, compiler: ModelCompiler):
        job = self.compile_queue.enqueue(compiler.compile_model, model_wrapper, self.redis_connection)

        return model_wrapper.uuid, job

    def queue_train_job(self, model: ModelWrapper, compiler: ModelCompiler, training_config: TrainingConfig):
        training_config = TrainingConfig(optimizer='adam', loss='mean_squared_error', metrics=['binary_accuracy'], epochs=100, batch_size=11)
        # above is a placeholder for now

        uuid, _ = self.queue_compile_job(model, ModelCompiler())

        training_config.set_compiled_model_uuid(uuid)

        job = self.train_queue.enqueue(train_model, model, training_config)

        return job


def train_model(training_config: TrainingConfig):
    callbacks = []

    if hasattr(training_config, 'early_stopping'):
        early_stopping = EarlyStopping(
            monitor=training_config.early_stopping['monitor'],
            min_delta=training_config.early_stopping['min_delta'],
            patience=training_config.early_stopping['patience'],
            mode=training_config.early_stopping['mode']
        )
        callbacks.append(early_stopping)

    if hasattr(training_config, 'checkpointing'):
        checkpointing = ModelCheckpoint(
            monitor=training_config.checkpointing['monitor'],
            mode=training_config.checkpointing['mode'],
            save_best_only=training_config.checkpointing['save_best_only']
        )
        callbacks.append(checkpointing)

    keras_model.compile(optimizer=training_config.optimizer, loss=training_config.loss, metrics=training_config.metrics)
    keras_model.fit(x, y, epochs=training_config.epochs, batch_size=training_config.batch_size, callbacks=callbacks)

    return keras_model