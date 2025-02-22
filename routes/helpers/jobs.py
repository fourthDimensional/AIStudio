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
from sklearn.model_selection import train_test_split

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
    def __init__(self, epochs, batch_size, test_size=0.2):
        self.epochs = epochs
        self.batch_size = batch_size
        self.test_size = test_size


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

    def queue_profile_report_job(self, dataset_key: str, storage_interface: StorageInterface):
        raw_data = storage_interface.get_file(dataset_key)
        job = self.data_queue.enqueue(generate_profile_report, raw_data, dataset_key, apikey, self.redis_connection)

        return job

    def _queue_data_job(self, model: ModelWrapper, path, storage_interface: StorageInterface):
        raw_data = storage_interface.get_file(path)
        cache_key = f'processed_data:{path}:version:{model.data_processing_engine.get_version()}'

        return self.data_queue.enqueue(process_data, model, raw_data, self.redis_connection, cache_key)

    def queue_train_job(self, model: ModelWrapper, training_config: TrainingConfig, path, storage_interface: StorageInterface):
        training_config = TrainingConfig(epochs=100, batch_size=64, test_size=0.2)
        # above is a placeholder for now

        data_compilation_job = self._queue_data_job(model, path, storage_interface)

        job = self.train_queue.enqueue(train_model, training_config, model, depends_on=data_compilation_job)

        return job

    def queue_evaluation_job(self, model: ModelWrapper, path, storage_interface: StorageInterface, trained_model):
        data_compilation_job = self._queue_data_job(model, path, storage_interface)

        job = self.evaluation_queue.enqueue(evaluate_model, trained_model, depends_on=data_compilation_job)

        return job


def generate_profile_report(raw_data: bytes, result_key: string, redis_connection_info: dict) -> str:
    """
    Queue a profile report job for a dataset.

    :param raw_data: The raw data to generate a profile report for.
    :param result_key: The key to store the result in Redis.
    :param redis_connection_info: The connection information for the Redis instance.
    """
    redis_client = Redis(**redis_connection_info)
    job = get_current_job()

    buffer = BytesIO(raw_data)
    df = pd.read_csv(buffer, index_col=0)

    metadata = df.describe(include='all').to_dict()

    job.meta['handled_by'] = socket.gethostname()
    job.meta.update(metadata)
    job.save_meta()

    profile = ProfileReport(df, title=f"Profile Report for {name}")
    json_data = profile.to_json()
    redis_client.json().set(f"profile_report:{result_key}", '$',
                            json.loads(json_data))

    return f"profile_report:{result_key}"


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
        x, y = model.data_processing_engine.separate_labels(dataframe)

        # Cache the processed data
        redis_connection.set(cache_key, pickle.dumps((x, y)), ex=3600)  # 1 hour

    return x, y

def train_model(training_config: TrainingConfig, model: ModelWrapper):
    job = get_current_job()

    # This dictionary will keep track of the best loss and number of epochs without improvement.
    plateau_tracker = {
        'best_loss': float('inf'),
        'wait': 0,
        'patience': 5  # change this value to set how many epochs to wait before deciding that training has plateaued
    }

    def update_logs(epoch, logs):
        # Initialize logs history if not yet present
        if 'logs_history' not in job.meta:
            job.meta['logs_history'] = []
        timestamp = time.time()
        job.meta['logs_history'].append({'epoch': epoch, 'timestamp': timestamp, **logs})

        # Monitor the loss (or validation loss if available)
        current_loss = logs.get('loss')
        # Optionally, if you have validation data, you might want to monitor:
        # current_loss = logs.get('val_loss', current_loss)

        if current_loss is not None:
            # Check if the loss has improved significantly (using a tolerance)
            if current_loss < plateau_tracker['best_loss'] - 1e-4:
                plateau_tracker['best_loss'] = current_loss
                plateau_tracker['wait'] = 0
            else:
                plateau_tracker['wait'] += 1

            # If the loss hasn't improved for 'patience' epochs, log the plateau epoch
            if plateau_tracker['wait'] >= plateau_tracker['patience'] and 'plateau_epoch' not in job.meta:
                job.meta['plateau_epoch'] = epoch
        job.save_meta()

    # Add the LambdaCallback that calls update_logs at the end of each epoch
    callbacks = [LambdaCallback(on_epoch_end=update_logs)]



    # Get the processed data from the dependency job
    x, y = job.dependency.return_value()

    x, x_val, y, y_val = train_test_split(x, y, test_size=training_config.test_size, random_state=42)

    # Compile the model using the compiler defined in the model wrapper
    compiled_model = model.compiler.compile_model(model)
    print("Compiled model", compiled_model)

    # Train the model with the given configuration and callbacks
    compiled_model.fit(
        x,
        y,
        epochs=training_config.epochs,
        batch_size=training_config.batch_size,
        callbacks=callbacks,
        validation_data=[x_val, y_val]
    )

    return compiled_model
