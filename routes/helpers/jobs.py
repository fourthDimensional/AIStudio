from redis import Redis
from rq import Queue
from rq.worker import Worker, WorkerStatus
from rq.command import send_shutdown_command, send_kill_horse_command
from rq.serializers import DefaultSerializer
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

class JobConfig:
    pass

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
    def __init__(self, redis_connection: Redis):
        self.redis_connection = redis_connection

        self.agents = {}

        self.compile_queue = Queue('compiling', connection=Redis())
        self.train_queue = Queue('training', connection=self.redis_connection)
        self.inference_queue = Queue('inference', connection=self.redis_connection)
        self.data_queue = Queue('data', connection=self.redis_connection)

    def queue_compile_job(self, model_wrapper, compiler):
        job = self.compile_queue.enqueue(compiler.compile_model, model_wrapper)

        return job