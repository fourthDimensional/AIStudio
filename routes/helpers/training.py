from redis import Redis
from rq import Queue, Worker
from rq.command import send_shutdown_command

"""
WIP Up-to-date Training Class Code

Currently being written.

Needs to be integrated into the training process across the codebase.

Will be used to train models and manage training agents and package configs.
"""

class TrainingConfigPackager:
    pass

class TrainingConfig:
    pass

class TrainingAgent:
    def __init__(self, redis_connection: Redis):
        self.redis_connection = redis_connection

    def kill(self):
        send_shutdown_command(self.redis_connection, self.name)


class TrainingManager:
    pass