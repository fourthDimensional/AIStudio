from redis import Redis
from rq import Queue, Worker
from rq.command import send_shutdown_command

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