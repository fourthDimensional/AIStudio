from abc import ABC, abstractmethod
from redis import Redis

class Datastore:
    def __init__(self, database):
        self.key_database = database


class AbstractKeyDatabase(ABC):
    @abstractmethod
    def exists(self, key):
        pass

    @abstractmethod
    def get(self, key):
        pass

    @abstractmethod
    def delete(self, key):
        pass

    @abstractmethod
    def set(self, key, value):
        pass


class RedisDatabase(AbstractKeyDatabase):
    def __init__(self, redis_connection):
        self.redis_connection: Redis = redis_connection

    def exists(self, key):
        return self.redis_connection.exists(key)

    def get(self, key):
        return self.redis_connection.get(key)

    def delete(self, key):
        self.redis_connection.delete(key)

    def set(self, key, value):
        self.redis_connection.set(key, value)