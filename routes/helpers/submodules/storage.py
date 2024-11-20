from redis import Redis
import base64
import os

"""
Up-to-date File Storage Code, currently being written.

Needs to be integrated into direct connections to Redis across the codebase.

Currently being used to store datasets for users, and will be used for other files.

Future Plans:
- Implement a more secure way to store files
- Implement a codebase wide way to consistently connection to Redis for misc. usage.
"""

class StorageInterface:
    """
    Interface for storing and accessing data from different sources. Composition based.
    """
    def __init__(self, implementation):
        self.implementation = implementation

    def get_file_metadata(self, file_id):
        """
        Get metadata for a file.

        :param file_id: The id of the file to get metadata for.
        :return: Metadata for the file.
        """
        return self.implementation.get_file_metadata(file_id)

    def get_chunk(self, file_id, chunk_address):
        """
        Get a chunk of a file.

        :param file_id: The id of the file to get a chunk of.
        :param chunk_address: The address of the chunk to get.
        :return: The chunk of the file.
        """
        return self.implementation.get_chunk(file_id, chunk_address)

    def get_file(self, file_id):
        """
        Get a file.

        :param file_id: The id of the file to get.
        :return: The file.
        """
        return self.implementation.get_file(file_id)

    def store_file(self, file_id, file, metadata=None):
        """
        Store a file.

        :param metadata:
        :param file_id: The id of the file to store.
        :param file: The file to store.
        """
        return self.implementation.store_file(file_id, file, metadata)

    def delete_file(self, file_id):
        """
        Delete a file.

        :param file_id: The id of the file to delete.
        """
        return self.implementation.delete_file(file_id)

    def exists(self, file_id):
        """
        Check if a file exists.

        :param file_id: The id of the file to check.
        """
        return self.implementation.exists(file_id)

    def start_upload(self, file_id, metadata=None):
        """
        Start an upload.

        :param metadata:
        :param file_id: The id of the file to upload.
        """
        return self.implementation.start_upload(file_id, metadata)

    def upload_chunk(self, file_id, chunk):
        """
        Upload a chunk.

        :param file_id: The id of the file to upload a chunk to.
        :param chunk: The chunk to upload.
        """
        return self.implementation.upload_chunk(file_id, chunk)

    def finish_upload(self, file_id):
        """
        Finish an upload.

        :param file_id: The id of the file to finish uploading.
        """
        return self.implementation.finish_upload(file_id)


class RedisFileStorage:
    def __init__(self, redis_connection: Redis, key_prefix: str = 'file'):
        self.redis_connection = redis_connection
        self.key_prefix = key_prefix

    def get_file(self, file_id):
        if not self.redis_connection.exists(f'{self.key_prefix}:{file_id}:meta'):
            return None

        if not self.redis_connection.json().get(f'{self.key_prefix}:{file_id}:meta', '$.status')[0] == 'uploaded':
            return None

        return decode_file(self.redis_connection.get(f'{self.key_prefix}:{file_id}:data'))

    def store_file(self, file_id, given_file, metadata=None):
        if not self.redis_connection.json().set(f'{self.key_prefix}:{file_id}:meta', '$', metadata):
            return 0

        return 1 if self.redis_connection.set(f'{self.key_prefix}:{file_id}:data', encode_file(given_file)) else 0

    def delete_file(self, file_id):
        keys: list = self.redis_connection.keys(f'{self.key_prefix}:{file_id}:*')
        if not keys:
            return 0

        return self.redis_connection.delete(*keys)

    def get_file_metadata(self, file_id):
        return self.redis_connection.json().get(f'{self.key_prefix}:{file_id}:meta', '$')[0]

    def exists(self, file_id):
        return self.redis_connection.exists(f'{self.key_prefix}:{file_id}:meta')

    # TODO Implement Chunked Dataset Transfer
    def get_chunk(self, file_id, chunk_address):
        pass

    def start_upload(self, file_id):
        pass

    def upload_chunk(self, file_id, chunk):
        pass

    def finish_upload(self, file_id):
        pass

def encode_file(given_file):
    return base64.b64encode(given_file).decode('utf-8')

def decode_file(given_file):
    return base64.b64decode(given_file.encode('utf-8'))

class RedisModelStorage:
    def __init__(self, redis_connection: Redis, key_prefix: str = 'model'):
        self.redis_connection = redis_connection
        self.key_prefix = key_prefix