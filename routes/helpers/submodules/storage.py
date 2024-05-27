from redis import Redis
import base64
import os

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


# class DatasetAccessInterface:
#     """
#     Interface for accessing datasets from different sources. Composition based.
#     """
#     def __init__(self, dataset_id, chunk_size=1000 * 1000 * 256):
#         """
#         Constructor for the DatasetAccessInterface class.
#
#         :param chunk_size: The size of the chunks to create.
#         """
#         self.id = dataset_id
#         self.chunk_size = chunk_size
#
#     def get_dataset_metadata(self, access_point):
#         """
#         Get metadata for the dataset.
#
#         :return: Metadata for the dataset.
#         """
#         return access_point.get_file_metadata(self.id)
#
#
#     def get_chunk(self, access_point, key):
#         """
#         Get a chunk of the dataset.
#
#         :param access_point:
#         :param key: The key of the chunk to get.
#         :return: The chunk of the dataset.
#         """
#         return access_point.get_chunk(self.id, self.chunk_keys[key])
#
#     def get_chunk_keys(self):
#         """
#         Get the keys of the chunks of the dataset.
#
#         :return: The keys of the chunks of the dataset.
#         """
#         return list(self.chunk_keys.keys())
#
#     def get_chunk_addresses(self):
#         """
#         Get the addresses of the chunks of the dataset.
#
#         :return: The addresses of the chunks of the dataset.
#         """
#         return list(self.chunk_keys.values())
#
#     def get_chunk_count(self):
#         """
#         Get the number of chunks in the dataset.
#
#         :return: The number of chunks in the dataset.
#         """
#         return len(self.chunk_keys)
#
#     def get_chunk_size(self):
#         """
#         Get the size of the chunks of the dataset.
#
#         :return: The size of the chunks of the dataset.
#         """
#         return self.chunk_size

class RedisFileStorage:
    def __init__(self, redis_connection: Redis, key_prefix: str = 'file'):
        self.redis_connection = redis_connection
        self.key_prefix = key_prefix

    def get_file(self, file_id):
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

    def get_chunk(self, file_id, chunk_address):
        pass

    def start_upload(self, file_id):
        pass

    def upload_chunk(self, file_id, chunk):
        pass

    def finish_upload(self, file_id):
        pass

def encode_file(given_file):
    print('Encoding: ', given_file)
    return base64.b64encode(given_file).decode('utf-8')

def decode_file(given_file):
    print('Decoding: ', given_file)
    return base64.b64decode(given_file.encode('utf-8'))
