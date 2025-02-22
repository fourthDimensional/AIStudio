import logging
import re
import redis
import jsonpickle
import hmac
import hashlib
import os
import json

import pandas as pd

from pathlib import Path
import uuid as uid

MODEL_KEY_FORMAT = "model:{}:{}:data"

MAX_NETWORKS_PER_PERSON = 30
MODEL_PATH = '$'
MAX_LOGS = 15
PROFILER_DIR = (Path(__file__).resolve().parent.parent.parent.parent / 'profiles').resolve()
SECRET_KEY = os.getenv("HMAC_SECRET")

# Configuration for Redis connection
redis_host: str = 'localhost'
redis_port: int = 6379
redis_db: int = 0

# Create a Redis connection using environment variables
redis_client = redis.Redis(
    host=os.getenv('REDIS_HOST', redis_host),
    port=int(os.getenv('REDIS_PORT', str(redis_port))),
    decode_responses=True
)

logger = logging.getLogger(__name__)

"""
Mostly up-to-date code, but some parts are deprecated and should be removed.

Various utility functions, including checking naming conventions, generating UUIDs, verifying signatures, and more.

Reasonable test coverage, but more will be needed once the deprecated code is removed.

Future Plans:
- Organize into different files
"""


def generate_uuid():
    """
    Generate a UUID using the Python UUID library.

    :return: A UUID.
    """
    uuid = str(uid.uuid4())
    attempts = 0
    while redis_client.exists(f'uuid:{uuid}'):
        if attempts > 20:
            raise Exception('UUID generation failed')

        uuid = str(uid.uuid4())

    redis_client.set(f'uuid:{uuid}', 1, ex=3600)

    return uuid


def check_naming_convention(string):
    """
    Check if a given string follows a specific naming convention.

    The naming convention is all lowercase letters with underscores separating words.

    :param string: The string to be checked.
    :return: True if the string follows the convention, False otherwise.
    """
    pattern = r'^[a-z]+(_[a-z]+)*$'
    if re.match(pattern, string):
        return True
    return False


def get_uuid(api_key, redis_server=redis_client):
    """
    Retrieve the UUID associated with a given API key from Redis.

    Logs a message indicating the action of querying for the UUID.

    :param redis_server:
    :param api_key: The API key for which the UUID is being queried.
    :return: The UUID associated with the provided API key.
    """
    logging.info('QUERYing for UUID by API KEY')
    return redis_server.json().get("api_key:{}".format(api_key), '$.uuid')[0]


def verify_signature(data, signature, secret_key):
    """
    Verify the HMAC signature of given data using the provided secret key.

    If the Flask environment is development, it also cleans up old logs before returning True unconditionally.

    :param data: The data for which the signature was generated.
    :param signature: The signature to be verified.
    :param secret_key: The secret key used to generate the signature.
    :return: True if the signature is valid or if in development mode, otherwise False.
    """
    if os.getenv("FLASK_ENV") == 'development':
        cleanup_old_logs(PROFILER_DIR, MAX_LOGS)
        return True
    expected_signature = hmac.new(secret_key.encode(), data.encode(), hashlib.sha256).hexdigest().encode()
    logging.info('Verifying HMAC signature on serialized model')
    print(data.encode())
    return hmac.compare_digest(expected_signature, signature)


def convert_to_dataframe(dataset_path):
    """
    Convert a dataset file to a pandas DataFrame.

    Attempts to read the file using multiple encodings until successful. Drops rows with any missing values.

    :param dataset_path: The path to the dataset file.
    :return: A pandas DataFrame containing the dataset.
    """
    dataframe_csv = None
    encodings = ["utf-8", "utf-8-sig", "iso-8859-1", "latin1", "cp1252"]
    for encoding in encodings:
        with open(dataset_path, 'r', encoding=encoding, errors='replace') as f:
            dataframe_csv = pd.read_csv(f)
            dataframe_csv = dataframe_csv.dropna(axis=0)
        break
    return dataframe_csv

def merge_lists(a, b):
    """
    Merge two lists or append an item to a list.

    If 'b' is a string, it is appended to 'a' as a single item. Otherwise, 'b' is extended to 'a'.

    :param a: The first list.
    :param b: The second list or item to be merged or appended.
    :return: The merged list.
    """
    return a + [b] if isinstance(b, str) else a + b


def cleanup_old_logs(profile_dir, max_logs):
    """
    Cleanup old logs in the specified directory, keeping only a set maximum number of the most recent logs.

    :param profile_dir: The directory containing profiler logs.
    :param max_logs: The maximum number of log files to keep.
    """
    # Get a list of profiler logs
    log_files = [f for f in os.listdir(profile_dir) if f.endswith('.prof')]

    # Sort logs by modification time (oldest first)
    log_files.sort(key=lambda x: os.path.getmtime(os.path.join(profile_dir, x)))

    # Calculate the number of logs to delete
    logs_to_delete = max(0, len(log_files) - max_logs)

    # Delete the oldest logs
    for i in range(logs_to_delete):
        file_to_delete = os.path.join(profile_dir, log_files[i])
        os.remove(file_to_delete)
        logging.info(f"Deleted old profiler log: {file_to_delete}")
