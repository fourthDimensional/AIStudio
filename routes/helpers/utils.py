import logging
import re
import redis
import jsonpickle
import hmac
import hashlib
import os
import json

import pandas as pd

from logging.handlers import RotatingFileHandler
from pathlib import Path

MAX_NETWORKS_PER_PERSON = 30
MODEL_PATH = '$'
MAX_LOGS = 15
PROFILER_DIR = (Path(__file__).resolve().parent.parent.parent / 'profiles').resolve()
SECRET_KEY = os.getenv("HMAC_SECRET")
FLASK_ENV = os.getenv("FLASK_ENV")

redis_client = redis.Redis(
    host='redis-16232.c282.east-us-mz.azure.cloud.redislabs.com',
    port=16232,
    password='GmIvRW7nt8slcgNmm7gjaARC0rCsRA6y')


logger = logging.getLogger(__name__)


def check_naming_convention(string):
    pattern = r'^[a-z]+(_[a-z]+)*$'
    if re.match(pattern, string):
        return True
    return False


def get_uuid(api_key):
    logging.info('QUERYing for UUID by API KEY')
    return redis_client.json().get("api_key:{}".format(api_key), '$.uuid')[0]


def verify_signature(data, signature, secret_key):
    if FLASK_ENV == 'development':
        cleanup_old_logs(PROFILER_DIR, MAX_LOGS)
        return True
    expected_signature = hmac.new(secret_key.encode(), data.encode(), hashlib.sha256).hexdigest().encode()
    logging.info('Verifying HMAC signature on serialized model')
    return hmac.compare_digest(expected_signature, signature)


def fetch_model(api_key, model_id):
    logging.info('Fetching model from database')
    uuid = get_uuid(api_key)

    model_key = "model:{}:{}:data".format(uuid, model_id)
    signature_key = "model:{}:{}:signature".format(uuid, model_id)

    if not redis_client.exists(model_key):
        logging.info('Requested model does not exist')
        return None, 0

    serialized_model = json.dumps(redis_client.json().get(model_key, MODEL_PATH)[0])
    signature = redis_client.get(signature_key)

    if verify_signature(serialized_model, signature, SECRET_KEY):
        logging.info('Verified HMAC signature')
        try:
            logging.info('Attempting to decode serialized model')
            return jsonpickle.decode(serialized_model, keys=True), 1
        except Exception as e:
            logging.error(f'Failed to decode serialized model at {model_key}')
            return None, -1
    else:
        logging.warning(f'SECURITY - Serialized model at "{model_key}" has been modified')
        return None, -1


def store_model(api_key, model_id, model):
    uuid = get_uuid(api_key)

    model_key = "model:{}:{}:data".format(uuid, model_id)
    signature_key = "model:{}:{}:signature".format(uuid, model_id)
    serialized_model = jsonpickle.encode(model, keys=True)

    signature = hmac.new(SECRET_KEY.encode(), serialized_model.encode(), hashlib.sha256).hexdigest()

    redis_client.set(signature_key, signature)
    redis_client.json().set(model_key, MODEL_PATH, json.loads(serialized_model))


def convert_to_dataframe(dataset_path):
    dataframe_csv = None
    encodings = ["utf-8", "utf-8-sig", "iso-8859-1", "latin1", "cp1252"]
    for encoding in encodings:
        try:
            with open(dataset_path, 'r', encoding=encoding, errors='replace') as f:
                dataframe_csv = pd.read_csv(f)
                dataframe_csv.dropna(axis=0, inplace=True)
            break
        except Exception as e:
            logging.error('No valid encoding when trying to convert a csv file to a pandas dataframe: {}'.format(e))
            pass

    # TODO Add error processing here and in implementation.

    return dataframe_csv


# TODO Please stop making idiotic programming decisions
def check_id(given_id):
    if not given_id.isdigit():
        return False

    given_id = int(given_id)

    if given_id > MAX_NETWORKS_PER_PERSON:
        return False

    return True


def find_index_of_specific_class(given_list, specific_class):
    try:
        index = next(i for i, obj in enumerate(given_list) if isinstance(obj, specific_class))
        return index
    except StopIteration:
        return None


def merge_lists(a, b):
    return a + [b] if isinstance(b, str) else a + b


def cleanup_old_logs(profile_dir, max_logs):
    """
    Cleanup old profiler logs, keeping at most max_logs logs.
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
