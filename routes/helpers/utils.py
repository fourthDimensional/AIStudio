import logging
import os
import pickle
import re

import pandas as pd

logging.basicConfig(format='%(levelname)s (%(asctime)s): %(message)s (Line: %(lineno)d [%(filename)s])',
                    datefmt='%I:%M:%S %p',
                    level=logging.DEBUG)

MAX_NETWORKS_PER_PERSON = 30


def check_naming_convention(string):
    pattern = r'^[a-z]+(_[a-z]+)*$'
    if re.match(pattern, string):
        return True
    return False


def load_model_from_file(given_id, api_key, upload_folder):
    model_path = os.path.join(upload_folder, "model_{}_{}.pk1".format(given_id, api_key))

    with open(model_path, 'rb') as file:
        return pickle.load(file)


def grab_api_keys():
    file_path = 'routes/helpers/apikeys.txt'

    with open(file_path, 'r') as file:
        lines = file.readlines()
        lines = [line.strip() for line in lines]
    return lines


def save(model, model_path):
    with open(model_path, 'wb') as file:
        pickle.dump(model, file)


def convert_to_dataframe(dataset_path):
    dataframe_csv = None
    encodings = ["utf-8", "utf-8-sig", "iso-8859-1", "latin1", "cp1252"]
    for encoding in encodings:
        try:
            with open(dataset_path, 'r', encoding=encoding, errors='replace') as f:
                dataframe_csv = pd.read_csv(f)
            break
        except Exception as e:
            logging.error('No valid encoding when trying to convert a csv file to a pandas dataframe: {}'.format(e))
            pass

    # TODO Add error processing here and in implementation.

    return dataframe_csv


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
