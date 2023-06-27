import pickle
import os
import re
import pandas as pd
import logging

logging.basicConfig(format='%(levelname)s (%(asctime)s): %(message)s (Line: %(lineno)d [%(filename)s])',
                    datefmt='%I:%M:%S %p',
                    level=logging.DEBUG)


def check_naming_convention(string):
    pattern = r'^[a-z]+(_[a-z]+)*$'
    if re.match(pattern, string):
        return True
    return False


def load_model_from_file(given_id, api_key, upload_folder):
    model_path = os.path.join(upload_folder, "model_{}_{}.pk1".format(given_id, api_key))

    with open(model_path, 'rb') as file:
        return pickle.load(file)


def convert_text_file_to_list(file_path):
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
