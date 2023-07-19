import logging
import os

import tensorflow as tf

from routes.helpers import data_proc, utils, layers

logging.basicConfig(format='%(levelname)s (%(asctime)s): %(message)s (Line: %(lineno)d [%(filename)s])',
                    datefmt='%I:%M:%S %p',
                    level=logging.DEBUG)


def create_model(file_path, name, visual_name, network_type, model_path):
    valid_network_types = ["regression", "classification"]

    # * Verifies that the inputs are supported/usable
    if network_type not in valid_network_types:
        return {'error': 'Unsupported network type'}

    if not os.path.exists(file_path):
        return {'error': 'Dataset does not exist; create one before trying to construct a model'}, 409

    if not utils.check_naming_convention(name):
        return {'error': 'Internal name not supported'}, 409

    if network_type == 'regression':
        pass
    elif network_type == 'classification':
        pass

    model = Model(name, visual_name, network_type, model_path, file_path)

    model.train()

    return [{'info': 'Model created successfully'}, model]


class Model:
    def __init__(self, name, visual_name, network_type, model_path, dataset_path):
        self.name = name
        self.visual_name = visual_name
        self.type = network_type
        self.dataset_path = dataset_path
        self.model_path = model_path

        self.network_count = 0
        self.data_modifications = []
        self.layers = []

        self.features = None
        self.labels = None
        self.column_hash = {}

    def train(self):
        dataframe_csv = utils.convert_to_dataframe(self.dataset_path)

        if dataframe_csv is not None:
            column_names = dataframe_csv.columns.tolist()
            return column_names
        else:
            print("Failed to read the dataset.")

        logging.info(self.data_modifications)
        for each in self.data_modifications:
            dataframe_csv = each.process(dataframe_csv)

        inputs = {}

        for name, column in dataframe_csv.items():
            dtype = column.dtype
            if dtype == object:
                dtype = tf.string
            else:
                dtype = tf.float32

            inputs[name] = tf.keras.Input(shape=(1,), name=name, dtype=dtype)

        return inputs

    # TODO Revamp data modification system after the general functions are implemented
    def process_columns(self, process_modifications):
        dataframe_csv = utils.convert_to_dataframe(self.dataset_path)

        if process_modifications:
            logging.info(self.data_modifications)
            for each in self.data_modifications:
                dataframe_csv = each.process(dataframe_csv)

        columns = dataframe_csv.columns.tolist()

        for x in range(0, len(columns)):
            self.column_hash[x] = columns[x]

        logging.info(self.column_hash)
        logging.info(self.data_modifications)

        return columns

    def delete_column(self, column_name):
        self.data_modifications.append(data_proc.Column_Deletion(column_name))

    def add_deleted_column(self, column_name):
        for modification in self.data_modifications:
            if isinstance(modification, data_proc.Column_Deletion) and str(modification) == column_name:
                self.data_modifications.remove(modification)

    def data_modification_exists(self, class_input, string_repr):
        for modification in self.data_modifications:
            if isinstance(modification, class_input) and str(modification) == string_repr:
                return True
        return

    def layer_exists(self, class_input, layer_id):
        for layer in self.layers:
            if isinstance(layer, class_input) and str(layer) == layer_id:
                return True
        return

    def pop_training_column(self, column_name):
        self.data_modifications.append(data_proc.Training_Column_Pop(column_name))

    def add_layer(self, layer_type):
        if layer_type == "input":
            self.layers.append([])

            dataframe_csv = utils.convert_to_dataframe(self.dataset_path)

            logging.info(self.data_modifications)
            for each in self.data_modifications:
                dataframe_csv = each.process(dataframe_csv)

            inputs = {}

            for name, column in dataframe_csv.items():
                dtype = column.dtype
                if dtype == object:
                    dtype = tf.string
                else:
                    dtype = tf.float32

                inputs[name] = tf.keras.Input(shape=(1,), name=name, dtype=dtype)

            logging.info(inputs)

    def remove_layer(self, layer_id):
        pass

    def __len__(self):
        pass

    def __str__(self):
        pass
