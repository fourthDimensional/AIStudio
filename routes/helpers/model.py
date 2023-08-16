import logging
import os
import numpy as np
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

    # move this shit elsewhere because there arent any modifications yet you dipshit
    column_count = model.process_columns(process_modifications=True)
    model.layers["Input"] = {}
    logging.info(len(column_count))
    for i in range(0, len(column_count) - 1):
        logging.info(i)
        model.layers["Input"][i] = layers.SpecialInput()

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
        self.layers = {}

        self.features = None
        self.labels = None
        self.column_hash = {}

        self.layer_count = 0

    def train(self):
        pass

    # TODO Revamp data modification system after the general functions are implemented
    def process_columns(self, process_modifications: bool):
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
        self.data_modifications.append(data_proc.ColumnDeletion(column_name))

    def add_deleted_column(self, column_name):
        for modification in self.data_modifications:
            if isinstance(modification, data_proc.ColumnDeletion) and str(modification) == column_name:
                self.data_modifications.remove(modification)

    def data_modification_exists(self, class_input, string_repr):
        for modification in self.data_modifications:
            if isinstance(modification, class_input) and str(modification) == string_repr:
                return True
        return

    # TODO Implement undo for this
    # TODO Do not allow layer or hyperparameter manipulation until this is done
    def specify_feature(self, column_name):
        self.data_modifications.append(data_proc.SpecifiedFeature(column_name))

    def add_layer(self, layer_type, vertical, position):
        logging.info(self.layers)
        if vertical in self.layers:
            if position in self.layers[vertical]:
                return False  # TODO Error handling here instead?
        else:
            self.layers[vertical] = {position: None}

        match layer_type:
            case "normalization":
                self.layers[vertical][position] = layers.Normalization()

        logging.info(self.layers)

        return True

    def remove_layer(self, vertical, position):
        try:
            self.layers[vertical].pop(position)
        except IndexError:
            return False
        return True

    def verify_layers(self):
        dataframe_csv = utils.convert_to_dataframe(self.dataset_path)

        if dataframe_csv is not None:
            column_names = dataframe_csv.columns.tolist()
        else:
            logging.error("Failed to read the dataset.")

        for data_mod in self.data_modifications:
            dataframe_csv = data_mod.process(dataframe_csv)

        inputs = {}

        for name, column in dataframe_csv.items():
            dtype = column.dtype
            if dtype == object:
                dtype = tf.string
            else:
                dtype = tf.float32

            inputs[name] = tf.keras.Input(shape=(1,), name=name, dtype=dtype)

        logging.info(inputs)

        # vertical_inputs = [[] for _ in range(1, len(self.layers) - 1)]
        # for i in range(0, len(vertical_inputs)):
        #     vertical_inputs[i] = [None for _ in self.layers[i]]

        vertical_inputs = {}

        logging.info(vertical_inputs)
        logging.info(self.layers)

        sym_input_tensors = [inputs[key] for key in inputs]
        logging.info(sym_input_tensors)

        errors = []
        i_count = 0
        for vertical in self.layers:
            for position in self.layers[vertical]:
                logging.info(self.layers[vertical][position])
                logging.info(vertical_inputs)
                match self.layers[vertical][position].name:
                    case 'input':
                        vertical_offset = self.layers[vertical][position].next_vertical
                        positional_offset = self.layers[vertical][position].offset

                        if not vertical_inputs:
                            vertical_inputs[vertical_offset] = {}
                        if positional_offset not in vertical_inputs[vertical_offset]:
                            vertical_inputs[vertical_offset][positional_offset] = []

                        vertical_inputs[vertical_offset][positional_offset].append(sym_input_tensors[i_count])
                        i_count += 1
                # if self.layers[vertical][position].subsplit:
                #     split_output = tf.split()

    def __len__(self):
        pass

    def __str__(self):
        pass
