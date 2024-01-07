import logging
import os
import matplotlib.pyplot as plt
import pandas
import seaborn as sns
import numpy as np
import tensorflow as tf
import keras_tuner

from routes.helpers import data_proc, utils, layers

logging.basicConfig(format='%(levelname)s (%(asctime)s): %(message)s (Line: %(lineno)d [%(filename)s])',
                    datefmt='%I:%M:%S %p',
                    level=logging.INFO)


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

    column_count = model.process_columns(process_modifications=False)
    model.layers["Input"] = {}
    logging.info(len(column_count))
    for i in range(0, len(column_count)):
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

        self.data_modifications = []
        self.layers = {}

        self.features = None
        self.labels = None
        self.column_hash = {}

        self.layer_count = 0

        self.tuner_type = "Bayesian"

        self.feature_count = 0

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

        return columns

    def delete_column(self, column_name):
        old_index = self.process_columns(process_modifications=False).index(column_name)
        self.data_modifications.append(data_proc.ColumnDeletion(column_name))

        return old_index

    def add_deleted_column(self, column_name):
        for modification in self.data_modifications:
            if isinstance(modification, data_proc.ColumnDeletion) and str(modification) == column_name:
                self.data_modifications.remove(modification)

                return self.process_columns(process_modifications=True).index(column_name)

    def data_modification_exists(self, class_input, string_repr):
        for modification in self.data_modifications:
            if isinstance(modification, class_input) and str(modification) == string_repr:
                return True
        return

    # TODO Implement undo for this
    # TODO Do not allow layer or hyperparameter manipulation until this is done
    def specify_feature(self, column_name):
        old_index = self.process_columns(process_modifications=False).index(column_name)
        self.data_modifications.append(data_proc.SpecifiedFeature(column_name))

        return old_index

    def add_layer(self, layer_type, horizontal, position):
        logging.info(self.layers)
        if horizontal in self.layers:
            if position in self.layers[horizontal]:
                return False  # TODO Error handling here instead?
        else:
            self.layers[horizontal] = {position: None}

        match layer_type:
            case "normalization":
                self.layers[horizontal][position] = layers.Normalization()
            case "dense":
                self.layers[horizontal][position] = layers.Dense()

        logging.info(self.layers)

        return True

    def remove_layer(self, horizontal, position):
        try:
            self.layers[horizontal].pop(position)
        except IndexError:
            return False
        return True

    # def point_layer(self, horizontal, position, new_horizontal, positional_offset):
    #     if isinstance(self.layers[horizontal][position], layers.SpecialInput):
    #         self.layers[horizontal][position].next_horizontal = new_horizontal
    #         return 1
    #
    #     while len(self.layers[horizontal][position].next_horizontal) < positional_offset + 1:
    #         self.layers[horizontal][position].next_horizontal.append(0)
    #
    #     self.layers[horizontal][position].next_horizontal[positional_offset] = new_horizontal
    #
    # def offset_layer(self, horizontal, position, new_offset, positional_offset):
    #     if isinstance(self.layers[horizontal][position], layers.SpecialInput):
    #         self.layers[horizontal][position].offset = new_offset
    #         return 1
    #
    #     while len(self.layers[horizontal][position].offset) < positional_offset + 1:
    #         self.layers[horizontal][position].offset.append(0)
    #
    #     self.layers[horizontal][position].offset[positional_offset] = new_offset
    #
    #
    # def subsplit_layer(self, horizontal, position, new_subsplit, positional_offset):
    #     try:
    #         if isinstance(self.layers[horizontal][position], layers.SpecialInput):
    #             self.layers[horizontal][position].subsplit = new_subsplit
    #             return 1
    #
    #         self.layers[horizontal][position].offset[positional_offset] = new_subsplit
    #     except KeyError:
    #         return 0

    def point_layer(self, horizontal, position, start_range, end_range, new_horizontal, positional_offset):
        # TODO add subsplit range verification
        try:
            subsplit_size = end_range - start_range

            layer = self.layers[horizontal][position]

        except KeyError:
            return 0

        if isinstance(self.layers[horizontal][position], layers.SpecialInput):
            return self.layers[horizontal][position].update_layer_output(positional_offset, new_horizontal)

        return self.layers[horizontal][position].update_layer_output(subsplit_size, new_horizontal, positional_offset)

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

        horizontal_inputs = {}
        output_tensors = []

        sym_input_tensors = [inputs[key] for key in inputs]

        errors = []
        i_count = 0

        layer_count = 0
        input_count = 0
        for layer_column in self.layers:
            for position in self.layers[layer_column]:
                layer_object = self.layers[layer_column][position]
                layer_count += 1

                if layer_object.name == 'input':
                    input_count += 1

                    horizontal_offset = layer_object.next_horizontal
                    positional_offset = layer_object.offset

                    logging.info(horizontal_offset)
                    logging.info(positional_offset)
                    if horizontal_offset not in horizontal_inputs:
                        horizontal_inputs[horizontal_offset] = {}
                    if positional_offset not in horizontal_inputs[horizontal_offset]:
                        horizontal_inputs[horizontal_offset][positional_offset] = []

                    logging.info(sym_input_tensors[i_count].name)
                    horizontal_inputs[horizontal_offset][positional_offset].append(sym_input_tensors[i_count])
                    i_count += 1

                    logging.info(horizontal_inputs)

                    continue

                if layer_column not in horizontal_inputs:
                    horizontal_inputs[layer_column] = {}

                if position not in horizontal_inputs[layer_column]:
                    horizontal_inputs[layer_column][position] = []

                if len(horizontal_inputs[layer_column][position]) == 0:
                    errors.append({'error_type': 'invalid_layer_input',
                                   'layer_type': layer_object.name,
                                   'layer_column': layer_column,
                                   'layer_position': position,
                                   'description': 'This layer has no inputs'})
                elif len(horizontal_inputs[layer_column][position]) == 1:
                    instanced_layer_input = horizontal_inputs[layer_column][position][0]
                elif len(horizontal_inputs[layer_column][position]) > 1:
                    instanced_layer_input = tf.keras.layers.Concatenate(axis=1)(horizontal_inputs[layer_column][position])
                else:
                    pass

                if isinstance(layer_object, layers.Normalization):
                    dataframe_csv = utils.convert_to_dataframe(self.dataset_path)

                    for each in self.data_modifications:
                        dataframe_csv = each.process(dataframe_csv)
                    instanced_layer = layer_object.create_instanced_layer(instanced_layer_input, dataframe_csv)
                else:
                    instanced_layer = layer_object.create_instanced_layer(instanced_layer_input)

                if not len(layer_object.offset) == len(layer_object.subsplit) == len(layer_object.next_horizontal):
                    errors.append({'layer_mapping_mismatch': layer_object.name})

                if len(layer_object.next_horizontal) == 0:  # can be subsplit or horizontal length too
                    output_tensors.append(instanced_layer)
                    continue
                elif len(layer_object.next_horizontal) == 1:
                    next_horizontal = layer_object.next_horizontal[0]
                    next_positional_offset = layer_object.offset[0]
                elif len(layer_object.next_horizontal) > 1:
                    subsplits = tf.split(instanced_layer, num_or_size_splits=layer_object.subsplit, axis=1)
                    for index in range(len(subsplits)):
                        hori = layer_object.next_horizontal[index]
                        posi = layer_object.offset[index]
                        if hori not in horizontal_inputs:
                            horizontal_inputs[hori] = {}
                        if posi not in horizontal_inputs[hori]:
                            horizontal_inputs[hori][posi] = []
                        horizontal_inputs[hori][posi].append(subsplits[index])

                    continue

                if next_horizontal not in horizontal_inputs:
                    horizontal_inputs[next_horizontal] = {}
                if next_positional_offset not in horizontal_inputs[next_horizontal]:
                    horizontal_inputs[next_horizontal][next_positional_offset] = []

                horizontal_inputs[next_horizontal][next_positional_offset].append(instanced_layer)

                # Attempts to combine and add layers.
                logging.info([horizontal.name for horizontal in horizontal_inputs[layer_column][position]])
                logging.info(layer_object)
                logging.info(layer_column)
                logging.info(position)

        if len(output_tensors) == 0:
            errors.append('invalid_output_layer_s')
        elif len(output_tensors) == 1:
            real_layer = output_tensors[0]
        elif len(output_tensors) > 1:
            real_layer = tf.keras.layers.Concatenate(axis=1)(output_tensors)

        if self.feature_count == 0:
            pass  # should do error
            feature_output = None
        elif self.feature_count == 1:
            feature_index = utils.find_index_of_specific_class(self.data_modifications,
                                                               data_proc.SpecifiedFeature)
            feature_output = self.data_modifications[feature_index]
            feature_output = feature_output.get_column(utils.convert_to_dataframe(self.dataset_path))

        elif self.feature_count > 1:
            features = [feature.get_column(utils.convert_to_dataframe(self.dataset_path))
                        for feature in self.data_modifications
                        if isinstance(feature, data_proc.SpecifiedFeature)]

            feature_output = pandas.concat(features, axis=1)

        output = tf.keras.layers.Dense(self.feature_count)(real_layer)
        tmodel = tf.keras.Model(sym_input_tensors, output)
        tmodel.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                       loss='mean_absolute_error',
                       metrics=['accuracy'])
        tf.keras.utils.plot_model(tmodel, to_file="model.png", show_shapes=True, expand_nested=True,
                                  show_layer_activations=True, show_layer_names=True, rankdir="LR")

        return {'layer_count': layer_count, 'input_count': input_count, 'errors': errors}

    def __len__(self):
        pass

    def __str__(self):
        pass
