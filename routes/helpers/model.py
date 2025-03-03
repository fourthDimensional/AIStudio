import os
import matplotlib.pyplot as plt
import pandas
import numpy as np
import tensorflow as tf

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from sklearn.model_selection import train_test_split

from routes.helpers.submodules import data_proc, layers, utils
import logging
import re
import jsonpickle
import json
import hashlib

from routes.helpers.submodules.utils import generate_uuid

logger = logging.getLogger(__name__)

"""
Very old legacy code

Needs to be transitioned into the new model compilation and model wrapper classes
"""

def create_model(file_path, name, visual_name, network_type):
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

    model = Model(name, visual_name, network_type, file_path)

    column_count = model.process_columns(process_modifications=False)
    model.layers["Input"] = {}
    for i in range(0, len(column_count)):
        model.layers["Input"][i] = layers.SpecialInput()

    return {'info': 'Model created successfully'}, model


class ModelWrapper:
    """
    A wrapper class for the model that contains the data processing engine, hyperparameter manager, and layer manipulator.

    Attributes:
        data_processing_engine (DataProcessingEngine): Handles the data modification pipeline.
        hyperparameter_manager (HyperparameterManager): Manages hyperparameters for layers, data modifications, and optimization algorithms.
        layer_manipulator (LayerManipulator): Keeps track of the current layer structure.
        compiler (ModelCompiler): Compiles the model into a form that can be trained and evaluated.
    """

    def __init__(self, data_processing_engine, layer_manipulator, compiler):
        """
        Initializes the ModelWrapper with the given components.

        Args:
            data_processing_engine (DataProcessingEngine): The data processing engine.
            layer_manipulator (LayerManipulator): The layer manipulator.
        """
        self.data_processing_engine = data_processing_engine
        self.layer_manipulator = layer_manipulator
        self.compiler = compiler

        # generate a unique identifier for the model
        self.uuid = generate_uuid()

    def serialize(self):
        """
        Serializes the model wrapper.

        Returns:
            str: The serialized model wrapper.
        """
        return json.loads(jsonpickle.encode(self))

    @classmethod
    def deserialize(cls, data):
        """
        Deserializes the model wrapper.

        Args:
            data (str): The serialized model wrapper.

        Returns:
            ModelWrapper: The deserialized model wrapper.
        """
        return jsonpickle.decode(json.dumps(data))

    def deregister(self, redis_connection):
        """
        Deregisters the model from the redis database.
        """
        redis_connection.delete(f"uuid:{self.uuid}")


class DataProcessingEngine:
    """
    Handles the data modification pipeline.

    Attributes:
        modifications (list): A list of data modifications to apply.
        input_fields (list): A list of input fields from the dataset.
        label_columns (list): A list of label columns from the dataset.
    """

    def __init__(self):
        """
        Initializes the DataProcessingEngine with an empty list of modifications, input fields, and label columns.
        """
        self.modifications = []
        self.input_fields = []
        self.label_columns = []

    def add_modification(self, modification):
        """
        Adds a data modification to the pipeline.

        Args:
            modification: The data modification to add.
        """
        self.modifications.append(modification)

    def process_data(self, data):
        """
        Processes the data through the modification pipeline.

        Args:
            data: The data to process.
        """
        inputs = None
        fields = None

        for modification in self.modifications:
            modification.adapt(data)
            data = modification.apply(data)

        return data


    def clear_modifications(self):
        """
        Clears all data modifications from the pipeline.
        """
        self.modifications = []

    def set_input_fields(self, dataframe_head):
        """
        Sets the input fields from the head of the dataset dataframe.

        Args:
            dataframe_head (pandas.DataFrame): The head of the dataset dataframe.
        """
        self.input_fields = dataframe_head.columns.tolist()

    def add_label_column(self, column):
        """
        Adds a specific column as a label.

        Args:
            column (str): The column to add as a label.
        """
        if column in self.input_fields and column not in self.label_columns:
            self.label_columns.append(column)

    def add_label_columns(self, columns):
        """
        Adds a list of columns as labels.

        Args:
            columns (list): The list of columns to add as labels.
        """
        for column in columns:
            self.add_label_column(column)

    def add_label_columns_by_regex(self, regex):
        """
        Adds columns as labels based on a regex expression.

        Args:
            regex (str): The regex expression to match column names.
        """
        pattern = re.compile(regex)
        for column in self.input_fields:
            if pattern.match(column):
                self.add_label_column(column)

    def separate_labels(self, data):
        """
        Separates the labels from the data.

        Args:
            data (pandas.DataFrame): The data to separate labels from.

        Returns:
            tuple: A tuple containing the fields dataframe and the labels dataframe.
        """
        data = self.process_data(data)

        pattern = '|'.join([f'^{col}' for col in self.label_columns])

        fields = data.drop(columns=data.filter(regex=pattern).columns)
        labels = data[data.filter(regex=pattern).columns]
        return fields, labels

    def separate_labels_with_split(self, data, test_size=0.2, random_state=None):
        """
        Separates the labels from the data and performs a train-test split.

        Args:
            data (pandas.DataFrame): The data to separate labels from.
            test_size (float): The proportion of the dataset to include in the test split.
            random_state (int): Controls the shuffling applied to the data before applying the split.

        Returns:
            tuple: A tuple containing the train and test dataframes for fields and labels.
        """

        fields, labels = self.separate_labels(data)
        fields_train, fields_test, labels_train, labels_test = train_test_split(
            fields, labels, test_size=test_size, random_state=random_state
        )
        return fields_train, fields_test, labels_train, labels_test

    def get_version(self):
        """
        Returns a deterministic number representing the current state of the data processing engine.
        Returns:
            int: The version number.
        """
        state = {
            'modifications': [str(mod) for mod in self.modifications],
            'input_fields': self.input_fields,
            'label_columns': self.label_columns
        }
        state_str = str(state).encode('utf-8')
        version_hash = hashlib.md5(state_str).hexdigest()
        return int(version_hash, 16)


class LayerManipulator:
    """
    Keeps track of the current layer structure.

    Attributes:
        layers (dict): A dictionary of layers in the model, stored in a two-dimensional form.
    """

    def __init__(self):
        """
        Initializes the LayerManipulator with an empty dictionary of layers.
        """
        self.layers = {}

        self.current_x_position = 0
        self.current_y_position = 0

    def add_layer(self, layer, x_position, y_position):
        """
        Adds a layer to the model at the specified position.

        Args:
            layer: The layer to add.
            x_position (int): The x-coordinate of the layer.
            y_position (int): The y-coordinate of the layer.
        """
        if x_position not in self.layers:
            self.layers[x_position] = {}

        # No subsplit, so output location is just the next layer with no y positional offset or subsplit
        output_location = []

        self.layers[x_position][y_position] = {'layer': layer, 'outputs': output_location}

    def remove_layer(self, x_position, y_position):
        """
        Removes a layer from the model at the specified position.

        Args:
            x_position (int): The x-coordinate of the layer.
            y_position (int): The y-coordinate of the layer.
        """
        if x_position in self.layers:
            if y_position in self.layers[x_position]:
                del self.layers[x_position][y_position]

    def get_layers(self):
        """
        Gets the dictionary of layers in the model.

        Returns:
            dict: The dictionary of layers.
        """
        return self.layers

    def clear_layers(self):
        """
        Clears all layers from the model.
        """
        self.layers = {}

    def point_layer(self, x_position, y_position, output_x_position, output_y_position, subsplit=1):
        """
        Points a layer to another layer in the model.

        Args:
            x_position (int): The x-coordinate of the layer to point from.
            y_position (int): The y-coordinate of the layer to point from.
            output_x_position (int): The x-coordinate of the layer to point to.
            output_y_position (int): The y-coordinate of the layer to point to.
            subsplit (int): The subsplit size
        """
        if x_position in self.layers and y_position in self.layers[x_position]:
            self.layers[x_position][y_position]['outputs'].append([subsplit, output_x_position, output_y_position])

    def forward_layer(self, x_position, y_position):
        """
        Forwards a layer to the next layer in the model.

        Args:
            x_position (int): The x-coordinate of the layer to forward.
            y_position (int): The y-coordinate of the layer to forward.
        """
        if x_position in self.layers and y_position in self.layers[x_position]:
            self.layers[x_position][y_position]['outputs'].append([0, x_position + 1, 0])

    def get_layer_hyperparameters(self, x_position, y_position):
        """
        Gets the hyperparameters of a layer at the specified position.

        Args:
            x_position (int): The x-coordinate of the layer.
            y_position (int): The y-coordinate of the layer.

        Returns:
            dict: The hyperparameters of the layer.
        """
        if x_position in self.layers and y_position in self.layers[x_position]:
            return self.layers[x_position][y_position]['layer'].get_hyperparameters()
        return None

    def set_layer_hyperparameter(self, x_position, y_position, hyperparameter, value):
        """
        Sets a hyperparameter of a layer at the specified position.

        Args:
            x_position (int): The x-coordinate of the layer.
            y_position (int): The y-coordinate of the layer.
            hyperparameter (str): The hyperparameter to set.
            value: The value to set the hyperparameter to.
        """
        if x_position in self.layers and y_position in self.layers[x_position]:
            self.layers[x_position][y_position]['layer'].hyperparameters[hyperparameter] = value

    def get_layer_hyperparameter(self, x_position, y_position, hyperparameter):
        """
        Gets a specific hyperparameter of a layer at the specified position.

        Args:
            x_position (int): The x-coordinate of the layer.
            y_position (int): The y-coordinate of the layer.
            hyperparameter (str): The hyperparameter to get.

        Returns:
            The value of the hyperparameter.
        """
        if x_position in self.layers and y_position in self.layers[x_position]:
            return self.layers[x_position][y_position]['layer'].hyperparameters.get(hyperparameter)
        return None

    def get_layer_hyperparameter_ranges(self, x_position, y_position):
        """
        Gets the ranges of acceptable values for hyperparameters of a layer at the specified position.

        Args:
            x_position (int): The x-coordinate of the layer.
            y_position (int): The y-coordinate of the layer.

        Returns:
            dict: The ranges of acceptable values for the hyperparameters of the layer.
        """
        if x_position in self.layers and y_position in self.layers[x_position]:
            return self.layers[x_position][y_position]['layer'].get_hyperparameter_ranges()
        return None

    def layer(self, layer):
        """
        Adds a layer to the model at the current position and forwards it.

        Args:
            layer: The layer to add.

        Returns:
            self: The LayerManipulator instance to allow chaining.
        """
        self.add_layer(layer, self.current_x_position, self.current_y_position)
        self.forward_layer(self.current_x_position, self.current_y_position)
        self.current_x_position += 1
        return self


class Model:
    def __init__(self, name, visual_name, network_type, dataset_path):
        self.name = name
        self.visual_name = visual_name
        self.type = network_type
        self.dataset_path = dataset_path

        self.data_modifications = []
        self.layers = {}

        self.features = None
        self.labels = None
        self.column_hash = {}

        self.layer_count = 0

        self.tuner_type = "Bayesian"

        self.feature_count = 0

    # TODO Revamp data modification system after the general functions are implemented
    def process_columns(self, process_modifications: bool):
        dataframe_csv = utils.convert_to_dataframe(self.dataset_path)

        if process_modifications:
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
        if horizontal in self.layers:
            if position in self.layers[horizontal]:
                return False  # TODO Error handling here instead?
        else:
            self.layers[horizontal] = {position: None}

        match layer_type:
            case "batch_normalization":
                self.layers[horizontal][position] = layers.BatchNormalization()
            case "string_lookup":
                self.layers[horizontal][position] = layers.StringLookup()
            case "dense":
                self.layers[horizontal][position] = layers.Dense()

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

        numeric_columns = dataframe_csv.columns[dataframe_csv.apply(lambda x: x.astype(str).str
                                                                    .contains(r'\d', na=False).all())]

        dataframe_csv[numeric_columns] = dataframe_csv[numeric_columns].replace(',', '', regex=True)
        dataframe_csv[numeric_columns] = dataframe_csv[numeric_columns].apply(pandas.to_numeric, errors='coerce')

        if dataframe_csv is not None:
            column_names = dataframe_csv.columns.tolist()
        else:
            logging.error("Failed to read the dataset.")

        for data_mod in self.data_modifications:
            dataframe_csv = data_mod.process(dataframe_csv)

        dataframe_csv.to_csv('test_plane.csv')

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

                    if horizontal_offset not in horizontal_inputs:
                        horizontal_inputs[horizontal_offset] = {}
                    if positional_offset not in horizontal_inputs[horizontal_offset]:
                        horizontal_inputs[horizontal_offset][positional_offset] = []

                    feature_name = sym_input_tensors[i_count].name
                    horizontal_inputs[horizontal_offset][positional_offset].append(
                        [sym_input_tensors[i_count], feature_name])
                    i_count += 1

                    continue

                new_origin_layers = []

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
                    instanced_layer_input = horizontal_inputs[layer_column][position][0][0]
                    names = horizontal_inputs[layer_column][position][0][1]
                    new_origin_layers = utils.merge_lists(new_origin_layers, names)
                elif len(horizontal_inputs[layer_column][position]) > 1:
                    instanced_inputs = [x[0] for x in horizontal_inputs[layer_column][position]]
                    try:
                        instanced_layer_input = tf.keras.layers.Concatenate(axis=1)(instanced_inputs)
                        new_origin_layers = [x[1] for x in horizontal_inputs[layer_column][position]]
                    except TypeError:
                        errors.append({'error_type': 'type_mismatch',
                                       'layer_type': layer_object.name,
                                       'layer_column': layer_column,
                                       'layer_position': position,
                                       'description': 'This layer has inputs of multiple types'})
                        return {'event': 'fatal_error', 'errors': errors}
                else:
                    pass

                if layer_object.type == 'preprocessing':
                    dataframe_csv = utils.convert_to_dataframe(self.dataset_path)

                    for each in self.data_modifications:
                        dataframe_csv = each.process(dataframe_csv)
                    instanced_layer = layer_object.create_instanced_layer(instanced_layer_input, dataframe_csv,
                                                                          new_origin_layers)
                else:
                    instanced_layer = layer_object.create_instanced_layer(instanced_layer_input)

                if not len(layer_object.offset) == len(layer_object.subsplit) == len(layer_object.next_horizontal):
                    errors.append({'layer_mapping_mismatch': layer_object.name})

                if len(layer_object.next_horizontal) == 0:  # can be subsplit or horizontal length too
                    output_tensors.append(instanced_layer)
                    continue
                elif len(layer_object.next_horizontal) > 1:
                    subsplits = tf.split(instanced_layer, num_or_size_splits=layer_object.subsplit, axis=1)
                    for index in range(len(subsplits)):
                        hori = layer_object.next_horizontal[index]
                        posi = layer_object.offset[index]
                        if hori not in horizontal_inputs:
                            horizontal_inputs[hori] = {}
                        if posi not in horizontal_inputs[hori]:
                            horizontal_inputs[hori][posi] = []
                        horizontal_inputs[hori][posi].append([subsplits[index], new_origin_layers])

                    continue
                else:
                    next_horizontal = layer_object.next_horizontal[0]
                    next_positional_offset = layer_object.offset[0]

                if next_horizontal not in horizontal_inputs:
                    horizontal_inputs[next_horizontal] = {}
                if next_positional_offset not in horizontal_inputs[next_horizontal]:
                    horizontal_inputs[next_horizontal][next_positional_offset] = []

                horizontal_inputs[next_horizontal][next_positional_offset].append([instanced_layer, new_origin_layers])

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

        lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
                                                    0.01,
                                                    decay_steps=149 * 1000,
                                                    decay_rate=1,
                                                    staircase=False)

        output = tf.keras.layers.Dense(self.feature_count, activation='sigmoid')(real_layer)
        tmodel = tf.keras.Model(sym_input_tensors, output)
        tmodel.compile(optimizer=tf.keras.optimizers.Adam(lr_schedule),
                       loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                       metrics=[tf.keras.metrics.BinaryCrossentropy()])
        tf.keras.utils.plot_model(tmodel, to_file=os.path.join("static/files", "model.png"), show_shapes=True,
                                  expand_nested=True,
                                  show_layer_activations=True, show_layer_names=True, rankdir="LR")

        train_y = feature_output

        x_train_true, x_test_true, y_train_true, y_test_true = train_test_split(dataframe_csv, train_y, test_size=.5,
                                                                                random_state=22)

        print(len(x_test_true), len(y_test_true))
        print(len(x_train_true), len(y_train_true))

        x_train_true = [tf.constant(x_train_true[col].values) for col in x_train_true]
        x_test_true = [tf.constant(x_test_true[col]) for col in x_test_true]

        tmodel.fit(x=x_train_true, y=y_train_true, epochs=50, validation_split=0.5,
                   callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_binary_crossentropy', patience=200)])

        predictions = tmodel.predict(x_test_true)

        predicted = tf.squeeze(predictions)
        predicted = np.array([1 if x >= 0.5 else 0 for x in predicted])
        actual = np.array(y_test_true)
        conf_mat = confusion_matrix(actual, predicted)
        displ = ConfusionMatrixDisplay(confusion_matrix=conf_mat)
        displ.plot()

        print("test")
        plt.show()

        # predictions_flat = predictions.flatten()
        # y_test_flat = test_y.values.flatten()
        #
        # tf.keras.utils.plot_model(tmodel, to_file="model.png", show_shapes=True, expand_nested=True,
        #                           show_layer_activations=True, show_layer_names=True, rankdir="LR")
        #
        # plt.figure(figsize=(12, 12))
        # sns.jointplot(x='True Values', y='Predictions',
        #               data=pandas.DataFrame({'True Values': y_test_flat, 'Predictions': predictions_flat}),
        #               kind="reg", truncate=False, color='m')
        # plt.show()
        #
        # sns.pairplot(
        #     utils.convert_to_dataframe(self.dataset_path)[self.process_columns(process_modifications=False)],
        #     diag_kind='kde')
        # plt.show()

        return {'layer_count': layer_count, 'input_count': input_count, 'errors': errors}

    def __len__(self):
        pass

    def __str__(self):
        pass
