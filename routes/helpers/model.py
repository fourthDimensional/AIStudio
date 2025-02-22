import os
import matplotlib.pyplot as plt
import pandas
import numpy as np
import tensorflow as tf

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from sklearn.model_selection import train_test_split

from routes.helpers.submodules import data_proc, utils
import logging
import re
import jsonpickle
import json
import hashlib

from routes.helpers.submodules.utils import generate_uuid

logger = logging.getLogger(__name__)

"""
Up-to-date Model wrapper code

This code is used to create a model wrapper that contains the data processing engine, compiler, and layer manipulator.
"""


class ModelWrapper:
    """
    A wrapper class for the model that contains the data processing engine, compiler, and layer manipulator.

    Attributes:
        data_processing_engine (DataProcessingEngine): Handles the data modification pipeline.
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