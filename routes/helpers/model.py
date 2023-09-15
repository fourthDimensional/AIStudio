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

    def offset_layer(self, vertical, position, new_offset, positional_offset):
        try:
            if isinstance(self.layers[vertical][position], layers.SpecialInput):
                self.layers[vertical][position].offset = new_offset
                return 1

            self.layers[vertical][position].offset[positional_offset] = new_offset
        except KeyError or IndexError:
            return 0

    def subsplit_layer(self, vertical, position, new_subsplit, positional_offset):
        try:
            if isinstance(self.layers[vertical][position], layers.SpecialInput):
                self.layers[vertical][position].subsplit = new_subsplit
                return 1

            self.layers[vertical][position].offset[positional_offset] = new_subsplit
        except KeyError:
            return 0

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

        # vertical_inputs = [[] for _ in range(1, len(self.layers) - 1)]
        # for i in range(0, len(vertical_inputs)):
        #     vertical_inputs[i] = [None for _ in self.layers[i]]

        vertical_inputs = {}

        sym_input_tensors = [inputs[key] for key in inputs]

        errors = []
        i_count = 0
        for vertical in self.layers:
            for position in self.layers[vertical]:
                layer_object = self.layers[vertical][position]
                logging.info(layer_object)

                if layer_object.name == 'input':
                    vertical_offset = layer_object.next_vertical
                    positional_offset = layer_object.offset

                    if not vertical_inputs:
                        vertical_inputs[vertical_offset] = {}
                    if positional_offset not in vertical_inputs[vertical_offset]:
                        vertical_inputs[vertical_offset][positional_offset] = []

                    vertical_inputs[vertical_offset][positional_offset].append(sym_input_tensors[i_count])
                    i_count += 1

                    logging.info(vertical_inputs)

                    continue

                if not len(layer_object.offset) == len(layer_object.subsplit) == len(layer_object.next_vertical):
                    errors.append({'layer_mapping_mismatch': layer_object.name})

                # Attempts to combine and add layers.
                logging.info(vertical_inputs)
                logging.info(vertical)
                logging.info(position)
                if len(vertical_inputs[vertical][position]) == 0:
                    errors.append({'invalid_layer_input': layer_object.name})
                elif len(vertical_inputs[vertical][position]) == 1:
                    real_layer = layer_object.create_instanced_layer(vertical_inputs[vertical][position][0])
                elif len(vertical_inputs[vertical][position]) > 1:
                    combined_layer = tf.keras.layers.Concatenate(axis=1)(vertical_inputs[vertical][position])
                    real_layer = layer_object.create_instanced_layer(combined_layer)
                    logging.info(real_layer)
                else:
                    pass

                def model_builder(hp):
                    hp_units = hp.Int('units', min_value=5, max_value=50, step=1)
                    hp_units2 = hp.Int('units2', min_value=5, max_value=50, step=1)
                    hp_units3 = hp.Int('units3', min_value=5, max_value=50, step=1)

                    dense_layer_2 = tf.keras.layers.Dense(hp_units, activation="relu")(real_layer)
                    dense_layer_3 = tf.keras.layers.Dense(hp_units2, activation="relu")(dense_layer_2)
                    dense_layer = tf.keras.layers.Dense(hp_units3, activation="relu")(dense_layer_3)
                    output = tf.keras.layers.Dense(1)(dense_layer)
                    tmodel = tf.keras.Model(sym_input_tensors, output)
                    # tf.keras.utils.plot_model(model=tmodel, rankdir="LR", dpi=72, show_shapes=True)
                    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
                    tmodel.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
                                   loss='mean_absolute_error',
                                   metrics=['accuracy'])

                    return tmodel

                tuner = keras_tuner.BayesianOptimization(model_builder,
                                                         objective='val_loss',
                                                         max_trials=100,
                                                         directory='testing',
                                                         project_name='001')

                stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

                train_y = self.data_modifications[2].get_column(utils.convert_to_dataframe(self.dataset_path))
                dataframe_new = [tf.constant(dataframe_csv[col].values) for col in dataframe_csv]

                tuner.search(dataframe_new, train_y, epochs=50, validation_split=0.2, callbacks=[stop_early])

                # Get the optimal hyperparameters
                best_hps = tuner.get_best_hyperparameters(num_trials=30)[0]

                print(f"""
                The hyperparameter search is complete. The optimal number of units in the first densely-connected
                layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
                is {best_hps.get('learning_rate')}.
                """)

                tmodel = tuner.hypermodel.build(best_hps)
                tmodel.fit(x=dataframe_new, y=train_y, epochs=200, validation_split=0.2)

                test_x = [tf.constant(dataframe_csv[col]) for col in dataframe_csv]
                test_y = self.data_modifications[2].get_column(utils.convert_to_dataframe(self.dataset_path))
                predictions = tmodel.predict(test_x)

                predictions_flat = predictions.flatten()
                y_test_flat = test_y.values.flatten()

                mae = np.mean(np.abs(predictions_flat - y_test_flat))

                plt.figure(figsize=(12, 12))
                sns.jointplot(x='True Values', y='Predictions',
                              data=pandas.DataFrame({'True Values': y_test_flat, 'Predictions': predictions_flat}),
                              kind="reg", truncate=False, color='m')
                plt.show()

                sns.pairplot(
                    utils.convert_to_dataframe(self.dataset_path)[self.process_columns(process_modifications=False)],
                    diag_kind='kde')
                plt.show()

                if not layer_object.offset:  # offset empty
                    pass
                elif layer_object > 1:
                    vertical_offset = layer_object.next_vertical
                    positional_offset = layer_object.offset

                    if not vertical_inputs:
                        vertical_inputs[vertical_offset] = {}
                    if positional_offset not in vertical_inputs[vertical_offset]:
                        vertical_inputs[vertical_offset][positional_offset] = []

                # if self.layers[vertical][position].subsplit:
                #     split_output = tf.split()

    def __len__(self):
        pass

    def __str__(self):
        pass
