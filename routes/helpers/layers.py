import tensorflow as tf
from abc import ABC, abstractmethod


class Layer(ABC):
    def __init__(self):
        self.input_size = []
        self.subsplit = []  # [] or [5, 5]
        self.next_vertical = []  # [] or [3, -1]
        self.offset = []  # [] or [1, 0]

    @abstractmethod
    def create_instanced_layer(self, previous_layer, dataframe_csv=None):
        pass

    @abstractmethod
    def list_hyperparameters(self):
        pass

    @abstractmethod
    def modify_hyperparameters(self):
        pass

    @abstractmethod
    def get_default_hyperparameter(self):
        pass

    @abstractmethod
    def suggested_hyperparameter(self):
        pass


class Input(Layer):
    def __init__(self):
        super().__init__()

    def create_instanced_layer(self, dataframe_csv=None, previous_layer=None):
        inputs = {}

        # TODO Add more complex data types

        for name, column in dataframe_csv.items():
            dtype = column.dtype
            if dtype == object:
                dtype = tf.string
            else:
                dtype = tf.float32

            inputs[name] = tf.keras.Input(shape=(1,), name=name, dtype=dtype)

        return inputs

    def list_hyperparameters(self):
        raise NotImplementedError

    def modify_hyperparameters(self):
        raise NotImplementedError

    def get_default_hyperparameter(self):
        raise NotImplementedError

    def suggested_hyperparameter(self):
        raise NotImplementedError


class Normalization(Layer):
    def __init__(self, axis, input_size=0):
        super().__init__(input_size)

        self.axis = axis

    def create_instanced_layer(self, previous_layer, dataframe_csv=None):
        return tf.keras.layers.Normalization()

    def list_hyperparameters(self):
        pass

    def modify_hyperparameters(self):
        pass

    def get_default_hyperparameter(self):
        pass

    def suggested_hyperparameter(self):
        pass
