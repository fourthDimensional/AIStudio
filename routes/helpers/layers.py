import tensorflow as tf
from abc import ABC, abstractmethod


class Layer(ABC):
    def __init__(self, layer_id, input_type, input_shape, previous_layer):
        self.layer_id = layer_id
        self.input_type = input_type
        self.input_shape = input_shape
        self.previous_layer = previous_layer

        self.layer_map = {}

    @abstractmethod
    def create_instanced_layer(self):
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
    def __init__(self, layer_id, input_type, input_shape, previous_layer):
        super().__init__(layer_id, input_type, input_shape, previous_layer)

    def create_instanced_layer(self):
        return tf.keras.Input(shape=self.input_shape, dtype=self.input_type)

    def list_hyperparameters(self):
        pass

    def modify_hyperparameters(self):
        pass

    def get_default_hyperparameter(self):
        pass

    def suggested_hyperparameter(self):
        pass
