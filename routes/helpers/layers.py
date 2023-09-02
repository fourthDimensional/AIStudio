import tensorflow as tf
from abc import ABC, abstractmethod


class Layer(ABC):
    def __init__(self):
        self.name = 'default'
        self.input_size = []
        self.subsplit = []  # [] or [5, 5]
        self.next_vertical = []  # [] or [3, -1]
        self.offset = []  # [] or [1, 0]z

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


class SpecialInput:
    def __init__(self):
        self.name = 'input'
        self.next_vertical = 0
        self.offset = 0


class Normalization(Layer):
    def __init__(self):
        super().__init__()

        self.axis = 1

    def create_instanced_layer(self, previous_layer):
        return tf.keras.layers.Normalization(axis=self.axis)(previous_layer)

    def list_hyperparameters(self):
        pass

    def modify_hyperparameters(self):
        pass

    def get_default_hyperparameter(self):
        pass

    def suggested_hyperparameter(self):
        pass
