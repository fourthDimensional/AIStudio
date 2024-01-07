import tensorflow as tf
from abc import ABC, abstractmethod


class Layer(ABC):
    def __init__(self):
        self.name = 'default'
        self.input_size = []
        self.subsplit = []  # [] or [5, 5]
        self.next_horizontal = []  # [] or [3, -1]
        self.offset = []  # [] or [1, 0]

    def update_layer_output(self, subsplit_size, new_horizontal, offset):
        for i in range(len(self.next_horizontal)):
            if offset == self.offset[i] and new_horizontal == self.next_horizontal[i]:
                self.subsplit[i] = subsplit_size
                self.offset[i] = offset
                self.next_horizontal[i] = new_horizontal

                return 2

        self.subsplit.append(subsplit_size)
        self.offset.append(offset)
        self.next_horizontal.append(new_horizontal)

        return 1

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
        self.next_horizontal = 0
        self.offset = 0

    def update_layer_output(self, offset, next_horizontal):
        self.offset = offset
        self.next_horizontal = next_horizontal


class Normalization(Layer):
    def __init__(self):
        super().__init__()

        self.name = 'normalization'
        self.axis = -1

    def create_instanced_layer(self, previous_layer, data, features):
        layer = tf.keras.layers.Normalization(axis=self.axis)
        print(features)
        data = data.loc[:, features]
        layer.adapt(data)

        return layer(previous_layer)

    def list_hyperparameters(self):
        pass

    def modify_hyperparameters(self):
        pass

    def get_default_hyperparameter(self):
        pass

    def suggested_hyperparameter(self):
        pass


class Dense(Layer):
    def __init__(self):
        super().__init__()

        self.name = 'dense'
        self.activation = 'relu'
        self.use_bias = True
        self.units = 6

    def create_instanced_layer(self, previous_layer):
        return tf.keras.layers.Dense(units=self.units, activation=self.activation, use_bias=self.use_bias,
                                     )(previous_layer)

    def list_hyperparameters(self):
        pass

    def modify_hyperparameters(self):
        pass

    def get_default_hyperparameter(self):
        pass

    def suggested_hyperparameter(self):
        pass
