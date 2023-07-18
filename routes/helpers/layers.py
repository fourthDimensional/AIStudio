import tensorflow as tf


class Layer:
    def __init__(self, layer_id, input_type, input_shape, previous_layer):
        self.layer_id = layer_id
        self.input_type = input_type
        self.input_shape = input_shape
        self.previous_layer = previous_layer

        self.next_layer = None

    def create_instanced_layer(self):
        pass

    def list_hyperparameters(self):
        pass

    def modify_hyperparameters(self):
        pass

    def get_default_hyperparameter(self):
        pass

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
