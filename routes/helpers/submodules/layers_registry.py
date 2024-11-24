import os

import keras
import tensorflow as tf
from class_registry import ClassRegistry

"""
Up-to-date Layer Registry and Class Definition Code

Needs class definitions and implementations for the following layers:
- BatchNormalization
- Dropout
- GaussianNoise
- Flatten
- Activation
- Embedding
- Identity
- LSTM
- GRU

Future Plans:
- Add more layer types
"""

class SplitLayer(keras.Layer):
    def __init__(self, num_or_size_splits, axis=-1):
        super(SplitLayer, self).__init__()
        self.num_or_size_splits = num_or_size_splits
        self.axis = axis

    def call(self, inputs, **kwargs):
        return tf.split(inputs, num_or_size_splits=self.num_or_size_splits, axis=self.axis)


class UniversalSplitLayer:
    def __init__(self, num_or_size_splits, axis=-1):
        """
        A universal split layer compatible with TensorFlow and PyTorch.

        This class will need to be passed in as a custom object when deserializing the model.
        https://keras.io/guides/serialization_and_saving/

        :param num_or_size_splits: Number or size of splits
        :param axis: Axis to split along
        """
        self.num_or_size_splits = num_or_size_splits
        self.axis = axis

    def split(self, tensor):
        return SplitLayer(num_or_size_splits=self.num_or_size_splits, axis=self.axis)(tensor)


class LayerSkeleton:
    def __init__(self):
        self.layer_name = 'skeleton'

        self.hyperparameters = {}

    def instance_layer(self, previous_layer):
        raise NotImplementedError

    def get_hyperparameters(self):
        return self.hyperparameters

    def modify_hyperparameters(self, hyperparameters):
        for key, value in hyperparameters.items():
            if key in self.hyperparameters:
                self.hyperparameters[key] = value
            else:
                raise KeyError(f"Hyperparameter {key} not found in layer {self.layer_name}")

    def get_hyperparameter_ranges(self):
        raise NotImplementedError

    def get_default_hyperparameters(self):
        raise NotImplementedError

    def suggested_hyperparameter(self):
        raise NotImplementedError


layer_registry = ClassRegistry[LayerSkeleton]()


@layer_registry.register('input')
class InputLayer(LayerSkeleton):
    def __init__(self, **kwargs):
        super().__init__()
        self.layer_name = 'input'

        self.hyperparameters['input_size'] = kwargs['input_size']

    def instance_layer(self, _previous_layer):
        layer = tf.keras.layers.Input(shape=(self.hyperparameters['input_size'],))
        return layer


@layer_registry.register('dense')
class DenseLayer(LayerSkeleton):
    def __init__(self, **kwargs):
        super().__init__()
        self.layer_name = 'dense'

        self.hyperparameters['units'] = kwargs['units']

    def instance_layer(self, previous_layer):
        layer = tf.keras.layers.Dense(units=self.hyperparameters['units'])
        return layer(previous_layer)

    def get_default_hyperparameters(self):
        return {
            'units': 10,
            'activation': 'relu',
            'use_bias': True,
            'kernel_initializer': 'glorot_uniform',
            'bias_initializer': 'zeros',
            'kernel_regularizer': None,
            'bias_regularizer': None,
            'activity_regularizer': None,
            'kernel_constraint': None,
            'bias_constraint': None,
            'lora_rank': None,
        }

    def get_hyperparameter_ranges(self):
        return {
            'units': (1, 1_000_000_000),
            'activation': ['relu', 'sigmoid', 'tanh', 'softmax', 'softplus', 'softsign', 'selu', 'elu', 'exponential', 'linear'],
            'use_bias': [True, False],
            'kernel_initializer': ['glorot_uniform', 'glorot_normal', 'he_normal', 'he_uniform', 'lecun_normal', 'lecun_uniform'],
            'bias_initializer': ['zeros', 'ones', 'constant', 'uniform', 'normal', 'truncated_normal', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform', 'lecun_normal', 'lecun_uniform'],
            'kernel_regularizer': {'type': 'regularizer'},
            'bias_regularizer': {'type': 'regularizer'},
            'activity_regularizer': {'type': 'regularizer'},
            'kernel_constraint': {'type': 'constraint'},
            'bias_constraint': {'type': 'constraint'},
            'lora_rank': [None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        }


@layer_registry.register('batch_normalization')
class BatchNormalizationLayer(LayerSkeleton):
    def __init__(self, **kwargs):
        super().__init__()
        self.layer_name = 'batch_normalization'

    def instance_layer(self, previous_layer):
        layer = tf.keras.layers.BatchNormalization(axis=-1)
        return layer(previous_layer)


@layer_registry.register('dropout')
class DropoutLayer(LayerSkeleton):
    pass


@layer_registry.register('gaussian_noise')
class GaussianNoiseLayer(LayerSkeleton):
    pass


@layer_registry.register('flatten')
class FlattenLayer(LayerSkeleton):
    pass


@layer_registry.register('activation')
class ActivationLayer(LayerSkeleton):
    pass


@layer_registry.register('embedding')
class EmbeddingLayer(LayerSkeleton):
    pass


@layer_registry.register('identity')
class IdentityLayer(LayerSkeleton):
    pass


@layer_registry.register('lstm')
class LSTMLayer(LayerSkeleton):
    pass


@layer_registry.register('gru')
class GRULayer(LayerSkeleton):
    pass
