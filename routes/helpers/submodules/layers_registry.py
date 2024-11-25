import os

import keras
from class_registry import ClassRegistry

"""
Up-to-date Layer Registry and Class Definition Code

Needs class definitions and implementations for the following layers:

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
        layer = keras.layers.Input(shape=(self.hyperparameters['input_size'],))
        return layer


@layer_registry.register('dense')
class DenseLayer(LayerSkeleton):
    def __init__(self, **kwargs):
        super().__init__()
        self.layer_name = 'dense'

        self.hyperparameters['units'] = kwargs['units']

    def instance_layer(self, previous_layer):
        layer = keras.layers.Dense(units=self.hyperparameters['units'])
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
        layer = keras.layers.BatchNormalization(axis=-1)
        return layer(previous_layer)


@layer_registry.register('dropout')
class DropoutLayer(LayerSkeleton):
    def __init__(self, **kwargs):
        super().__init__()
        self.layer_name = 'dropout'
        self.hyperparameters['rate'] = kwargs.get('rate', 0.5)

    def instance_layer(self, previous_layer):
        layer = keras.layers.Dropout(rate=self.hyperparameters['rate'])
        return layer(previous_layer)

    def get_default_hyperparameters(self):
        return {
            'rate': 0.5,
        }

    def get_hyperparameter_ranges(self):
        return {
            'rate': (0.0, 1.0),
        }


@layer_registry.register('gaussian_noise')
class GaussianNoiseLayer(LayerSkeleton):
    def __init__(self, **kwargs):
        super().__init__()
        self.layer_name = 'gaussian_noise'
        self.hyperparameters['stddev'] = kwargs.get('stddev', 0.1)

    def instance_layer(self, previous_layer):
        layer = keras.layers.GaussianNoise(stddev=self.hyperparameters['stddev'])
        return layer(previous_layer)

    def get_default_hyperparameters(self):
        return {
            'stddev': 0.1,
        }

    def get_hyperparameter_ranges(self):
        return {
            'stddev': (0.0, 1.0),
        }


@layer_registry.register('flatten')
class FlattenLayer(LayerSkeleton):
    def __init__(self, **kwargs):
        super().__init__()
        self.layer_name = 'flatten'

    def instance_layer(self, previous_layer):
        layer = keras.layers.Flatten()
        return layer(previous_layer)


@layer_registry.register('activation')
class ActivationLayer(LayerSkeleton):
    def __init__(self, **kwargs):
        super().__init__()
        self.layer_name = 'activation'
        self.hyperparameters['activation'] = kwargs.get('activation', 'relu')

    def instance_layer(self, previous_layer):
        layer = keras.layers.Activation(activation=self.hyperparameters['activation'])
        return layer(previous_layer)

    def get_default_hyperparameters(self):
        return {
            'activation': 'relu',
        }

    def get_hyperparameter_ranges(self):
        return {
            'activation': ['relu', 'sigmoid', 'tanh', 'softmax', 'softplus', 'softsign', 'selu', 'elu', 'exponential', 'linear'],
        }


@layer_registry.register('embedding')
class EmbeddingLayer(LayerSkeleton):
    def __init__(self, **kwargs):
        super().__init__()
        self.layer_name = 'embedding'
        self.hyperparameters['input_dim'] = kwargs.get('input_dim', 100)
        self.hyperparameters['output_dim'] = kwargs.get('output_dim', 64)
        self.hyperparameters['input_length'] = kwargs.get('input_length', None)

    def instance_layer(self, previous_layer):
        layer = keras.layers.Embedding(
            input_dim=self.hyperparameters['input_dim'],
            output_dim=self.hyperparameters['output_dim'],
            input_length=self.hyperparameters['input_length']
        )
        return layer(previous_layer)

    def get_default_hyperparameters(self):
        return {
            'input_dim': 100,
            'output_dim': 64,
            'input_length': None,
        }

    def get_hyperparameter_ranges(self):
        return {
            'input_dim': (1, 1_000_000),
            'output_dim': (1, 1_000),
            'input_length': (1, 1_000),
        }


@layer_registry.register('identity')
class IdentityLayer(LayerSkeleton):
    def __init__(self, **kwargs):
        super().__init__()
        self.layer_name = 'identity'

    def instance_layer(self, previous_layer):
        return previous_layer


@layer_registry.register('lstm')
class LSTMLayer(LayerSkeleton):
    def __init__(self, **kwargs):
        super().__init__()
        self.layer_name = 'lstm'
        self.hyperparameters['units'] = kwargs.get('units', 50)
        self.hyperparameters['activation'] = kwargs.get('activation', 'tanh')
        self.hyperparameters['recurrent_activation'] = kwargs.get('recurrent_activation', 'sigmoid')
        self.hyperparameters['use_bias'] = kwargs.get('use_bias', True)
        self.hyperparameters['kernel_initializer'] = kwargs.get('kernel_initializer', 'glorot_uniform')
        self.hyperparameters['recurrent_initializer'] = kwargs.get('recurrent_initializer', 'orthogonal')
        self.hyperparameters['bias_initializer'] = kwargs.get('bias_initializer', 'zeros')
        self.hyperparameters['unit_forget_bias'] = kwargs.get('unit_forget_bias', True)
        self.hyperparameters['kernel_regularizer'] = kwargs.get('kernel_regularizer', None)
        self.hyperparameters['recurrent_regularizer'] = kwargs.get('recurrent_regularizer', None)
        self.hyperparameters['bias_regularizer'] = kwargs.get('bias_regularizer', None)
        self.hyperparameters['activity_regularizer'] = kwargs.get('activity_regularizer', None)
        self.hyperparameters['kernel_constraint'] = kwargs.get('kernel_constraint', None)
        self.hyperparameters['recurrent_constraint'] = kwargs.get('recurrent_constraint', None)
        self.hyperparameters['bias_constraint'] = kwargs.get('bias_constraint', None)
        self.hyperparameters['dropout'] = kwargs.get('dropout', 0.0)
        self.hyperparameters['recurrent_dropout'] = kwargs.get('recurrent_dropout', 0.0)
        self.hyperparameters['return_sequences'] = kwargs.get('return_sequences', False)
        self.hyperparameters['return_state'] = kwargs.get('return_state', False)
        self.hyperparameters['go_backwards'] = kwargs.get('go_backwards', False)
        self.hyperparameters['stateful'] = kwargs.get('stateful', False)
        self.hyperparameters['unroll'] = kwargs.get('unroll', False)

    def instance_layer(self, previous_layer):
        layer = keras.layers.LSTM(
            units=self.hyperparameters['units'],
            activation=self.hyperparameters['activation'],
            recurrent_activation=self.hyperparameters['recurrent_activation'],
            use_bias=self.hyperparameters['use_bias'],
            kernel_initializer=self.hyperparameters['kernel_initializer'],
            recurrent_initializer=self.hyperparameters['recurrent_initializer'],
            bias_initializer=self.hyperparameters['bias_initializer'],
            unit_forget_bias=self.hyperparameters['unit_forget_bias'],
            kernel_regularizer=self.hyperparameters['kernel_regularizer'],
            recurrent_regularizer=self.hyperparameters['recurrent_regularizer'],
            bias_regularizer=self.hyperparameters['bias_regularizer'],
            activity_regularizer=self.hyperparameters['activity_regularizer'],
            kernel_constraint=self.hyperparameters['kernel_constraint'],
            recurrent_constraint=self.hyperparameters['recurrent_constraint'],
            bias_constraint=self.hyperparameters['bias_constraint'],
            dropout=self.hyperparameters['dropout'],
            recurrent_dropout=self.hyperparameters['recurrent_dropout'],
            return_sequences=self.hyperparameters['return_sequences'],
            return_state=self.hyperparameters['return_state'],
            go_backwards=self.hyperparameters['go_backwards'],
            stateful=self.hyperparameters['stateful'],
            unroll=self.hyperparameters['unroll']
        )
        return layer(previous_layer)

    def get_default_hyperparameters(self):
        return {
            'units': 50,
            'activation': 'tanh',
            'recurrent_activation': 'sigmoid',
            'use_bias': True,
            'kernel_initializer': 'glorot_uniform',
            'recurrent_initializer': 'orthogonal',
            'bias_initializer': 'zeros',
            'unit_forget_bias': True,
            'kernel_regularizer': None,
            'recurrent_regularizer': None,
            'bias_regularizer': None,
            'activity_regularizer': None,
            'kernel_constraint': None,
            'recurrent_constraint': None,
            'bias_constraint': None,
            'dropout': 0.0,
            'recurrent_dropout': 0.0,
            'return_sequences': False,
            'return_state': False,
            'go_backwards': False,
            'stateful': False,
            'unroll': False,
        }

    def get_hyperparameter_ranges(self):
        return {
            'units': (1, 1_000_000_000),
            'activation': ['tanh', 'relu', 'sigmoid', 'linear'],
            'recurrent_activation': ['sigmoid', 'hard_sigmoid', 'relu', 'tanh'],
            'use_bias': [True, False],
            'kernel_initializer': ['glorot_uniform', 'glorot_normal', 'he_normal', 'he_uniform', 'lecun_normal', 'lecun_uniform'],
            'recurrent_initializer': ['orthogonal', 'glorot_uniform', 'glorot_normal', 'he_normal', 'he_uniform', 'lecun_normal', 'lecun_uniform'],
            'bias_initializer': ['zeros', 'ones', 'constant', 'uniform', 'normal', 'truncated_normal', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform', 'lecun_normal', 'lecun_uniform'],
            'unit_forget_bias': [True, False],
            'kernel_regularizer': {'type': 'regularizer'},
            'recurrent_regularizer': {'type': 'regularizer'},
            'bias_regularizer': {'type': 'regularizer'},
            'activity_regularizer': {'type': 'regularizer'},
            'kernel_constraint': {'type': 'constraint'},
            'recurrent_constraint': {'type': 'constraint'},
            'bias_constraint': {'type': 'constraint'},
            'dropout': (0.0, 1.0),
            'recurrent_dropout': (0.0, 1.0),
            'return_sequences': [True, False],
            'return_state': [True, False],
            'go_backwards': [True, False],
            'stateful': [True, False],
            'time_major': [True, False],
            'unroll': [True, False],
        }


@layer_registry.register('gru')
class GRULayer(LayerSkeleton):
    def __init__(self, **kwargs):
        super().__init__()
        self.layer_name = 'gru'
        self.hyperparameters['units'] = kwargs.get('units', 50)
        self.hyperparameters['activation'] = kwargs.get('activation', 'tanh')
        self.hyperparameters['recurrent_activation'] = kwargs.get('recurrent_activation', 'sigmoid')
        self.hyperparameters['use_bias'] = kwargs.get('use_bias', True)
        self.hyperparameters['kernel_initializer'] = kwargs.get('kernel_initializer', 'glorot_uniform')
        self.hyperparameters['recurrent_initializer'] = kwargs.get('recurrent_initializer', 'orthogonal')
        self.hyperparameters['bias_initializer'] = kwargs.get('bias_initializer', 'zeros')
        self.hyperparameters['kernel_regularizer'] = kwargs.get('kernel_regularizer', None)
        self.hyperparameters['recurrent_regularizer'] = kwargs.get('recurrent_regularizer', None)
        self.hyperparameters['bias_regularizer'] = kwargs.get('bias_regularizer', None)
        self.hyperparameters['activity_regularizer'] = kwargs.get('activity_regularizer', None)
        self.hyperparameters['kernel_constraint'] = kwargs.get('kernel_constraint', None)
        self.hyperparameters['recurrent_constraint'] = kwargs.get('recurrent_constraint', None)
        self.hyperparameters['bias_constraint'] = kwargs.get('bias_constraint', None)
        self.hyperparameters['dropout'] = kwargs.get('dropout', 0.0)
        self.hyperparameters['recurrent_dropout'] = kwargs.get('recurrent_dropout', 0.0)
        self.hyperparameters['return_sequences'] = kwargs.get('return_sequences', False)
        self.hyperparameters['return_state'] = kwargs.get('return_state', False)
        self.hyperparameters['go_backwards'] = kwargs.get('go_backwards', False)
        self.hyperparameters['stateful'] = kwargs.get('stateful', False)
        self.hyperparameters['unroll'] = kwargs.get('unroll', False)

    def instance_layer(self, previous_layer):
        layer = keras.layers.GRU(
            units=self.hyperparameters['units'],
            activation=self.hyperparameters['activation'],
            recurrent_activation=self.hyperparameters['recurrent_activation'],
            use_bias=self.hyperparameters['use_bias'],
            kernel_initializer=self.hyperparameters['kernel_initializer'],
            recurrent_initializer=self.hyperparameters['recurrent_initializer'],
            bias_initializer=self.hyperparameters['bias_initializer'],
            kernel_regularizer=self.hyperparameters['kernel_regularizer'],
            recurrent_regularizer=self.hyperparameters['recurrent_regularizer'],
            bias_regularizer=self.hyperparameters['bias_regularizer'],
            activity_regularizer=self.hyperparameters['activity_regularizer'],
            kernel_constraint=self.hyperparameters['kernel_constraint'],
            recurrent_constraint=self.hyperparameters['recurrent_constraint'],
            bias_constraint=self.hyperparameters['bias_constraint'],
            dropout=self.hyperparameters['dropout'],
            recurrent_dropout=self.hyperparameters['recurrent_dropout'],
            return_sequences=self.hyperparameters['return_sequences'],
            return_state=self.hyperparameters['return_state'],
            go_backwards=self.hyperparameters['go_backwards'],
            stateful=self.hyperparameters['stateful'],
            unroll=self.hyperparameters['unroll']
        )
        return layer(previous_layer)

    def get_default_hyperparameters(self):
        return {
            'units': 50,
            'activation': 'tanh',
            'recurrent_activation': 'sigmoid',
            'use_bias': True,
            'kernel_initializer': 'glorot_uniform',
            'recurrent_initializer': 'orthogonal',
            'bias_initializer': 'zeros',
            'kernel_regularizer': None,
            'recurrent_regularizer': None,
            'bias_regularizer': None,
            'activity_regularizer': None,
            'kernel_constraint': None,
            'recurrent_constraint': None,
            'bias_constraint': None,
            'dropout': 0.0,
            'recurrent_dropout': 0.0,
            'return_sequences': False,
            'return_state': False,
            'go_backwards': False,
            'stateful': False,
            'unroll': False,
        }

    def get_hyperparameter_ranges(self):
        return {
            'units': (1, 1_000_000_000),
            'activation': ['tanh', 'relu', 'sigmoid', 'linear'],
            'recurrent_activation': ['sigmoid', 'hard_sigmoid', 'relu', 'tanh'],
            'use_bias': [True, False],
            'kernel_initializer': ['glorot_uniform', 'glorot_normal', 'he_normal', 'he_uniform', 'lecun_normal', 'lecun_uniform'],
            'recurrent_initializer': ['orthogonal', 'glorot_uniform', 'glorot_normal', 'he_normal', 'he_uniform', 'lecun_normal', 'lecun_uniform'],
            'bias_initializer': ['zeros', 'ones', 'constant', 'uniform', 'normal', 'truncated_normal', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform', 'lecun_normal', 'lecun_uniform'],
            'kernel_regularizer': {'type': 'regularizer'},
            'recurrent_regularizer': {'type': 'regularizer'},
            'bias_regularizer': {'type': 'regularizer'},
            'activity_regularizer': {'type': 'regularizer'},
            'kernel_constraint': {'type': 'constraint'},
            'recurrent_constraint': {'type': 'constraint'},
            'bias_constraint': {'type': 'constraint'},
            'dropout': (0.0, 1.0),
            'recurrent_dropout': (0.0, 1.0),
            'return_sequences': [True, False],
            'return_state': [True, False],
            'go_backwards': [True, False],
            'stateful': [True, False],
            'unroll': [True, False],
        }


@layer_registry.register('reshape')
class ReshapeLayer(LayerSkeleton):
    def __init__(self, **kwargs):
        super().__init__()
        self.layer_name = 'reshape'
        self.hyperparameters['target_shape'] = kwargs.get('target_shape')

    def instance_layer(self, previous_layer):
        layer = keras.layers.Reshape(target_shape=self.hyperparameters['target_shape'])
        return layer(previous_layer)

    def get_default_hyperparameters(self):
        return {
            'target_shape': (1, 1),
        }

    def get_hyperparameter_ranges(self):
        return {
            'target_shape': [(1, 1), (None, None)],
        }
