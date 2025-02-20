import os

import keras
import tensorflow as tf
from class_registry import ClassRegistry

# Import the new hyperparameter classes
from routes.helpers.submodules.hyperparameters import RealHyperparameter, IntegerHyperparameter, CategoricalHyperparameter

"""
Up-to-date Layer Registry and Class Definition Code

This version uses Hyperparameter objects (RealHyperparameter, IntegerHyperparameter,
CategoricalHyperparameter) for most tunable layer hyperparameters, except for input sizes
and reshape sizes, which are kept as plain values.
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
        This class must be passed as a custom object when deserializing the model.
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
        # hyperparameters stored as a dict; values may be plain or hyperparameter objects.
        self.hyperparameters = {}

    def instance_layer(self, previous_layer):
        raise NotImplementedError

    def get_hyperparameters(self):
        """
        Return a dict mapping hyperparameter names to their current value.
        """
        hp_values = {}
        for key, hp in self.hyperparameters.items():
            if hasattr(hp, "get_value"):
                hp_values[key] = hp.get_value()
            else:
                hp_values[key] = hp
        return hp_values

    def modify_hyperparameters(self, hyperparameters):
        """
        Update hyperparameters by name.
        """
        for key, value in hyperparameters.items():
            if key in self.hyperparameters:
                hp = self.hyperparameters[key]
                if hasattr(hp, "set_value"):
                    hp.set_value(value)
                else:
                    self.hyperparameters[key] = value
            else:
                raise KeyError(f"Hyperparameter {key} not found in layer {self.layer_name}")

    def get_hyperparameter_ranges(self):
        """
        Returns a dict mapping hyperparameter names to their allowed ranges.
        For hyperparameter objects, this returns the 'range' attribute.
        """
        ranges = {}
        for key, hp in self.hyperparameters.items():
            if hasattr(hp, "range"):
                ranges[key] = hp.range
            else:
                ranges[key] = None
        return ranges

    def get_default_hyperparameters(self):
        """
        Returns a dict mapping hyperparameter names to their default values.
        For hyperparameter objects, this returns the 'default' attribute.
        """
        defaults = {}
        for key, hp in self.hyperparameters.items():
            if hasattr(hp, "default"):
                defaults[key] = hp.default
            else:
                defaults[key] = hp
        return defaults

    def suggested_hyperparameter(self):
        raise NotImplementedError


layer_registry = ClassRegistry[LayerSkeleton]()


@layer_registry.register('input')
class InputLayer(LayerSkeleton):
    def __init__(self, **kwargs):
        super().__init__()
        self.layer_name = 'input'
        # Do NOT use a hyperparameter class for input sizes.
        self.hyperparameters['input_size'] = kwargs['input_size']

    def instance_layer(self, _previous_layer):
        return keras.layers.Input(shape=(self.hyperparameters['input_size'],))


@layer_registry.register('dense')
class DenseLayer(LayerSkeleton):
    def __init__(self, **kwargs):
        super().__init__()
        self.layer_name = 'dense'
        self.hyperparameters['units'] = IntegerHyperparameter(
            'units', kwargs['units'], (1, 1_000_000_000)
        )
        self.hyperparameters['activation'] = CategoricalHyperparameter(
            'activation', kwargs.get('activation', 'relu'),
            ['relu', 'sigmoid', 'tanh', 'softmax', 'softplus', 'softsign', 'selu', 'elu', 'exponential', 'linear']
        )
        self.hyperparameters['use_bias'] = CategoricalHyperparameter(
            'use_bias', kwargs.get('use_bias', True), [True, False]
        )
        self.hyperparameters['kernel_initializer'] = CategoricalHyperparameter(
            'kernel_initializer', kwargs.get('kernel_initializer', 'glorot_uniform'),
            ['glorot_uniform', 'glorot_normal', 'he_normal', 'he_uniform', 'lecun_normal', 'lecun_uniform']
        )
        self.hyperparameters['bias_initializer'] = CategoricalHyperparameter(
            'bias_initializer', kwargs.get('bias_initializer', 'zeros'),
            ['zeros', 'ones', 'constant', 'uniform', 'normal', 'truncated_normal',
             'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform', 'lecun_normal', 'lecun_uniform']
        )
        self.hyperparameters['kernel_regularizer'] = kwargs.get('kernel_regularizer', None)
        self.hyperparameters['bias_regularizer'] = kwargs.get('bias_regularizer', None)
        self.hyperparameters['activity_regularizer'] = kwargs.get('activity_regularizer', None)
        self.hyperparameters['kernel_constraint'] = kwargs.get('kernel_constraint', None)
        self.hyperparameters['bias_constraint'] = kwargs.get('bias_constraint', None)
        self.hyperparameters['lora_rank'] = CategoricalHyperparameter(
            'lora_rank', kwargs.get('lora_rank', None),
            [None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        )

    def instance_layer(self, previous_layer):
        layer = keras.layers.Dense(
            units=self.hyperparameters['units'].get_value(),
            activation=self.hyperparameters['activation'].get_value(),
            use_bias=self.hyperparameters['use_bias'].get_value(),
            kernel_initializer=self.hyperparameters['kernel_initializer'].get_value(),
            bias_initializer=self.hyperparameters['bias_initializer'].get_value(),
        )
        return layer(previous_layer)


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
        self.hyperparameters['rate'] = RealHyperparameter(
            'rate', kwargs.get('rate', 0.5), (0.0, 1.0)
        )

    def instance_layer(self, previous_layer):
        layer = keras.layers.Dropout(rate=self.hyperparameters['rate'].get_value())
        return layer(previous_layer)


@layer_registry.register('gaussian_noise')
class GaussianNoiseLayer(LayerSkeleton):
    def __init__(self, **kwargs):
        super().__init__()
        self.layer_name = 'gaussian_noise'
        self.hyperparameters['stddev'] = RealHyperparameter(
            'stddev', kwargs.get('stddev', 0.1), (0.0, 1.0)
        )

    def instance_layer(self, previous_layer):
        layer = keras.layers.GaussianNoise(stddev=self.hyperparameters['stddev'].get_value())
        return layer(previous_layer)


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
        self.hyperparameters['activation'] = CategoricalHyperparameter(
            'activation', kwargs.get('activation', 'relu'),
            ['relu', 'sigmoid', 'tanh', 'softmax', 'softplus', 'softsign', 'selu', 'elu', 'exponential', 'linear']
        )

    def instance_layer(self, previous_layer):
        layer = keras.layers.Activation(activation=self.hyperparameters['activation'].get_value())
        return layer(previous_layer)


@layer_registry.register('embedding')
class EmbeddingLayer(LayerSkeleton):
    def __init__(self, **kwargs):
        super().__init__()
        self.layer_name = 'embedding'
        # For embedding, we treat input_dim as a fixed size (not wrapped in a hyperparameter class)
        self.hyperparameters['input_dim'] = kwargs.get('input_dim', 100)
        self.hyperparameters['output_dim'] = IntegerHyperparameter(
            'output_dim', kwargs.get('output_dim', 64), (1, 1_000)
        )
        input_length = kwargs.get('input_length', None)
        if input_length is not None:
            # Here, input_length is considered an input size so we leave it as a raw value.
            self.hyperparameters['input_length'] = input_length
        else:
            self.hyperparameters['input_length'] = None

    def instance_layer(self, previous_layer):
        layer = keras.layers.Embedding(
            input_dim=self.hyperparameters['input_dim'],
            output_dim=self.hyperparameters['output_dim'].get_value(),
            input_length=self.hyperparameters['input_length']
        )
        return layer(previous_layer)


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
        self.hyperparameters['units'] = IntegerHyperparameter(
            'units', kwargs.get('units', 50), (1, 1_000_000_000)
        )
        self.hyperparameters['activation'] = CategoricalHyperparameter(
            'activation', kwargs.get('activation', 'tanh'),
            ['tanh', 'relu', 'sigmoid', 'linear']
        )
        self.hyperparameters['recurrent_activation'] = CategoricalHyperparameter(
            'recurrent_activation', kwargs.get('recurrent_activation', 'sigmoid'),
            ['sigmoid', 'hard_sigmoid', 'relu', 'tanh']
        )
        self.hyperparameters['use_bias'] = CategoricalHyperparameter(
            'use_bias', kwargs.get('use_bias', True), [True, False]
        )
        self.hyperparameters['kernel_initializer'] = CategoricalHyperparameter(
            'kernel_initializer', kwargs.get('kernel_initializer', 'glorot_uniform'),
            ['glorot_uniform', 'glorot_normal', 'he_normal', 'he_uniform', 'lecun_normal', 'lecun_uniform']
        )
        self.hyperparameters['recurrent_initializer'] = CategoricalHyperparameter(
            'recurrent_initializer', kwargs.get('recurrent_initializer', 'orthogonal'),
            ['orthogonal', 'glorot_uniform', 'glorot_normal', 'he_normal', 'he_uniform', 'lecun_normal', 'lecun_uniform']
        )
        self.hyperparameters['bias_initializer'] = CategoricalHyperparameter(
            'bias_initializer', kwargs.get('bias_initializer', 'zeros'),
            ['zeros', 'ones', 'constant', 'uniform', 'normal', 'truncated_normal',
             'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform', 'lecun_normal', 'lecun_uniform']
        )
        self.hyperparameters['unit_forget_bias'] = CategoricalHyperparameter(
            'unit_forget_bias', kwargs.get('unit_forget_bias', True), [True, False]
        )
        self.hyperparameters['kernel_regularizer'] = kwargs.get('kernel_regularizer', None)
        self.hyperparameters['recurrent_regularizer'] = kwargs.get('recurrent_regularizer', None)
        self.hyperparameters['bias_regularizer'] = kwargs.get('bias_regularizer', None)
        self.hyperparameters['activity_regularizer'] = kwargs.get('activity_regularizer', None)
        self.hyperparameters['kernel_constraint'] = kwargs.get('kernel_constraint', None)
        self.hyperparameters['recurrent_constraint'] = kwargs.get('recurrent_constraint', None)
        self.hyperparameters['bias_constraint'] = kwargs.get('bias_constraint', None)
        self.hyperparameters['dropout'] = RealHyperparameter(
            'dropout', kwargs.get('dropout', 0.0), (0.0, 1.0)
        )
        self.hyperparameters['recurrent_dropout'] = RealHyperparameter(
            'recurrent_dropout', kwargs.get('recurrent_dropout', 0.0), (0.0, 1.0)
        )
        self.hyperparameters['return_sequences'] = CategoricalHyperparameter(
            'return_sequences', kwargs.get('return_sequences', False), [True, False]
        )
        self.hyperparameters['return_state'] = CategoricalHyperparameter(
            'return_state', kwargs.get('return_state', False), [True, False]
        )
        self.hyperparameters['go_backwards'] = CategoricalHyperparameter(
            'go_backwards', kwargs.get('go_backwards', False), [True, False]
        )
        self.hyperparameters['stateful'] = CategoricalHyperparameter(
            'stateful', kwargs.get('stateful', False), [True, False]
        )
        self.hyperparameters['unroll'] = CategoricalHyperparameter(
            'unroll', kwargs.get('unroll', False), [True, False]
        )

    def instance_layer(self, previous_layer):
        layer = keras.layers.LSTM(
            units=self.hyperparameters['units'].get_value(),
            activation=self.hyperparameters['activation'].get_value(),
            recurrent_activation=self.hyperparameters['recurrent_activation'].get_value(),
            use_bias=self.hyperparameters['use_bias'].get_value(),
            kernel_initializer=self.hyperparameters['kernel_initializer'].get_value(),
            recurrent_initializer=self.hyperparameters['recurrent_initializer'].get_value(),
            bias_initializer=self.hyperparameters['bias_initializer'].get_value(),
            unit_forget_bias=self.hyperparameters['unit_forget_bias'].get_value(),
            kernel_regularizer=self.hyperparameters['kernel_regularizer'],
            recurrent_regularizer=self.hyperparameters['recurrent_regularizer'],
            bias_regularizer=self.hyperparameters['bias_regularizer'],
            activity_regularizer=self.hyperparameters['activity_regularizer'],
            kernel_constraint=self.hyperparameters['kernel_constraint'],
            recurrent_constraint=self.hyperparameters['recurrent_constraint'],
            bias_constraint=self.hyperparameters['bias_constraint'],
            dropout=self.hyperparameters['dropout'].get_value(),
            recurrent_dropout=self.hyperparameters['recurrent_dropout'].get_value(),
            return_sequences=self.hyperparameters['return_sequences'].get_value(),
            return_state=self.hyperparameters['return_state'].get_value(),
            go_backwards=self.hyperparameters['go_backwards'].get_value(),
            stateful=self.hyperparameters['stateful'].get_value(),
            unroll=self.hyperparameters['unroll'].get_value()
        )
        return layer(previous_layer)


@layer_registry.register('gru')
class GRULayer(LayerSkeleton):
    def __init__(self, **kwargs):
        super().__init__()
        self.layer_name = 'gru'
        self.hyperparameters['units'] = IntegerHyperparameter(
            'units', kwargs.get('units', 50), (1, 1_000_000_000)
        )
        self.hyperparameters['activation'] = CategoricalHyperparameter(
            'activation', kwargs.get('activation', 'tanh'),
            ['tanh', 'relu', 'sigmoid', 'linear']
        )
        self.hyperparameters['recurrent_activation'] = CategoricalHyperparameter(
            'recurrent_activation', kwargs.get('recurrent_activation', 'sigmoid'),
            ['sigmoid', 'hard_sigmoid', 'relu', 'tanh']
        )
        self.hyperparameters['use_bias'] = CategoricalHyperparameter(
            'use_bias', kwargs.get('use_bias', True), [True, False]
        )
        self.hyperparameters['kernel_initializer'] = CategoricalHyperparameter(
            'kernel_initializer', kwargs.get('kernel_initializer', 'glorot_uniform'),
            ['glorot_uniform', 'glorot_normal', 'he_normal', 'he_uniform', 'lecun_normal', 'lecun_uniform']
        )
        self.hyperparameters['recurrent_initializer'] = CategoricalHyperparameter(
            'recurrent_initializer', kwargs.get('recurrent_initializer', 'orthogonal'),
            ['orthogonal', 'glorot_uniform', 'glorot_normal', 'he_normal', 'he_uniform', 'lecun_normal', 'lecun_uniform']
        )
        self.hyperparameters['bias_initializer'] = CategoricalHyperparameter(
            'bias_initializer', kwargs.get('bias_initializer', 'zeros'),
            ['zeros', 'ones', 'constant', 'uniform', 'normal', 'truncated_normal',
             'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform', 'lecun_normal', 'lecun_uniform']
        )
        self.hyperparameters['kernel_regularizer'] = kwargs.get('kernel_regularizer', None)
        self.hyperparameters['recurrent_regularizer'] = kwargs.get('recurrent_regularizer', None)
        self.hyperparameters['bias_regularizer'] = kwargs.get('bias_regularizer', None)
        self.hyperparameters['activity_regularizer'] = kwargs.get('activity_regularizer', None)
        self.hyperparameters['kernel_constraint'] = kwargs.get('kernel_constraint', None)
        self.hyperparameters['recurrent_constraint'] = kwargs.get('recurrent_constraint', None)
        self.hyperparameters['bias_constraint'] = kwargs.get('bias_constraint', None)
        self.hyperparameters['dropout'] = RealHyperparameter(
            'dropout', kwargs.get('dropout', 0.0), (0.0, 1.0)
        )
        self.hyperparameters['recurrent_dropout'] = RealHyperparameter(
            'recurrent_dropout', kwargs.get('recurrent_dropout', 0.0), (0.0, 1.0)
        )
        self.hyperparameters['return_sequences'] = CategoricalHyperparameter(
            'return_sequences', kwargs.get('return_sequences', False), [True, False]
        )
        self.hyperparameters['return_state'] = CategoricalHyperparameter(
            'return_state', kwargs.get('return_state', False), [True, False]
        )
        self.hyperparameters['go_backwards'] = CategoricalHyperparameter(
            'go_backwards', kwargs.get('go_backwards', False), [True, False]
        )
        self.hyperparameters['stateful'] = CategoricalHyperparameter(
            'stateful', kwargs.get('stateful', False), [True, False]
        )
        self.hyperparameters['unroll'] = CategoricalHyperparameter(
            'unroll', kwargs.get('unroll', False), [True, False]
        )

    def instance_layer(self, previous_layer):
        layer = keras.layers.GRU(
            units=self.hyperparameters['units'].get_value(),
            activation=self.hyperparameters['activation'].get_value(),
            recurrent_activation=self.hyperparameters['recurrent_activation'].get_value(),
            use_bias=self.hyperparameters['use_bias'].get_value(),
            kernel_initializer=self.hyperparameters['kernel_initializer'].get_value(),
            recurrent_initializer=self.hyperparameters['recurrent_initializer'].get_value(),
            bias_initializer=self.hyperparameters['bias_initializer'].get_value(),
            kernel_regularizer=self.hyperparameters['kernel_regularizer'],
            recurrent_regularizer=self.hyperparameters['recurrent_regularizer'],
            bias_regularizer=self.hyperparameters['bias_regularizer'],
            activity_regularizer=self.hyperparameters['activity_regularizer'],
            kernel_constraint=self.hyperparameters['kernel_constraint'],
            recurrent_constraint=self.hyperparameters['recurrent_constraint'],
            bias_constraint=self.hyperparameters['bias_constraint'],
            dropout=self.hyperparameters['dropout'].get_value(),
            recurrent_dropout=self.hyperparameters['recurrent_dropout'].get_value(),
            return_sequences=self.hyperparameters['return_sequences'].get_value(),
            return_state=self.hyperparameters['return_state'].get_value(),
            go_backwards=self.hyperparameters['go_backwards'].get_value(),
            stateful=self.hyperparameters['stateful'].get_value(),
            unroll=self.hyperparameters['unroll'].get_value()
        )
        return layer(previous_layer)


@layer_registry.register('reshape')
class ReshapeLayer(LayerSkeleton):
    def __init__(self, **kwargs):
        super().__init__()
        self.layer_name = 'reshape'
        # Do NOT use a hyperparameter class for reshape sizes.
        self.hyperparameters['target_shape'] = kwargs.get('target_shape')

    def instance_layer(self, previous_layer):
        layer = keras.layers.Reshape(target_shape=self.hyperparameters['target_shape'])
        return layer(previous_layer)
