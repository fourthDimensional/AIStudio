import tensorflow as tf
import torch
from class_registry import ClassRegistry

"""
Up-to-date Layer Registry and Class Definition Code

Needs class definitions and implementations for the following layers:
- Input
- BatchNormalization
- Dense
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


class UniversalSplitLayer:
    def __init__(self, num_or_size_splits, axis=-1):
        """
        A universal split layer compatible with TensorFlow and PyTorch.

        :param num_or_size_splits: Number or size of splits
        :param axis: Axis to split along
        """
        self.num_or_size_splits = num_or_size_splits
        self.axis = axis

    def split(self, tensor):
        """
        Split the tensor into multiple parts.

        :param tensor: Input tensor
        :return: A list of split tensors
        """
        return tf.split(tensor, self.num_or_size_splits, axis=self.axis)

        # if isinstance(tensor, tf.Tensor):
        #     return tf.split(tensor, self.num_or_size_splits, axis=self.axis)
        # elif isinstance(tensor, torch.Tensor):
        #     return torch.split(tensor, self.num_or_size_splits, dim=self.axis)
        # else:
        #     raise TypeError(f"Unsupported tensor type: {type(tensor)}")


class LayerSkeleton:
    def __init__(self):
        self.layer_name = 'default'

        self.hyperparameters = {}

    def instance_layer(self, previous_layer):
        raise NotImplementedError

    def list_hyperparameters(self):
        raise NotImplementedError

    def modify_hyperparameters(self):
        raise NotImplementedError

    def get_default_hyperparameter(self):
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
        layer = tf.keras.layers.Input(shape=self.hyperparameters['input_size'])
        return layer


@layer_registry.register('batch_normalization')
class BatchNormalizationLayer(LayerSkeleton):
    pass


@layer_registry.register('dense')
class DenseLayer(LayerSkeleton):
    def __init__(self, **kwargs):
        super().__init__()
        self.layer_name = 'dense'

        self.hyperparameters = self.get_default_hyperparameters()

    def instance_layer(self, previous_layer):
        layer = tf.keras.layers.Dense(**self.hyperparameters)
        return layer(previous_layer)

    def get_default_hyperparameters(self):
        return {
            'units': 10,
            'activation': 'relu'
        }


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
