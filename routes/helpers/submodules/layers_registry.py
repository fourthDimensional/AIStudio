import tensorflow as tf
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

class LayerSkeleton:
    def __init__(self):
        self.layer_name = 'default'
        self.input_size = []
        self.output_tensor_split = []
        self.output_locations_x = []
        self.output_locations_y = []

        self.hyperparameters = {}

    def instance_layer(self):
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
        self.output_tensor_split = [1]
        self.output_locations_x = [0]
        self.output_locations_y = [0]

    def instance_layer(self):
        layer = tf.keras.layers.Input(shape=self.input_size)
        return layer

@layer_registry.register('batch_normalization')
class BatchNormalizationLayer(LayerSkeleton):
    pass

@layer_registry.register('dense')
class DenseLayer(LayerSkeleton):
    def __init__(self, **kwargs):
        super().__init__()
        self.layer_name = 'dense'

    def instance_layer(self):
        layer = tf.keras.layers.Dense(**self.hyperparameters)
        return layer


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
