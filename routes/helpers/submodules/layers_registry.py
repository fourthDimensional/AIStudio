import tensorflow as tf
from class_registry import ClassRegistry

class LayerSkeleton:
    def __init__(self):
            self.layer_name = 'default'
            self.input_size = []
            self.output_tensor_split = []
            self.output_locations_x = []
            self.output_locations_y = []

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
    pass

@layer_registry.register('batch_normalization')
class BatchNormalizationLayer(LayerSkeleton):
    pass

@layer_registry.register('dense')
class DenseLayer(LayerSkeleton):
    pass

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
