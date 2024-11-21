"""
WIP Up-to-date Model Compiler Code

Currently being written.

Needs to be integrated into the model compilation process across the codebase.

Will take model wrapper objects and compile them into tensorflow model objects.

Future Plans:
- Add Pytorch model compilation
"""
import keras
from routes.helpers.submodules.layers_registry import InputLayer


class ModelCompiler:
    """Base class for taking in a model wrapper object and compiling it into a model object."""
    def __init__(self, backend='tensorflow'):
        self.backend = backend

    def compile_model(self, layers):
        input_layer = None
        previous_layer = 1

        for slice in layers:
            input_layer, previous_layer = self._process_slice(layers[slice], input_layer, previous_layer)

        return keras.Model(inputs=input_layer, outputs=previous_layer)

    def _process_slice(self, slice_layers, input_layer, previous_layer):
        for layer in slice_layers:
            layer_object = slice_layers[layer]['layer']
            layer_tensor = self._process_layer(layer_object, previous_layer)

            if isinstance(layer_object, InputLayer):
                input_layer = layer_tensor

            previous_layer = layer_tensor

        return input_layer, previous_layer

    @staticmethod
    def _process_layer(layer_object, previous_layer):
        """Process a single layer and return its tensor."""
        layer_tensor = layer_object.instance_layer(previous_layer)
        print(f"Layer: {layer_object}")
        print(f"Layer Tensor: {layer_tensor}")
        return layer_tensor


# class TensorflowModelCompiler(ModelCompiler):
#     """Takes in a model wrapper object and compiles it into a tensorflow model object."""
#     pass
