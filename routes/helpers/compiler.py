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
    def compile_model(self, layers):
        current_x_position = 0
        current_y_position = 0

        previous_layer = 4
        input_layer = 0

        for slice in layers:
            for layer in layers[slice]:
                print(f"Layer: {layers[slice][layer]}")
                print(f"Position: ({current_x_position}, {current_y_position})")

                layer_object = layers[slice][layer]['layer']
                layer_tensor = layer_object.instance_layer(previous_layer)

                if isinstance(layer_object, InputLayer):
                    print(f"Layer Tensor: {layer_tensor}")
                    input_layer = layer_tensor

                previous_layer = layer_tensor

                current_y_position += 1
            current_x_position += 1
            current_y_position = 0

        model = keras.Model(inputs=input_layer, outputs=previous_layer)

        return model


# class TensorflowModelCompiler(ModelCompiler):
#     """Takes in a model wrapper object and compiles it into a tensorflow model object."""
#     pass
