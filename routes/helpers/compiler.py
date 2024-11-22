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
from routes.helpers.submodules.layers_registry import UniversalSplitLayer


class ModelCompiler:
    """Base class for taking in a model wrapper object and compiling it into a model object."""
    def __init__(self, backend='tensorflow'):
        self.backend = backend

        self.input_storage = {'input': []}
        self.current_x = 0
        self.current_y = 0

    def compile_model(self, layers):
        self.input_storage = {'input': [], 0: {0: []}}
        self.current_x = 0
        self.current_y = 0

        input_layer = None
        previous_layer = 2

        for slice in layers:
            input_layer, previous_layer = self._process_slice(layers[slice], input_layer, previous_layer)
            self.current_x += 1
            self.current_y = 0

        return keras.Model(inputs=input_layer, outputs=previous_layer)

    def _process_slice(self, slice, input_layer, previous_layer):
        for layer in slice:
            layer_object = slice[layer]
            layer_tensor = self._process_layer(layer_object, previous_layer)

            if isinstance(layer_object['layer'], InputLayer):
                input_layer = layer_tensor

            previous_layer = layer_tensor
            self.current_y += 1

        return input_layer, previous_layer

    def _process_layer(self, layer_object, previous_layer):
        """Process a single layer and return its tensor."""
        if isinstance(previous_layer, int):
            layer_tensor = layer_object['layer'].instance_layer(previous_layer)
        elif len(self.input_storage[self.current_x][self.current_y]) > 1:
            concat_layer = keras.layers.Concatenate()(self.input_storage[self.current_x][self.current_y])
            layer_tensor = layer_object['layer'].instance_layer(concat_layer)
        else:
            layer_tensor = layer_object['layer'].instance_layer(self.input_storage[self.current_x][self.current_y][0])

        outputs = layer_object['outputs']

        # Subsplit is present
        if outputs:
            if outputs[0][0] > 0:
                subsplit_dimensions = self._get_subsplit_dimensions(outputs)
                splitter = UniversalSplitLayer(num_or_size_splits=subsplit_dimensions, axis=-1)
                split_tensors_list = splitter.split(layer_tensor)

                index = 0
                for output in outputs:
                    positional_index = output[1]
                    vertical_index = output[2]

                    if positional_index not in self.input_storage:
                        self.input_storage[positional_index] = {}

                    if vertical_index not in self.input_storage[positional_index]:
                        self.input_storage[positional_index][vertical_index] = []

                    self.input_storage[positional_index][vertical_index].append(split_tensors_list[index])
                    index += 1
            else:
                positional_index = outputs[0][1]
                vertical_index = outputs[0][2]

                if positional_index not in self.input_storage:
                    self.input_storage[positional_index] = {}

                if vertical_index not in self.input_storage[positional_index]:
                    self.input_storage[positional_index][vertical_index] = []

                self.input_storage[positional_index][vertical_index].append(layer_tensor)
        else:
            # this is either a final or unconnected layer
            pass

        return layer_tensor

    @staticmethod
    def _get_subsplit_dimensions(outputs):
        subsplit_dimensions = []
        for subsplit in outputs:
            # get the subsplit value
            subsplit_dimensions.append(subsplit[0])

        return subsplit_dimensions


# class TensorflowModelCompiler(ModelCompiler):
#     """Takes in a model wrapper object and compiles it into a tensorflow model object."""
#     pass
