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
from redis import Redis
from routes.helpers.model import ModelWrapper


class ModelCompiler:
    """Base class for taking in a model wrapper object and compiling it into a model object."""
    def __init__(self, backend='tensorflow'):
        self.backend = backend

        self.input_storage = {'input': []}
        self.current_x = 0
        self.current_y = 0

    def compile_model(self, model_wrapper, redis_connection: dict):
        redis = Redis(**redis_connection)

        if redis.exists(f"compiled_model:{model_wrapper.uuid}"):
            return ModelWrapper.deserialize(self.redis_connection.JSON().get(f"compiled_model:{model_wrapper.uuid}"))

        layers = model_wrapper.layer_manipulator.get_layers()

        self.input_storage = {'input': [], 0: {0: []}}
        self.current_x = 0
        self.current_y = 0

        input_layer = None
        previous_layer = 2 # don't change this, it breaks it.

        for layer_slice in layers:
            input_layer, previous_layer = self._process_slice(layers[layer_slice], input_layer, previous_layer)
            self.current_x += 1
            self.current_y = 0

        keras.Model(inputs=input_layer, outputs=previous_layer).summary()

        redis.json().set(f"compiled_model:{model_wrapper.uuid}", '$', model_wrapper.serialize())
        redis.expire(f"compiled_model:{model_wrapper.uuid}", 3600) # 1 hour

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
        layer_tensor = self._get_layer_tensor(layer_object, previous_layer)
        outputs = layer_object['outputs']

        if outputs:
            self._handle_outputs(layer_tensor, outputs)
        else:
            return layer_tensor

        return layer_tensor

    def _get_layer_tensor(self, layer_object, previous_layer):
        """Get the tensor for the current layer based on the previous layer."""
        if isinstance(previous_layer, int):
            return layer_object['layer'].instance_layer(previous_layer)
        elif len(self.input_storage[self.current_x][self.current_y]) > 1:
            concat_layer = keras.layers.Concatenate()(self.input_storage[self.current_x][self.current_y])
            return layer_object['layer'].instance_layer(concat_layer)
        else:
            return layer_object['layer'].instance_layer(self.input_storage[self.current_x][self.current_y][0])

    def _handle_outputs(self, layer_tensor, outputs):
        """Handle the outputs of the current layer."""
        if outputs[0][0] > 0:
            self._handle_subsplit(layer_tensor, outputs)
        else:
            self._store_output(layer_tensor, outputs)

    def _handle_subsplit(self, layer_tensor, outputs):
        """Handle subsplit of the layer tensor."""
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

    def _store_output(self, layer_tensor, outputs):
        """Store the output tensor in the input storage."""
        for output in outputs:
            positional_index = output[1]
            vertical_index = output[2]

            if positional_index not in self.input_storage:
                self.input_storage[positional_index] = {}

            if vertical_index not in self.input_storage[positional_index]:
                self.input_storage[positional_index][vertical_index] = []

            self.input_storage[positional_index][vertical_index].append(layer_tensor)

    @staticmethod
    def _get_subsplit_dimensions(outputs):
        """Get the dimensions for subsplit from the outputs."""
        subsplit_dimensions = []
        for subsplit in outputs:
            subsplit_dimensions.append(subsplit[0])
        return subsplit_dimensions
