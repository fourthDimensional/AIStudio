"""
WIP Up-to-date Model Compiler Code

Currently being written.

Needs to be integrated into the model compilation process across the codebase.

Will take model wrapper objects and compile them into tensorflow model objects.

Future Plans:
- Add Pytorch model compilation
"""

class ModelCompiler:
    """Base class for taking in a model wrapper object and compiling it into a model object."""
    pass

class TensorflowModelCompiler(ModelCompiler):
    """Takes in a model wrapper object and compiles it into a tensorflow model object."""
    pass