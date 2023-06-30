import os
import utils
import logging
import pandas as pd

logging.basicConfig(format='%(levelname)s (%(asctime)s): %(message)s (Line: %(lineno)d [%(filename)s])',
                    datefmt='%I:%M:%S %p',
                    level=logging.DEBUG)


class Data_Modification:
    def process(self, dataframe):
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError


class Column_Deletion(Data_Modification):
    def __init__(self, column_name):
        self.column_name = column_name

    def process(self, dataframe):
        return dataframe.drop(labels=self.column_name, axis=1)

    def __str__(self):
        return self.column_name


class Preprocessor_Layer:
    def __init__(self, layer_id):
        self.layer_id = layer_id
        self.type = ""  # Numerical, Text, Categorical

    def create_instanced_layer(self):
        raise NotImplementedError


# TODO add text vectorization
class Normalization_Layer(Preprocessor_Layer):
    def __init__(self, layer_id):
        super().__init__(layer_id)
        self.layer_id = layer_id
        self.type = "numerical"

    def create_instanced_layer(self):
        pass


class Discretization_Layer(Preprocessor_Layer):
    def __init__(self, layer_id):
        super().__init__(layer_id)
        self.layer_id = layer_id
        self.type = "numerical"

    def create_instanced_layer(self):
        pass


class Category_Encoding_Layer(Preprocessor_Layer):
    def __init__(self, layer_id):
        super().__init__(layer_id)
        self.layer_id = layer_id
        self.type = "numerical"

    def create_instanced_layer(self):
        pass


class Hashing_Layer(Preprocessor_Layer):
    def __init__(self, layer_id):
        super().__init__(layer_id)
        self.layer_id = layer_id
        self.type = "numerical"

    def create_instanced_layer(self):
        pass


class String_Lookup_Layer(Preprocessor_Layer):
    def __init__(self, layer_id):
        super().__init__(layer_id)
        self.layer_id = layer_id
        self.type = "numerical"

    def create_instanced_layer(self):
        pass


class Integer_Lookup_Layer(Preprocessor_Layer):
    def __init__(self, layer_id):
        super().__init__(layer_id)
        self.layer_id = layer_id
        self.type = "numerical"

    def create_instanced_layer(self):
        pass
