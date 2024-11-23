import logging
from class_registry import ClassRegistry
from keras.layers import StringLookup as KerasStringLookup
import pandas as pd

logger = logging.getLogger(__name__)

"""
Legacy Code

Currently being transitioned to new refactored data processing and modification classes.

Should be removed once the transition is complete and is no longer in use.
"""


class OldDataModification:
    def process(self, dataframe):
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError


class ColumnDeletion(OldDataModification):
    def __init__(self, column_name):
        self.type = 'ColumnDeletion'
        self.column_name = column_name

    def process(self, dataframe):
        return dataframe.drop(labels=self.column_name, axis=1)

    def __str__(self):
        return self.column_name


class SpecifiedFeature(OldDataModification):
    def __init__(self, column_name):
        self.type = 'SpecifiedFeature'
        self.column_name = column_name

    def process(self, dataframe):
        return dataframe.drop(labels=self.column_name, axis=1)

    def get_column(self, dataframe):
        return dataframe.pop(self.column_name)

    def __str__(self):
        return self.column_name


"""
Up-to-date Data Modification and Processing Class Code
"""


class DataModification:
    def apply(self, dataframe):
        raise NotImplementedError

    def adapt(self, data):
        pass


data_mod_registry = ClassRegistry[DataModification]()


@data_mod_registry.register('column_deletion')
class ColumnDeletion(DataModification):
    def __init__(self, column_name):
        self.column_name = column_name

    def apply(self, dataframe):
        columns_to_drop = dataframe.filter(regex=f'^{self.column_name}').columns
        return dataframe.drop(columns=columns_to_drop)

    def __str__(self):
        return self.column_name


class StringLookup(DataModification):
    def __init__(self, column_name, lookup_map=None):
        self.column_name = column_name
        self.lookup_layer = KerasStringLookup(output_mode='one_hot', vocabulary=lookup_map)

    def adapt(self, data):
        self.lookup_layer.adapt(data[self.column_name])

    def apply(self, data):
        one_hot_encoded = self.lookup_layer(data[self.column_name])
        one_hot_encoded_df = pd.DataFrame(one_hot_encoded.numpy(), columns=[f"{self.column_name}_{i}" for i in range(one_hot_encoded.shape[1])])
        data = data.drop(columns=[self.column_name])
        data = pd.concat([data, one_hot_encoded_df], axis=1)
        return data

    def __str__(self):
        return self.column_name
