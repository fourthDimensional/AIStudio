import logging
from class_registry import ClassRegistry

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

data_mod_registry = ClassRegistry[DataModification]()

@data_mod_registry.register('column_deletion')
class ColumnDeletion(DataModification):
    def __init__(self, column_name):
        self.column_name = column_name

    def apply(self, dataframe):
        return dataframe.drop(labels=self.column_name, axis=1)

    def __str__(self):
        return self.column_name

