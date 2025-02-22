import logging
from class_registry import ClassRegistry
from keras.layers import StringLookup as KerasStringLookup
import pandas as pd
import re

logger = logging.getLogger(__name__)

"""
Up-to-date Data Modification and Processing Class Code
"""


class DataModification:
    def apply(self, dataframe):
        raise NotImplementedError

    def adapt(self, data):
        raise NotImplementedError


modification_registry = ClassRegistry[DataModification]()


# TODO Column deletion of date_day also deletes date_dayofyear because of the multi-column regex checking
@modification_registry.register('column_deletion')
class ColumnDeletion(DataModification):
    def __init__(self, column_names):
        self.column_names = column_names

    def apply(self, dataframe):
        columns_to_drop = []
        for column_name in self.column_names:
            columns_to_drop.extend(dataframe.filter(regex=f'^{column_name}').columns)
        return dataframe.drop(columns=columns_to_drop)

    def __str__(self):
        return ', '.join(self.column_names)


@modification_registry.register('date_feature')
class DateFeatureExtraction(DataModification):
    def __init__(self, column_name):
        self.column_name = column_name
        self.date_patterns = [
            re.compile(r'\d{4}-\d{2}-\d{2}'),  # YYYY-MM-DD
            re.compile(r'\d{2}/\d{2}/\d{4}'),  # MM/DD/YYYY
            re.compile(r'\d{2}-\d{2}-\d{4}'),  # DD-MM-YYYY
            re.compile(r'\d{4}/\d{2}/\d{2}'),  # YYYY/MM/DD
        ]

    def apply(self, dataframe):
        def extract_date_features(date_str):
            for pattern in self.date_patterns:
                if pattern.match(date_str):
                    date = pd.to_datetime(date_str, errors='coerce')
                    return pd.Series({
                        f'{self.column_name}_year': date.year,
                        f'{self.column_name}_month': date.month,
                        f'{self.column_name}_day': date.day,
                        f'{self.column_name}_dy': date.dayofyear,
                    })
            return pd.Series({
                f'{self.column_name}_year': None,
                f'{self.column_name}_month': None,
                f'{self.column_name}_day': None,
                f'{self.column_name}_dy': None,
            })

        date_features = dataframe[self.column_name].apply(extract_date_features)

        dataframe = dataframe.drop(columns=[self.column_name])
        dataframe = pd.concat([dataframe, date_features], axis=1)
        return dataframe

    def __str__(self):
        return self.column_name


@modification_registry.register('specified_feature')
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
