import logging

logging.basicConfig(format='%(levelname)s (%(asctime)s): %(message)s (Line: %(lineno)d [%(filename)s])',
                    datefmt='%I:%M:%S %p',
                    level=logging.INFO)


class DataModification:
    def process(self, dataframe):
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError


class ColumnDeletion(DataModification):
    def __init__(self, column_name):
        self.column_name = column_name

    def process(self, dataframe):
        return dataframe.drop(labels=self.column_name, axis=1)

    def __str__(self):
        return self.column_name


class SpecifiedFeature(DataModification):
    def __init__(self, column_name):
        self.column_name = column_name

    def process(self, dataframe):
        return dataframe.drop(labels=self.column_name, axis=1)

    def get_column(self, dataframe):
        return dataframe.pop(self.column_name)

    def __str__(self):
        return self.column_name
