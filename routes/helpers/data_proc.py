import logging

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


pointer = {'network_id': ['previous_network_output_tensor_id', 'previous_network_output_tensor_id_2']}

network_hash = {'network_id': ['network_x', 'network']}

