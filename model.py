import os
import utils
import logging

logging.basicConfig(format='%(levelname)s (%(asctime)s): %(message)s (Line: %(lineno)d [%(filename)s])',
                    datefmt='%I:%M:%S %p',
                    level=logging.DEBUG)





def create_model(file_path, name, visual_name, network_type, model_path):
    valid_network_types = ["regression", "classification"]

    # * Verifies that the inputs are supported/usable
    if network_type not in valid_network_types:
        return {'error': 'Unsupported network type'}

    if not os.path.exists(file_path):
        return {'error': 'Dataset does not exist; create one before trying to construct a model'}, 409

    if not utils.check_naming_convention(name):
        return {'error': 'Internal name not supported'}, 409

    if network_type == 'regression':
        pass
    elif network_type == 'classification':
        pass

    model = Model(name, visual_name, network_type, model_path, file_path)

    return [{'info': 'Model created successfully'}, model]


class Model:
    def __init__(self, name, visual_name, network_type, model_path, dataset_path):
        self.name = name
        self.visual_name = visual_name
        self.type = network_type
        self.dataset_path = dataset_path
        self.model_path = model_path

        self.network_count = 0
        self.networks = []

        self.features = None
        self.labels = None
        self.column_hash = {}

    def train(self):
        dataframe_csv = utils.convert_to_dataframe(self.dataset_path)

        if dataframe_csv is not None:
            column_names = dataframe_csv.columns.tolist()
            # labels = dataframe_csv.pop('salary')
            return column_names
        else:
            print("Failed to read the dataset.")

    def process_columns(self):
        dataframe_csv = utils.convert_to_dataframe(self.dataset_path)

        columns = dataframe_csv.columns.tolist()

        for x in range(0, len(columns)):
            self.column_hash[x] = columns[x]

        logging.info(self.column_hash)

        return len(columns)

    def add_preprocessing_layer(self, network_type):
        pass

    def remove_preprocessing_layer(self, layer_id):
        pass

    def __len__(self):
        pass

    def __str__(self):
        pass

