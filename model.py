import os
import re
import pandas as pd


def check_naming_convention(string):
    pattern = r'^[a-z]+(_[a-z]+)*$'
    if re.match(pattern, string):
        return True
    return False


def create_model(file_path, name, visual_name, network_type, model_path):
    valid_network_types = ["regression", "classification"]

    # * Verifies that the inputs are supported/usable
    if network_type not in valid_network_types:
        return {'error': 'Unsupported network type'}

    if not os.path.exists(file_path):
        return {'error': 'Dataset does not exist; create one before trying to construct a model'}, 409

    if not check_naming_convention(name):
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

    def train(self):
        encodings = ["utf-8", "utf-8-sig", "iso-8859-1", "latin1", "cp1252"]
        for encoding in encodings:
            try:
                with open(self.dataset_path, 'r', encoding=encoding, errors='replace') as f:
                    dataframe_csv = pd.read_csv(f)
                break
            except Exception as e:
                dataframe_csv = None
                print(f"Error reading with {encoding}: {e}")
                pass

        if dataframe_csv is not None:
            column_names = dataframe_csv.columns.tolist()
            # labels = dataframe_csv.pop('salary')
            return column_names
        else:
            print("Failed to read the dataset.")

    def add_preprocessing_layer(self, network_type):
        pass

    def remove_preprocessing_layer(self, layer_id):
        pass

    def __len__(self):
        pass

    def __str__(self):
        pass
