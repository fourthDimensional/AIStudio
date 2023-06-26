import pickle
import os

def load_model_from_file(id, api_key):
    model_path = os.path.join(app.config['UPLOAD_FOLDER'], "model_{}_{}.pk1".format(id, api_key))

    with open(model_path, 'rb') as file:
        return pickle.load(file)

def convert_text_file_to_list(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        lines = [line.strip() for line in lines]
    return lines

def save(model, model_path):
    with open(model_path, 'wb') as file:
            pickle.dump(model, file)