from flask import Flask, jsonify, request
from flask_cors import CORS
import os
import model as md
import utils
import data_proc

# instantiate the app
app = Flask(__name__)
app.config.from_object(__name__)
app.config['MAX_CONTENT_LENGTH'] = 300 + (1024 * 1024)  # Basic request size + large dataset limit

text_file_path = 'apikeys.txt'
api_keys = utils.convert_text_file_to_list(text_file_path)

# enable CORS
CORS(app, resources={r'/*': {'origins': '*'}})

UPLOAD_FOLDER = 'static/files'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

REQUEST_SUCCEEDED = 200
REQUEST_CREATED = 201

BAD_REQUEST = 400
UNAUTHENTICATED_REQUEST = 401
FORBIDDEN_REQUEST = 403
PAGE_NOT_FOUND = 404
REQUEST_CONFLICT = 409

REQUEST_NOT_IMPLEMENTED = 501


@app.route('/data/upload', methods=['POST'])
def upload():
    api_key = request.headers.get('API-Key')
    if api_key not in api_keys:
        return {'error': 'Invalid API Key'}, UNAUTHENTICATED_REQUEST

    uploaded_file = request.files['file']

    # TODO verify safe file ID

    given_id = request.form.get('id')

    if given_id is None:
        return {'error': 'No Dataset ID provided'}, BAD_REQUEST

    if os.path.exists(os.path.join(app.config['UPLOAD_FOLDER'], "dataset_{}_{}.csv".format(given_id, api_key))):
        return {'error': 'Dataset already exists; delete existing set before trying again'}, REQUEST_CONFLICT

    if uploaded_file.filename != '':
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], "dataset_{}_{}.csv".format(given_id, api_key))
        uploaded_file.save(file_path)
        return {'info': 'Dataset uploaded'}, REQUEST_SUCCEEDED

    return {'error': 'Dataset not uploaded or given a file name'}, BAD_REQUEST


@app.route('/data/delete', methods=['DELETE'])
def delete_data():
    api_key = request.headers.get('API-Key')
    if api_key not in api_keys:
        return {'error': 'Invalid API Key'}, UNAUTHENTICATED_REQUEST

    given_id = request.form.get('id')
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], "dataset_{}_{}.csv".format(given_id, api_key))

    if given_id is None or given_id == '':
        return {'error': 'No Dataset ID provided'}, BAD_REQUEST

    if not os.path.exists(file_path):
        return {'error': 'Dataset does not exist; create one before trying again'}, REQUEST_CONFLICT

    os.remove(file_path)
    return {'info': 'Dataset deleted'}, REQUEST_SUCCEEDED


@app.route('/model/create', methods=['POST'])
def create_model():
    api_key = request.headers.get('API-Key')
    if api_key not in api_keys:
        return {'error': 'Invalid API Key'}, UNAUTHENTICATED_REQUEST

    given_id = request.form.get('id')
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], "dataset_{}_{}.csv".format(given_id, api_key))

    if not os.path.exists(file_path):
        return {'error': 'Dataset does not exist; create one before trying again'}, REQUEST_CONFLICT

    model_name = request.form.get('model_name')
    given_type = request.form.get('type')
    visual_name = request.form.get('visual_name')
    model_path = os.path.join(app.config['UPLOAD_FOLDER'], "model_{}_{}.pk1".format(given_id, api_key))

    model = md.create_model(file_path, model_name, visual_name, given_type, model_path)
    return_value = model[0]
    model = model[1]
    print(model.train())

    utils.save(model, model_path)

    return return_value, REQUEST_CREATED


@app.route('/model/name', methods=['GET'])
def get_model_name():
    api_key = request.headers.get('API-Key')
    if api_key not in api_keys:
        return {'error': 'Invalid API Key'}, UNAUTHENTICATED_REQUEST

    given_id = request.form.get('id')

    if given_id is None or given_id == '':
        return {'error': 'No Dataset ID provided'}, BAD_REQUEST

    if not os.path.exists(os.path.join(app.config['UPLOAD_FOLDER'], "model_{}_{}.pk1".format(given_id, api_key))):
        return {'error': 'Model does not exist; create one before trying again'}, REQUEST_CONFLICT

    model = utils.load_model_from_file(given_id, api_key, app.config['UPLOAD_FOLDER'])

    return model.name


@app.route('/model/columns/', methods=['GET'])
def get_columns():
    api_key = request.headers.get('API-Key')
    if api_key not in api_keys:
        return {'error': 'Invalid API Key'}, UNAUTHENTICATED_REQUEST

    # TODO Add id checking

    given_id = request.form.get('id')

    model = utils.load_model_from_file(given_id, api_key, app.config['UPLOAD_FOLDER'])

    return model.process_columns(process_modifications=True), REQUEST_SUCCEEDED


@app.route('/model/columns/', methods=['DELETE'])
def add_column_deletion():
    api_key = request.headers.get('API-Key')
    if api_key not in api_keys:
        return {'error': 'Invalid API Key'}, UNAUTHENTICATED_REQUEST

    given_id = request.form.get('id')
    given_column = request.form.get('column')

    # TODO Add id checking

    model = utils.load_model_from_file(given_id, api_key, app.config['UPLOAD_FOLDER'])

    if model.data_modification_exists(data_proc.Column_Deletion, given_column):
        return {'error': 'Column Deletion already added'}, REQUEST_CONFLICT

    if given_column not in model.process_columns(process_modifications=False):
        return {'error': 'Given column does not exist'}, BAD_REQUEST

    model.delete_column(str(given_column))

    utils.save(model, model.model_path)

    return {'info': 'Column Deletion added'}, REQUEST_CREATED


@app.route('/model/columns/', methods=['POST'])
def undo_column_deletion():
    api_key = request.headers.get('API-Key')
    if api_key not in api_keys:
        return {'error': 'Invalid API Key'}, UNAUTHENTICATED_REQUEST

    given_id = request.form.get('id')
    given_column = request.form.get('column')

    # TODO Add id checking

    model = utils.load_model_from_file(given_id, api_key, app.config['UPLOAD_FOLDER'])

    if not model.data_modification_exists(data_proc.Column_Deletion, given_column):
        return {'error': 'Column Deletion does not exist'}, BAD_REQUEST

    model.add_deleted_column(given_column)

    utils.save(model, model.model_path)

    return {'info': 'Column Deletion removed'}, REQUEST_SUCCEEDED


@app.route('/model/preprocessing/create', methods=['POST'])
def create_preprocessing_layer():
    api_key = request.headers.get('API-Key')
    if api_key not in api_keys:
        return {'error': 'Invalid API Key'}, UNAUTHENTICATED_REQUEST

    given_id = request.form.get('id')

    # TODO Add id checking

    model = utils.load_model_from_file(given_id, api_key, app.config['UPLOAD_FOLDER'])


@app.route('/model/preprocessing/delete', methods=['DELETE'])
def delete_preprocessing_layer():
    api_key = request.headers.get('API-Key')
    if api_key not in api_keys:
        return {'error': 'Invalid API Key'}, UNAUTHENTICATED_REQUEST
    
    
@app.route('/', methods=[])
def ():
    api_key = request.headers.get('API-Key')
    if api_key not in api_keys:
        return {'error': 'Invalid API Key'}, UNAUTHENTICATED_REQUEST
    



if __name__ == '__main__':
    app.run(port=5001)
