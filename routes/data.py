import os

from flask import Blueprint, current_app, request

from routes.helpers import data_proc, utils

data_views = Blueprint('data_views', __name__)

api_keys = utils.grab_api_keys()

REQUEST_SUCCEEDED = 200
REQUEST_CREATED = 201

BAD_REQUEST = 400
UNAUTHENTICATED_REQUEST = 401
FORBIDDEN_REQUEST = 403
PAGE_NOT_FOUND = 404
REQUEST_CONFLICT = 409

REQUEST_NOT_IMPLEMENTED = 501


@data_views.route('/data/upload', methods=['POST'])
def data_upload():
    api_key = request.headers.get('API-Key')
    if api_key not in api_keys:
        return {'error': 'Invalid API Key'}, UNAUTHENTICATED_REQUEST

    uploaded_file = request.files['file']

    # TODO verify safe file ID

    given_id = request.form.get('id')

    if given_id is None:
        return {'error': 'No Dataset ID provided'}, BAD_REQUEST

    if os.path.exists(os.path.join(current_app.config['UPLOAD_FOLDER'], "dataset_{}_{}.csv".format(given_id, api_key))):
        return {'error': 'Dataset already exists; delete existing set before trying again'}, REQUEST_CONFLICT

    if uploaded_file.filename != '':
        file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], "dataset_{}_{}.csv".format(given_id, api_key))
        uploaded_file.save(file_path)
        return {'info': 'Dataset uploaded'}, REQUEST_SUCCEEDED

    return {'error': 'Dataset not uploaded or given a file name'}, BAD_REQUEST


@data_views.route('/data/delete', methods=['DELETE'])
def delete_data():
    api_key = request.headers.get('API-Key')
    if api_key not in api_keys:
        return {'error': 'Invalid API Key'}, UNAUTHENTICATED_REQUEST

    given_id = request.form.get('id')
    file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], "dataset_{}_{}.csv".format(given_id, api_key))

    if given_id is None or given_id == '':
        return {'error': 'No Dataset ID provided'}, BAD_REQUEST

    if not os.path.exists(file_path):
        return {'error': 'Dataset does not exist; create one before trying again'}, REQUEST_CONFLICT

    os.remove(file_path)
    return {'info': 'Dataset deleted'}, REQUEST_SUCCEEDED


@data_views.route('/data/columns/', methods=['GET'])
def get_columns():
    api_key = request.headers.get('API-Key')
    if api_key not in api_keys:
        return {'error': 'Invalid API Key'}, UNAUTHENTICATED_REQUEST

    # TODO Add id checking

    given_id = request.form.get('id')

    model = utils.load_model_from_file(given_id, api_key, current_app.config['UPLOAD_FOLDER'])

    return model.process_columns(process_modifications=True), REQUEST_SUCCEEDED


@data_views.route('/data/columns/', methods=['DELETE'])
def add_column_deletion():
    api_key = request.headers.get('API-Key')
    if api_key not in api_keys:
        return {'error': 'Invalid API Key'}, UNAUTHENTICATED_REQUEST

    given_id = request.form.get('id')
    given_column = request.form.get('column')

    # TODO Add id checking

    model = utils.load_model_from_file(given_id, api_key, current_app.config['UPLOAD_FOLDER'])

    if model.data_modification_exists(data_proc.Column_Deletion, given_column):
        return {'error': 'Column Deletion already added'}, REQUEST_CONFLICT

    if given_column not in model.process_columns(process_modifications=False):
        return {'error': 'Given column does not exist'}, BAD_REQUEST

    model.delete_column(str(given_column))

    utils.save(model, model.model_path)

    return {'info': 'Column Deletion added'}, REQUEST_CREATED


@data_views.route('/data/columns/', methods=['POST'])
def undo_column_deletion():
    api_key = request.headers.get('API-Key')
    if api_key not in api_keys:
        return {'error': 'Invalid API Key'}, UNAUTHENTICATED_REQUEST

    given_id = request.form.get('id')
    given_column = request.form.get('column')

    # TODO Add id checking

    model = utils.load_model_from_file(given_id, api_key, current_app.config['UPLOAD_FOLDER'])

    if not model.data_modification_exists(data_proc.Column_Deletion, given_column):
        return {'error': 'Column Deletion does not exist'}, BAD_REQUEST

    model.add_deleted_column(given_column)

    utils.save(model, model.model_path)

    return {'info': 'Column Deletion removed'}, REQUEST_SUCCEEDED


@data_views.route('/data/verification', methods=['GET'])
def verify_data_integrity():
    api_key = request.headers.get('API-Key')
    if api_key not in api_keys:
        return {'error': 'Invalid API Key'}, UNAUTHENTICATED_REQUEST

