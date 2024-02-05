import os
import logging

from flask import Blueprint, current_app, request
from routes.helpers.auth import require_api_key

from routes.helpers import data_proc, utils, layers

data_views = Blueprint('data_views', __name__)

REQUEST_SUCCEEDED = 200
REQUEST_CREATED = 201

BAD_REQUEST = 400
UNAUTHENTICATED_REQUEST = 401
FORBIDDEN_REQUEST = 403
PAGE_NOT_FOUND = 404
REQUEST_CONFLICT = 409

REQUEST_NOT_IMPLEMENTED = 501

AUTHKEY_HEADER = 'authkey'

logger = logging.getLogger(__name__)


@data_views.route('/data/upload', methods=['POST'])
@require_api_key
def data_upload():
    uploaded_file = request.files['file']

    given_id = request.form.get('id')
    if not utils.check_id(given_id):
        return {'error': 'Invalid ID'}, BAD_REQUEST

    if os.path.exists(os.path.join(current_app.config['UPLOAD_FOLDER'], "dataset_{}_{}.csv".format(given_id, api_key))):
        return {'error': 'Dataset already exists; delete existing set before trying again'}, REQUEST_CONFLICT

    if uploaded_file.filename != '':
        file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], "dataset_{}_{}.csv".format(given_id, api_key))
        uploaded_file.save(file_path)
        return {'info': 'Dataset uploaded'}, REQUEST_SUCCEEDED

    return {'error': 'Dataset not uploaded or given a file name'}, BAD_REQUEST


@data_views.route('/data/delete', methods=['DELETE'])
@require_api_key
def delete_data():
    given_id = request.form.get('id')
    file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], "dataset_{}_{}.csv".format(given_id, api_key))

    if given_id is None or given_id == '':
        return {'error': 'No Dataset ID provided'}, BAD_REQUEST

    if not os.path.exists(file_path):
        return {'error': 'Dataset does not exist; create one before trying again'}, REQUEST_CONFLICT

    os.remove(file_path)
    return {'info': 'Dataset deleted'}, REQUEST_SUCCEEDED


@data_views.route('/data/columns/', methods=['GET'])
@require_api_key
def get_columns():
    api_key = request.headers.get(AUTHKEY_HEADER)

    given_id = request.form.get('id')
    if not utils.check_id(given_id):
        return {'error': 'Invalid ID'}, BAD_REQUEST

    model, error = utils.fetch_model(api_key, given_id)

    match error:
        case -1:
            return {'error': 'Specified model is corrupted'}
        case 0:
            return {'error': 'Specified model id does not exist'}

    return {'columns': model.process_columns(process_modifications=True)}, REQUEST_SUCCEEDED


@data_views.route('/data/columns/', methods=['DELETE'])
@require_api_key
def add_column_deletion():
    given_column = request.form.get('column')
    api_key = request.headers.get(AUTHKEY_HEADER)

    given_id = request.form.get('id')
    if not utils.check_id(given_id):
        return {'error': 'Invalid ID'}, BAD_REQUEST

    model, error = utils.fetch_model(api_key, given_id)

    match error:
        case -1:
            return {'error': 'Specified model is corrupted'}
        case 0:
            return {'error': 'Specified model id does not exist'}

    if model.data_modification_exists(data_proc.ColumnDeletion, given_column):
        return {'error': 'Column Deletion already added'}, REQUEST_CONFLICT

    if given_column not in model.process_columns(process_modifications=False):
        return {'error': 'Given column does not exist'}, BAD_REQUEST

    old_index = model.delete_column(str(given_column))
    del model.layers["Input"][old_index]

    utils.store_model(api_key, given_id, model)

    return {'info': 'Column Deletion added'}, REQUEST_CREATED


# TODO Revamp data manipulation and redo these two functions. Make sure you do it in a separate branch once
#  everything else is finished.


@data_views.route('/data/columns/', methods=['POST'])
@require_api_key
def undo_column_deletion():
    given_column = request.form.get('column')

    given_id = request.form.get('id')
    if not utils.check_id(given_id):
        return {'error': 'Invalid ID'}, BAD_REQUEST

    model = utils.fetch_model(given_id, api_key, current_app.config['UPLOAD_FOLDER'])

    if not model.data_modification_exists(data_proc.ColumnDeletion, given_column):
        return {'error': 'Column Deletion does not exist'}, BAD_REQUEST

    new_index = model.add_deleted_column(given_column)
    model.layers["Input"][new_index] = layers.SpecialInput()

    utils.store_model(model, model.model_path)

    return {'info': 'Column Deletion removed'}, REQUEST_SUCCEEDED


@data_views.route('/data/verification', methods=['GET'])
@require_api_key
def verify_data_integrity():
    return {}, REQUEST_NOT_IMPLEMENTED
