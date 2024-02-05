import os

from flask import Blueprint, current_app, request, send_file
from routes.helpers.auth import require_api_key

from routes.helpers import model as md, utils, data_proc
import logging

model_basic = Blueprint('model_basic', __name__)

REQUEST_SUCCEEDED = 200
REQUEST_CREATED = 201

BAD_REQUEST = 400
UNAUTHENTICATED_REQUEST = 401
FORBIDDEN_REQUEST = 403
PAGE_NOT_FOUND = 404
REQUEST_CONFLICT = 409

REQUEST_NOT_IMPLEMENTED = 501

logger = logging.getLogger(__name__)


@model_basic.route('/model/create', methods=['POST'])
@require_api_key
def create_model():
    api_key = request.headers.get('API-Key')
    if api_key not in api_keys:
        return {'error': 'Invalid API Key'}, UNAUTHENTICATED_REQUEST

    given_id = request.form.get('id')
    if not utils.check_id(given_id):
        return {'error': 'Invalid ID'}, BAD_REQUEST
    file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], "dataset_{}_{}.csv".format(given_id, api_key))

    if not os.path.exists(file_path):
        return {'error': 'Dataset does not exist; create one before trying again'}, REQUEST_CONFLICT

    model_name = request.form.get('model_name')
    given_type = request.form.get('type')
    visual_name = request.form.get('visual_name')

    model = md.create_model(file_path, model_name, visual_name, given_type)
    return_value = model[0]
    model = model[1]

    utils.store_model(model, model_path)

    return return_value, REQUEST_CREATED


@model_basic.route('/model', methods=['delete'])
def delete_model():
    api_key = request.headers.get('API-Key')
    if api_key not in api_keys:
        return {'error': 'Invalid API Key'}, UNAUTHENTICATED_REQUEST


@model_basic.route('/model/name', methods=['GET'])
def get_model_name():
    given_id = request.form.get('id')
    if not utils.check_id(given_id):
        return {'error': 'Invalid ID'},
    model_path = os.path.join(current_app.config['UPLOAD_FOLDER'], "model_{}_{}.pk1".format(given_id, api_key))

    if not os.path.exists(model_path):
        return {'error': 'Model does not exist; create one before trying again'}, REQUEST_CONFLICT

    model = utils.fetch_model(given_id, api_key, current_app.config['UPLOAD_FOLDER'])

    return model.name


@model_basic.route('/model/features', methods=['POST'])
def specify_model_features():
    api_key = request.headers.get('API-Key')
    if api_key not in api_keys:
        return {'error': 'Invalid API Key'}, UNAUTHENTICATED_REQUEST

    given_id = request.form.get('id')
    if not utils.check_id(given_id):
        return {'error': 'Invalid ID'}, BAD_REQUEST
    model_path = os.path.join(current_app.config['UPLOAD_FOLDER'], "model_{}_{}.pk1".format(given_id, api_key))

    if not os.path.exists(model_path):
        return {'error': 'Model does not exist; create one before trying again'}, REQUEST_CONFLICT

    model = utils.fetch_model(given_id, api_key, current_app.config['UPLOAD_FOLDER'])

    given_column = request.form.get('column')

    if model.data_modification_exists(data_proc.SpecifiedFeature, given_column):
        return {'error': 'Column feature already added'}, REQUEST_CONFLICT

    if given_column not in model.process_columns(process_modifications=False):
        return {'error': 'Given column does not exist'}, BAD_REQUEST

    old_index = model.specify_feature(str(given_column))
    del model.layers["Input"][old_index]

    model.feature_count += 1

    utils.store_model(model, model_path)

    return {'info': 'Feature specified and will be a training metric'}, REQUEST_SUCCEEDED


@model_basic.route('/model/image', methods=['GET'])
def generate_model_image():
    api_key = request.headers.get('API-Key')
    if api_key not in api_keys:
        return {'error': 'Invalid API Key'}, UNAUTHENTICATED_REQUEST

    given_id = request.form.get('id')
    if not utils.check_id(given_id):
        return {'error': 'Invalid ID'}, BAD_REQUEST
    model_path = os.path.join(current_app.config['UPLOAD_FOLDER'], "model_{}_{}.pk1".format(given_id, api_key))

    if not os.path.exists(model_path):
        return {'error': 'Model does not exist; create one before trying again'}, REQUEST_CONFLICT

    model = utils.fetch_model(given_id, api_key, current_app.config['UPLOAD_FOLDER'])

    model.verify_layers()

    return send_file(path_or_file=os.path.join(current_app.config['UPLOAD_FOLDER'], "model.png")), REQUEST_SUCCEEDED

