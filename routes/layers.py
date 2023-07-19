from flask import Blueprint, current_app, request

from routes.helpers import utils

layers = Blueprint('layers', __name__)

api_keys = utils.grab_api_keys()

REQUEST_SUCCEEDED = 200
REQUEST_CREATED = 201

BAD_REQUEST = 400
UNAUTHENTICATED_REQUEST = 401
FORBIDDEN_REQUEST = 403
PAGE_NOT_FOUND = 404
REQUEST_CONFLICT = 409

REQUEST_NOT_IMPLEMENTED = 501


@layers.route('/model/layers/create', methods=['POST'])
def create_layer():
    api_key = request.headers.get('API-Key')
    if api_key not in api_keys:
        return {'error': 'Invalid API Key'}, UNAUTHENTICATED_REQUEST

    given_id = request.form.get('id')
    layer_type = request.form.get('type')  # TODO check for valid layer type

    given_id = request.form.get('id')
    if not utils.check_id(given_id):
        return {'error': 'Invalid ID'}, BAD_REQUEST

    model = utils.load_model_from_file(given_id, api_key, current_app.config['UPLOAD_FOLDER'])

    model.add_layer(layer_type=str(layer_type))


@layers.route('/model/layers/delete', methods=['DELETE'])
def delete_layer():
    api_key = request.headers.get('API-Key')
    if api_key not in api_keys:
        return {'error': 'Invalid API Key'}, UNAUTHENTICATED_REQUEST


@layers.route('/model/layers/hyperparameter', methods=['PUT'])
def change_layer_hyperparameter():
    api_key = request.headers.get('API-Key')
    if api_key not in api_keys:
        return {'error': 'Invalid API Key'}, UNAUTHENTICATED_REQUEST


@layers.route('/model/layers/hyperparameter', methods=['GET'])
def get_layer_hyperparameter():
    api_key = request.headers.get('API-Key')
    if api_key not in api_keys:
        return {'error': 'Invalid API Key'}, UNAUTHENTICATED_REQUEST


@layers.route('/model/layers/', methods=['GET'])
def get_layer():
    api_key = request.headers.get('API-Key')
    if api_key not in api_keys:
        return {'error': 'Invalid API Key'}, UNAUTHENTICATED_REQUEST


@layers.route('/model/layers/position', methods=['PUT'])
def change_position():
    api_key = request.headers.get('API-Key')
    if api_key not in api_keys:
        return {'error': 'Invalid API Key'}, UNAUTHENTICATED_REQUEST


@layers.route('/model/layers/preset', methods=['POST'])
def create_preset_network():
    api_key = request.headers.get('API-Key')
    if api_key not in api_keys:
        return {'error': 'Invalid API Key'}, UNAUTHENTICATED_REQUEST


@layers.route('/model/layers/preset/hyperparameter', methods=['PUT'])
def change_preset_hyperparameter():
    api_key = request.headers.get('API-Key')
    if api_key not in api_keys:
        return {'error': 'Invalid API Key'}, UNAUTHENTICATED_REQUEST
