import os
import logging

from flask import Blueprint, current_app, request
from routes.helpers.auth import require_api_key

from routes.helpers import utils

layers = Blueprint('layers', __name__)

REQUEST_SUCCEEDED = 200
REQUEST_CREATED = 201

BAD_REQUEST = 400
UNAUTHENTICATED_REQUEST = 401
FORBIDDEN_REQUEST = 403
PAGE_NOT_FOUND = 404
REQUEST_CONFLICT = 409

REQUEST_NOT_IMPLEMENTED = 501

logger = logging.getLogger(__name__)

@layers.route('/model/layers/create', methods=['POST'])  # TODO Change to put?
@require_api_key
def create_layer():
    given_id = request.form.get('id')
    layer_type = request.form.get('type')  # TODO check for valid layer type and class
    column = request.form.get('column')
    position = request.form.get('position')

    if not utils.check_id(given_id):
        return {'error': 'Invalid ID'}, BAD_REQUEST

    model = utils.fetch_model(given_id, api_key, current_app.config['UPLOAD_FOLDER'])
    model_path = os.path.join(current_app.config['UPLOAD_FOLDER'], "model_{}_{}.pk1".format(given_id, api_key))

    if not model.add_layer(layer_type, int(column), int(position)):
        return {'error': 'Layer already exists in that position'}, BAD_REQUEST

    utils.store_model(model, model_path)

    return {'info': 'Layer added'}, REQUEST_CREATED


# TODO Add layer adding by adding to the previous layer instead of specifying a position and vertical


@layers.route('/model/layers/delete', methods=['DELETE'])
@require_api_key
def delete_layer():
    given_id = request.form.get('id')
    column = request.form.get('column')
    position = request.form.get('position')

    if not utils.check_id(given_id):
        return {'error': 'Invalid ID'}, BAD_REQUEST

    model = utils.fetch_model(given_id, api_key, current_app.config['UPLOAD_FOLDER'])
    model_path = os.path.join(current_app.config['UPLOAD_FOLDER'], "model_{}_{}.pk1".format(given_id, api_key))

    if not model.remove_layer(int(column), int(position)):
        return {'error': 'Invalid layer position'}, BAD_REQUEST

    utils.store_model(model, model_path)

    return {'info': 'Layer removed'}, REQUEST_CREATED


@layers.route('/model/layers/verify', methods=['GET'])
@require_api_key
def verify_layers():
    given_id = request.form.get('id')

    if not utils.check_id(given_id):
        return {'error': 'Invalid ID'}, BAD_REQUEST

    model = utils.fetch_model(given_id, api_key, current_app.config['UPLOAD_FOLDER'])

    result = model.verify_layers()

    return result, REQUEST_SUCCEEDED


@layers.route('/model/layers/modify', methods=['PUT'])
@require_api_key
def modify_layers():
    given_id = request.form.get('id')

    if not utils.check_id(given_id):
        return {'error': 'Invalid ID'}, BAD_REQUEST

    model = utils.fetch_model(given_id, api_key, current_app.config['UPLOAD_FOLDER'])


@layers.route('/model/layers/hyperparameter', methods=['PUT'])
@require_api_key
def change_layer_hyperparameter():
    return {}, REQUEST_NOT_IMPLEMENTED


@layers.route('/model/layers/hyperparameter', methods=['GET'])
@require_api_key
def get_layer_hyperparameter():
    return {}, REQUEST_NOT_IMPLEMENTED


@layers.route('/model/layers/', methods=['GET'])
@require_api_key
def get_layer():
    return {}, REQUEST_NOT_IMPLEMENTED


@layers.route('/model/layers/position', methods=['PUT'])
@require_api_key
def change_position():
    return {}, REQUEST_NOT_IMPLEMENTED


@layers.route('/model/layers/preset', methods=['POST'])
@require_api_key
def create_preset_network():
    return {}, REQUEST_NOT_IMPLEMENTED


@layers.route('/model/layers/point', methods=['PUT'])
@require_api_key
def point_layer_output():
    given_id = request.form.get('id')
    next_column = int(request.form.get('next_column'))
    next_position = int(request.form.get('next_position'))
    column = request.form.get('column')
    if column.isnumeric():
        column = int(column)
    position = int(request.form.get('position'))
    start_range = int(request.form.get('start_tensor'))
    end_range = int(request.form.get('end_tensor'))

    if not utils.check_id(given_id):
        return {'error': 'Invalid ID'}, BAD_REQUEST

    model = utils.fetch_model(given_id, api_key, current_app.config['UPLOAD_FOLDER'])

    match model.point_layer(column, position, start_range, end_range, next_column, next_position):
        case 0:
            return {'error': 'The layer at the specified position was not found'}, BAD_REQUEST
        case 1:
            pass
        case 2:
            return {'success': 'The subsplit has been modified to the specified range'}

    model_path = os.path.join(current_app.config['UPLOAD_FOLDER'], "model_{}_{}.pk1".format(given_id, api_key))
    utils.store_model(model, model_path)

    return {'success': 'Layer subsplit/point successfully added'}, REQUEST_SUCCEEDED
