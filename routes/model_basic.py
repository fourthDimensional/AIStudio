from flask import request, Blueprint, current_app
from routes.helpers import utils, model as md
import os

model_basic = Blueprint('model_basic', __name__)

api_keys = utils.grab_api_keys()

REQUEST_SUCCEEDED = 200
REQUEST_CREATED = 201

BAD_REQUEST = 400
UNAUTHENTICATED_REQUEST = 401
FORBIDDEN_REQUEST = 403
PAGE_NOT_FOUND = 404
REQUEST_CONFLICT = 409

REQUEST_NOT_IMPLEMENTED = 501


@model_basic.route('/model/create', methods=['POST'])
def create_model():
    api_key = request.headers.get('API-Key')
    if api_key not in api_keys:
        return {'error': 'Invalid API Key'}, UNAUTHENTICATED_REQUEST

    given_id = request.form.get('id')
    file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], "dataset_{}_{}.csv".format(given_id, api_key))

    if not os.path.exists(file_path):
        return {'error': 'Dataset does not exist; create one before trying again'}, REQUEST_CONFLICT

    model_name = request.form.get('model_name')
    given_type = request.form.get('type')
    visual_name = request.form.get('visual_name')
    model_path = os.path.join(current_app.config['UPLOAD_FOLDER'], "model_{}_{}.pk1".format(given_id, api_key))

    model = md.create_model(file_path, model_name, visual_name, given_type, model_path)
    return_value = model[0]
    model = model[1]
    print(model.train())

    utils.save(model, model_path)

    return return_value, REQUEST_CREATED


@model_basic.route('/model/name', methods=['GET'])
def get_model_name():
    api_key = request.headers.get('API-Key')
    if api_key not in api_keys:
        return {'error': 'Invalid API Key'}, UNAUTHENTICATED_REQUEST

    given_id = request.form.get('id')

    if given_id is None or given_id == '':
        return {'error': 'No Dataset ID provided'}, BAD_REQUEST

    if not os.path.exists(os.path.join(current_app.config['UPLOAD_FOLDER'], "model_{}_{}.pk1".format(given_id, api_key))):
        return {'error': 'Model does not exist; create one before trying again'}, REQUEST_CONFLICT

    model = utils.load_model_from_file(given_id, api_key, current_app.config['UPLOAD_FOLDER'])

    return model.name
