import os
import logging

from flask import Blueprint, current_app, request, jsonify, current_app
from routes.helpers.submodules.auth import require_api_key, is_valid_auth

from routes.helpers.submodules import data_proc, layers, utils
from routes.helpers.submodules.worker import data_info
from routes.helpers.submodules.storage import StorageInterface, RedisFileStorage

from rq import Queue
from redis import Redis
from io import BytesIO
import pandas as pd

# Configuration for Redis connection
redis_host: str = 'localhost'
redis_port: int = 6379
redis_db: int = 0

redis_queue = Queue('data', connection=Redis())

REDIS_CONNECTION_INFO = {
    'host': os.getenv('REDIS_HOST', redis_host),
    'port': int(os.getenv('REDIS_PORT', str(redis_port))),
    'decode_responses': True
}

data_views = Blueprint('data_views', __name__)

dataset_storage = StorageInterface(RedisFileStorage(Redis(**REDIS_CONNECTION_INFO)))

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
    api_key = request.headers.get(AUTHKEY_HEADER)
    given_id = request.form.get('id')

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
    api_key = request.headers.get(AUTHKEY_HEADER)
    file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], "dataset_{}_{}.csv".format(given_id, api_key))

    if not os.path.exists(file_path):
        return {'error': 'Dataset does not exist; create one before trying again'}, REQUEST_CONFLICT

    os.remove(file_path)
    return {'info': 'Dataset deleted'}, REQUEST_SUCCEEDED


@data_views.route('/data/columns/', methods=['GET'])
@require_api_key
def get_columns():
    api_key = request.headers.get(AUTHKEY_HEADER)
    given_id = request.form.get('id')

    model, error = utils.fetch_model(api_key, given_id)

    match error:
        case -1, -2:
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

    model, error = utils.fetch_model(api_key, given_id)

    match error:
        case -1, -2:
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
    api_key = request.headers.get(AUTHKEY_HEADER)
    given_id = request.form.get('id')

    model, error = utils.fetch_model(api_key, given_id)

    match error:
        case -1, -2:
            return {'error': 'Specified model is corrupted'}
        case 0:
            return {'error': 'Specified model id does not exist'}

    if not model.data_modification_exists(data_proc.ColumnDeletion, given_column):
        return {'error': 'Column Deletion does not exist'}, BAD_REQUEST

    new_index = model.add_deleted_column(given_column)
    model.layers["Input"][new_index] = layers.SpecialInput()

    utils.store_model(api_key, given_id, model)

    return {'info': 'Column Deletion removed'}, REQUEST_SUCCEEDED


@data_views.route('/data/verification', methods=['GET'])
@require_api_key
def verify_data_integrity():
    return {}, REQUEST_NOT_IMPLEMENTED


@data_views.route('/data/template/list', methods=['GET'])
@require_api_key
def list_datasets():
    datasets = []
    for file in os.listdir(current_app.config['DATASET_FOLDER']):
        if file.endswith('.csv'):
            # also have file size and csv # of entries
            dataset = {
                'name': file[:-4],
                'size': os.path.getsize(os.path.join(current_app.config['DATASET_FOLDER'], file)),
                'entries': len(open(os.path.join(current_app.config['DATASET_FOLDER'], file)).readlines())
            }

            datasets.append(dataset)

    return jsonify(datasets), REQUEST_SUCCEEDED


@data_views.route('/data/template/<template_name>', methods=['POST'])
@require_api_key
def register_dataset_from_template(template_name, api_key):
    if dataset_storage.exists(f"{api_key}:{template_name}"):
        return {'error': 'Dataset already exists'}, REQUEST_CONFLICT

    # find the csv file labelled the given template name in this folder: os.listdir(current_app.config['DATASET_FOLDER'])
    # then upload the file to the dataset storage with the metadata 'name', 'size', and 'entries'
    for file in os.listdir(current_app.config['DATASET_FOLDER']):
        if file == template_name + '.csv':
            file_path = os.path.join(current_app.config['DATASET_FOLDER'], file)
            break
    else:
        return {'error': 'Template not found'}, 40

    dataframe = pd.read_csv(file_path)

    csv_buffer = BytesIO()
    dataframe.to_csv(csv_buffer)
    csv_buffer.seek(0)  # Move the cursor to the beginning of the buffer

    # TODO find a better way to access the database here
    redis = current_app.config['DATABASE']
    redis.json().arrappend(f'api_key:{api_key}', '$.dataset_keys', template_name)
    id = redis.json().arrindex(f'api_key:{api_key}', '$.dataset_keys', template_name)[0]

    dataset_storage.store_file(f"{api_key}:{template_name}", csv_buffer.read(), {
                                'status': 'uploaded',
                                'name': template_name,
                                'size': os.path.getsize(file_path),
                                'entries': len(open(file_path).readlines()),
                                'id': id,
                                'columns': dataframe.columns.tolist(),}
                               )

    return {'info': 'Dataset registered'}, REQUEST_CREATED


@data_views.route('/data/list', methods=['GET'])
@require_api_key
def list_private_datasets(api_key, metadata):
    dataset_keys = metadata['dataset_keys']

    dataset_info = []

    for dataset_key in dataset_keys:
        if not dataset_storage.exists(f"{api_key}:{dataset_key}"):
            logging.info(f'Invalid dataset key is present in {api_key}\'s metadata. Skipping.')

            logging.info(f"{api_key}:{dataset_key}")
            continue

        dataset = dataset_storage.get_file_metadata(f"{api_key}:{dataset_key}")
        dataset_info.append({
            'key': dataset_key,
            'name': dataset['name'],
            'size': dataset['size'],
            'entries': dataset['entries'],
            'id': dataset['id'],
        })

    return jsonify(dataset_info), REQUEST_SUCCEEDED


@data_views.route('/data/<dataset_key>', methods=['GET'])
@require_api_key
def get_private_dataset(dataset_key, api_key):
    dataset = dataset_storage.get_file(f"{api_key}:{dataset_key}")

    if not dataset:
        return {'error': 'Dataset not found'}, REQUEST_CONFLICT

    return jsonify(dataset.decode('utf-8')), REQUEST_SUCCEEDED


@data_views.route('/data/<dataset_key>', methods=['DELETE'])
@require_api_key
def delete_private_dataset(dataset_key, api_key):
    if not dataset_storage.exists(f"{api_key}:{dataset_key}"):
        return {'error': 'Dataset not found'}, REQUEST_CONFLICT

    dataset_storage.delete_file(f"{api_key}:{dataset_key}")

    redis = current_app.config['DATABASE']

    index = redis.json().arrindex(f'api_key:{api_key}', '$.dataset_keys', dataset_key)[0]

    if index == -1:
        return {'error': 'Dataset not found in metadata'}, REQUEST_CONFLICT

    redis.json().arrpop(f'api_key:{api_key}', '$.dataset_keys', index)

    return {'info': 'Dataset deleted'}, REQUEST_SUCCEEDED


# TODO Deprecated function remove this because it's a proof of concept
@data_views.route('/data/information', methods=['POST'])
@require_api_key
def start_data_processing(api_key):
    job = redis_queue.enqueue(data_info.generate_profile_report, 'classification_small', api_key, REDIS_CONNECTION_INFO)
    return jsonify({'info': 'Data processing started', 'job_id': job.id}), REQUEST_CREATED




