from flask import Blueprint, current_app, request, send_file
from routes.helpers.submodules.auth import require_api_key
from routes.helpers.compiler import ModelCompiler
from routes.helpers.jobs import TrainingConfigPackager
from routes.helpers.project import Project
from routes.helpers.submodules.sanitation import sanitize_input

import logging

import os

from redis import Redis

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

project = Blueprint('project', __name__)

# Configuration for Redis connection
redis_host: str = 'localhost'
redis_port: int = 6379
redis_db: int = 0

# Create a Redis connection using environment variables
redis_client = Redis(
    host=os.getenv('REDIS_HOST', redis_host),
    port=int(os.getenv('REDIS_PORT', str(redis_port))),
    decode_responses=True
)


def get_project_from_redis(api_key, project_id):
    try:
        if redis_client.json().arrindex(f"api_key:{api_key}", '$.project_keys', project_id) is None:
            return {'error': 'Project does not exist'}, BAD_REQUEST, None
        existing_project = redis_client.json().get(f"project:{project_id}")
        if existing_project is None:
            return {'error': 'Project does not exist'}, BAD_REQUEST, None
    except Exception as error:
        logging.error(f'Failed to retrieve project from Redis, {error}')
        return {'error': 'A database connection error occurred, project has not been retrieved'}, BAD_REQUEST, None

    return existing_project, REQUEST_SUCCEEDED, None


@project.route('/project', methods=['POST'])
@require_api_key
def create_project(api_key):
    dataset_id = int(sanitize_input(request.args.get('dataset_id')))

    dataset_key = redis_client.json().get(f"api_key:{api_key}", "$.dataset_keys")[dataset_id]

    if dataset_key is None:
        return {'error': 'Dataset key is required'}, BAD_REQUEST
    if not redis_client.exists(f"file:{api_key}:{dataset_key[0]}:meta"):
        return {'error': 'Dataset key does not exist'}, BAD_REQUEST

    if request.form.get('title') is not None:
        title = sanitize_input(request.form.get('title'))
    else:
        title = None

    if request.form.get('description') is not None:
        description = sanitize_input(request.form.get('description'))
    else:
        description = None

    new_project = Project(dataset_key[0], title, description)
    new_project.get_dataset_fields(api_key)

    try:
        redis_client.json().set(f"project:{new_project.project_key}", '$', new_project.serialize())
        redis_client.json().arrappend(f"api_key:{api_key}", '$.project_keys', new_project.project_key)
    except Exception as error:
        logging.error(f'Failed to save project to Redis, {error}')
        return {'error': 'A database connection error occurred, project has not been created'}, BAD_REQUEST

    return {'info': 'Project created', 'key': new_project.project_key}, REQUEST_CREATED

@project.route('/project/<project_id>', methods=['DELETE'])
@require_api_key
def delete_project(project_id, api_key):
    try:
        existing_project = redis_client.json().get(f"project:{project_id}")
        if existing_project is None:
            return {'error': 'Project does not exist'}, BAD_REQUEST
        redis_client.delete(f"project:{project_id}")
        redis_client.json().arrpop(f"api_key:{api_key}", '$.project_keys', redis_client.json().arrindex(f"api_key:{api_key}", '$.project_keys', project_id)[0])
    except Exception as error:
        logging.error(f'Failed to delete project from Redis, {error}')
        return {'error': 'A database connection error occurred, project has not been deleted'}, BAD_REQUEST

    return {'info': 'Successfully deleted the project'}

@project.route('/project/<project_id>', methods=['GET'])
@require_api_key
def get_project(project_id, api_key):
    existing_project, status, error = get_project_from_redis(api_key, project_id)
    if error:
        return existing_project, status

    return existing_project, REQUEST_SUCCEEDED

@project.route('/project/list', methods=['GET'])
@require_api_key
def list_projects(api_key):
    try:
        project_keys = redis_client.json().get(f"api_key:{api_key}", "$.project_keys")[0]
    except Exception as error:
        logging.error(f'Failed to retrieve project list from Redis, {error}')
        return {'error': 'A database connection error occurred, project list has not been retrieved'}, BAD_REQUEST

    if project_keys is None:
        return {'projects': {}}, REQUEST_SUCCEEDED

    project_info = {}
    for project_key in project_keys:
        existing_project = redis_client.json().get(f"project:{project_key}")
        project_info[project_key] = {'title': existing_project['title'], 'description': existing_project['description']}


    return {'projects': project_info}, REQUEST_SUCCEEDED

@project.route('/project/<project_id>', methods=['PUT'])
@require_api_key
def update_project_settings(project_id):
    # update project settings

    return {}, REQUEST_SUCCEEDED


@project.route('/project/<project_id>/feature/<field>', methods=['POST'])
@require_api_key
def add_feature(api_key, project_id, field):
    existing_project, status, error = get_project_from_redis(api_key, project_id)
    if error:
        return existing_project, status

    logging.info(Project.deserialize(existing_project))

    return {}, REQUEST_SUCCEEDED