from flask import Blueprint, current_app, request, send_file
from routes.helpers.submodules.auth import require_api_key
from routes.helpers.project import Project

import logging

import os

# Configuration for Redis connection
redis_host: str = 'localhost'
redis_port: int = 6379
redis_db: int = 0

REDIS_CONNECTION_INFO = {
    'host': os.getenv('REDIS_HOST', redis_host),
    'port': int(os.getenv('REDIS_PORT', str(redis_port))),
    'decode_responses': True
}

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

@require_api_key
@project.route('/project', methods=['POST'])
def create_project():
    api_key = request.headers.get(AUTHKEY_HEADER)
    project_name = request.form.get('name')

    # create new project

    # register project in database

    # return project id

    return error, REQUEST_CREATED

@require_api_key
@project.route('/project', methods=['DELETE'])
def delete_project():
    api_key = request.headers.get(AUTHKEY_HEADER)
    project_id = request.form.get('id')

    # delete project

    return {'info': 'Successfully deleted the project'}

@require_api_key
@project.route('/project', methods=['GET'])
def get_project():
    api_key = request.headers.get(AUTHKEY_HEADER)
    project_id = request.form.get('id')

    # return project information

    return project, REQUEST_SUCCEEDED