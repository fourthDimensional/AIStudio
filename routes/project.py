from flask import Blueprint, current_app, request, send_file
from routes.helpers.submodules.auth import require_api_key
from routes.helpers.project import Project

import logging

import os

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
@project.route('/project/<project_id>', methods=['POST'])
def create_project(project_id):
    # create new project

    # register project in database

    # return project id

    return error, REQUEST_CREATED

@require_api_key
@project.route('/project/<project_id>', methods=['DELETE'])
def delete_project(project_id):
    # delete project

    return {'info': 'Successfully deleted the project'}

@require_api_key
@project.route('/project/<project_id>', methods=['GET'])
def get_project(project_id):
    # return project information

    return {}, REQUEST_SUCCEEDED

@require_api_key
@project.route('/project/list', methods=['GET'])
def list_projects():
    # return list of projects

    return {}, REQUEST_SUCCEEDED

@require_api_key
@project.route('/project/<project_id>', methods=['PUT'])
def update_project_settings(project_id):
    # update project settings

    return {}, REQUEST_SUCCEEDED