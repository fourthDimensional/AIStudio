import os

from flask import Blueprint, current_app, request, send_file
from routes.helpers.submodules.auth import require_api_key
import routes.helpers.model as model_interface

from routes.helpers.submodules import data_proc, utils
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

AUTHKEY_HEADER = 'authkey'

logger = logging.getLogger(__name__)

