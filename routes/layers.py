import os
import logging

from flask import Blueprint, current_app, request
from routes.helpers.submodules.auth import require_api_key

from routes.helpers.submodules import utils

layers = Blueprint('layers', __name__)

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
