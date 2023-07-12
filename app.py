from flask import Flask
from flask_cors import CORS
from routes.helpers import utils

from routes.data import data_views
from routes.layers import layers
from routes.model_basic import model_basic

# instantiate the app
app = Flask(__name__)
app.config.from_object(__name__)
app.config['MAX_CONTENT_LENGTH'] = 300 + (1024 * 1024)  # Basic request size + large dataset limit

api_keys = utils.grab_api_keys()

# enable CORS
CORS(app, resources={r'/*': {'origins': '*'}})

UPLOAD_FOLDER = 'static/files'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

REQUEST_SUCCEEDED = 200
REQUEST_CREATED = 201

BAD_REQUEST = 400
UNAUTHENTICATED_REQUEST = 401
FORBIDDEN_REQUEST = 403
PAGE_NOT_FOUND = 404
REQUEST_CONFLICT = 409

REQUEST_NOT_IMPLEMENTED = 501


app.register_blueprint(model_basic)
app.register_blueprint(layers)
app.register_blueprint(data_views)
