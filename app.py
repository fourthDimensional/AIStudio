from flask import Flask
from flask_cors import CORS

from routes.data import data_views
from routes.helpers import utils
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

app.register_blueprint(model_basic)
app.register_blueprint(layers)
app.register_blueprint(data_views)
