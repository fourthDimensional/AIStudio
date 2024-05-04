import os
import logging

import redis
from flask import Flask, jsonify, abort
from flask_cors import CORS

from routes.data import data_views
from routes.layers import layers
from routes.basic import model_basic
from pathlib import Path
from logging.handlers import RotatingFileHandler

from flask_talisman import Talisman
from werkzeug.middleware.profiler import ProfilerMiddleware

from routes.helpers.submodules.auth import generate_api_key, save_api_key

# instantiate the app
app = Flask(__name__)

# Talisman(app)

redis_host: str = 'localhost'
redis_port: int = 6379
redis_db: int = 0

app.config['DATABASE'] = redis.Redis(
    host=os.getenv('REDIS_HOST', redis_host),
    port=int(os.getenv('REDIS_PORT', str(redis_port))),
    decode_responses=True
)

app.config.from_object(__name__)
app.config['MAX_CONTENT_LENGTH'] = 300 + (1024 * 1024 * 200)  # Basic request size + large dataset limit
profile_dir = (Path(__file__).resolve().parent / 'profiles').resolve()

if not os.path.exists(profile_dir):
    os.makedirs(profile_dir)

# Use ProfilerMiddleware
app.wsgi_app = ProfilerMiddleware(
    app.wsgi_app,
    profile_dir=profile_dir,
    stream=None
)

# enable CORS
CORS(app, resources={r'/*': {'origins': '*'}})

UPLOAD_FOLDER = 'static/files'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

app.register_blueprint(model_basic)
app.register_blueprint(layers)
app.register_blueprint(data_views)

save_api_key(generate_api_key(), 'development', 'test', 'user', 'test@email.com')

def setup_logging():
    # Create the root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Create a formatter with 12-hour clock format
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] - (Line: %(lineno)d [%(filename)s]) - %(message)s',
                                  datefmt='%Y-%m-%d %I:%M:%S %p')

    # Create a rotating file handler
    file_handler = RotatingFileHandler('app.log', maxBytes=10 * 1024 * 1024, backupCount=5)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # Add the handlers to the root logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


setup_logging()

@app.route('/authenticate', methods=['GET'])
def authenticate():
    data = request.get_json()
    username = data['username']
    password = data['password']



def main():
    app.run(port=5000)
