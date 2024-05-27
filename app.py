import os
import logging

import redis
from flask import Flask, jsonify, abort, request, make_response
from flask_cors import CORS

from routes.data import data_views
from routes.layers import layers
from routes.basic import model_basic
from routes.workers import workers
from routes.project import project

from pathlib import Path
from logging.handlers import RotatingFileHandler

from flask_talisman import Talisman
from werkzeug.middleware.profiler import ProfilerMiddleware

from routes.helpers.submodules.auth import generate_api_key, save_api_key, require_api_key, register_session_token, is_valid_api_key, deregister_session_token

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
CORS(app, resources={r'/*': {'origins': '*'}}, supports_credentials=True)

UPLOAD_FOLDER = 'static/files'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DATASET_FOLDER'] = 'static/datasets'

app.register_blueprint(model_basic)
app.register_blueprint(layers)
app.register_blueprint(data_views)
app.register_blueprint(workers)
app.register_blueprint(project)

# save_api_key(generate_api_key(), 'development', 'test', 'user', 'test@email.com')

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

@app.route('/generate_api_key', methods=['POST'])
def generate_key():

    logging.warning('The generate api key route was just used. Please remember to remove this route before deployment.')

    api_key = generate_api_key()
    save_api_key(api_key, 'development', 'test', 'user', 'test@email.com')
    return jsonify({'api_key': api_key}), 200


# locked testing api key verified route
@app.route('/authenticated', methods=['GET'])
@require_api_key
def authenticated():
    return jsonify({'message': 'API Key verified'}), 200


@app.route('/login', methods=['POST'])
def login():
    api_key = request.headers.get('authkey')

    if api_key is None:
        return jsonify({'error': 'No API Key provided'}), 401

    if not is_valid_api_key(api_key):
        return jsonify({'error': 'Invalid API Key'}), 401

    session_token, _ = register_session_token(api_key)

    response = make_response()
    response.set_cookie('session', value=session_token, secure=True, httponly=True, samesite='Strict')

    return response, 200


@app.route('/logout', methods=['POST'])
def logout():
    session_token = request.cookies.get('session')

    if session_token is None:
        return jsonify({'error': 'No session token provided'}), 401

    if not is_valid_session_token(session_token):
        return jsonify({'error': 'Invalid session token'}), 401

    # TODO add error handling here
    deregister_session_token(session_token)

    response = make_response()
    response.set_cookie('session', value='', secure=True, httponly=True, samesite='Strict')

    return response, 200


def main():
    app.run(port=5001)
