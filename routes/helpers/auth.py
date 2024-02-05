from flask import request, abort, jsonify
import redis
import secrets
import time
import logging

from functools import wraps

redis_host = 'localhost'
redis_port = 6379
redis_db = 0

# Create a Redis connection
redis_client = redis.Redis(
    host='redis-16232.c282.east-us-mz.azure.cloud.redislabs.com',
    port=16232,
    password='GmIvRW7nt8slcgNmm7gjaARC0rCsRA6y')

logger = logging.getLogger(__name__)

def generate_api_key():
    """Generate a random API key using secrets module."""
    return secrets.token_hex(16)


def save_api_key(api_key, env, first_name, last_name, email):
    """Save API key to Redis."""
    key_structure = {
        'active': True,
        'time_created': int(time.time()),
        'uuid': secrets.token_hex(4),
        'environment': env,
        'type': 'unlimited',
        'owner': {
            'first_name': first_name,
            'last_name': last_name,
            'email': email
        },
        'usage': {
            'total_requests': 0,
            'successful_requests': 0
        }
    }

    redis_client.json().set("api_key:{}".format(api_key), '$', key_structure)

    return api_key


def is_valid_api_key(api_key):
    """Check if the API key is valid."""
    logging.info("QUERYing for API KEY provided by request")
    return False if redis_client.json().get("api_key:{}".format(api_key), '$.active') is None else True


def delete_api_key(api_key):
    """Delete API key from Redis."""
    return redis_client.delete(api_key)


def require_api_key(view_func):
    """Decorator function to require API key for authentication."""

    @wraps(view_func)
    def wrapper(*args, **kwargs):
        api_key = request.headers.get('authkey')
        logging.info('Checking if API KEY is valid')

        if is_valid_api_key(api_key) == 0:
            return {'error': 'Invalid API Key'}, 401

        logging.info('API KEY verified')
        try:
            # Try executing the original view function
            response = view_func(*args, **kwargs)

            # If successful, update metadata
            update_api_key_metadata(api_key, success=True)

            return response
        except Exception as e:
            # If an exception occurs, update metadata with total requests only
            update_api_key_metadata(api_key, success=False)
            # return {'internal_error': str(e),
            #         'traceback': traceback.format_exc()}, 500
            raise e

    return wrapper


def update_api_key_metadata(api_key, success=True):
    logging.info("QUERYing for API KEY metadata")
    metadata = redis_client.json().get("api_key:{}".format(api_key), '$')
    logging.info('API Key Metadata: {}'.format(str(metadata)))

    if success:
        metadata[0]['usage']['successful_requests'] += 1

    metadata[0]['usage']['total_requests'] += 1

    redis_client.json().set("api_key:{}".format(api_key), '$', metadata[0])

    logging.info('Updated API KEY metadata')

    pass
