import os
import traceback
import secrets
import time
import logging
from flask import request
from typing import Callable, Any, Dict
from functools import wraps

from redis import Redis, RedisError

# Configuration for Redis connection
redis_host: str = 'localhost'
redis_port: int = 6379
redis_db: int = 0

# Create a Redis connection using environment variables
redis_client = Redis(
    host=os.getenv('REDIS_HOST', redis_host),
    port=int(os.getenv('REDIS_PORT', str(redis_port))),
    password=os.getenv('REDIS_PASSWORD'),
    decode_responses=True
)

logger: logging.Logger = logging.getLogger(__name__)


def generate_api_key() -> str:
    """
    Generates a secure, random API key.

    :return: A 32-character hexadecimal string representing the generated API key.
    """
    return secrets.token_hex(16)


def save_api_key(api_key: str, env: str, first_name: str, last_name: str, email: str) -> str:
    """
    Saves the API key along with associated metadata to Redis.

    :param api_key: The API key to save.
    :param env: The environment where the API key is used (e.g., production, development).
    :param first_name: The first name of the API key owner.
    :param last_name: The last name of the API key owner.
    :param email: The email address of the API key owner.

    :return: The API key that was saved.
    """
    key_structure: Dict[str, Any] = {
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

    try:
        redis_client.json().set(f"api_key:{api_key}", '$', key_structure)
    except RedisError as e:
        logging.error(f"Failed to save API key {api_key} to Redis: {e}")

    return api_key


def is_valid_api_key(api_key: str) -> bool:
    """
    Checks if the provided API key is valid and active.

    :param api_key: The API key to validate.

    :return: True if the API key is valid; False otherwise.
    """
    logging.info("QUERYing for API KEY provided by request")
    result: any = None

    try:
        result = redis_client.json().get("api_key:{}".format(api_key), '$.active')
    except RedisError as e:
        logging.error(f"Failed to verify API key {api_key} from Redis: {e}")

    return False if result is None else True


def delete_api_key(api_key: str) -> int:
    """
    Deletes the specified API key from Redis.

    :param api_key: The API key to delete.

    :return: The result of the deletion operation from Redis.
    """
    result: bool = False
    try:
        result = redis_client.json().delete("api_key:{}".format(api_key), '$')
    except RedisError as e:
        logging.error(f"Failed to delete API key {api_key} on Redis: {e}")
    return result


def require_api_key(view_func: Callable) -> Callable:
    """
    Decorator function that requires an API key for accessing a view.

    :param view_func: The view function to wrap with API key authentication.

    :return: The wrapped view function with API key authentication.
    """

    @wraps(view_func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        api_key = request.headers.get('authkey')
        logging.info('Checking if API KEY is valid')

        if not is_valid_api_key(api_key):
            return {'error': 'Invalid API Key'}, 401

        logging.info('API KEY verified')
        try:
            response = view_func(*args, **kwargs)
            update_api_key_metadata(api_key, success=True)
            return response
        except Exception as e:
            update_api_key_metadata(api_key, success=False)
            raise e
            # return {'internal_error': str(e), 'traceback': traceback.format_exc()}, 500

    return wrapper


def update_api_key_metadata(api_key: str, success: bool = True) -> None:
    """
    Updates the metadata associated with an API key after a request.

    :param api_key: The API key whose metadata is to be updated.
    :param success: Indicates whether the request was successful.
    """
    logging.info("QUERYing for API KEY metadata")

    try:
        if success:
            redis_client.json().numincrby("api_key:{}".format(api_key), '$.usage.successful_requests', 1)
        redis_client.json().numincrby("api_key:{}".format(api_key), '$.usage.total_requests', 1)
    except RedisError as e:
        logging.error(f"Failed to delete API key {api_key} on Redis: {e}")

    logging.info('Updated API KEY metadata')
