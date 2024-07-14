import os
import traceback
import secrets
import time
import logging
import inspect
from flask import request
from typing import Callable, Any, Dict
from functools import wraps, lru_cache

from redis import Redis, RedisError

"""
Up-to-date Authentication Code

Currently does not need modification

Has reasonable test coverage

Future Plans:
- Implement a more secure way to store API keys
- Stop directly using redis connection information. Use some sort of interface to connect to redis
- Implement a more secure way to store session tokens
"""


def generate_api_key() -> str:
    """
    Generates a secure, random API key.

    :return: A 32-character hexadecimal string representing the generated API key.
    """
    return secrets.token_hex(16)

# Configuration for Redis connection
redis_host: str = 'localhost'
redis_port: int = 6379
redis_db: int = 0

# Create a Redis connection using environment variables
redis_client = Redis(
    host=os.getenv('REDIS_HOST', redis_host),
    port=int(os.getenv('REDIS_PORT', str(redis_port))),
    decode_responses=True
)

logger: logging.Logger = logging.getLogger(__name__)

SESSION_TOKEN_TTL: int = 60 * 60 * 24 * 7  # 1 week in seconds




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
        },
        'active_tokens': [],
        'dataset_keys': [],
        'project_keys': [],
    }

    try:
        redis_client.json().set(f"api_key:{api_key}", '$', key_structure)
    except RedisError as e:
        logging.error(f"Failed to save API key {api_key} to Redis: {e}")

    logging.warning('Using temporary memory saving code to delete previous API-keys')
    keys: list = redis_client.keys('api_key:*')
    for key in keys:
        if key != f"api_key:{api_key}":
            redis_client.delete(key)

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


def deregister_session_token(session_token: str) -> int:
    """
    Removes the specified session token from Redis.

    :param session_token:
    """
    api_key = redis_client.json().get(f"session_token:{session_token}", '$.apikey')[0]
    token_index = redis_client.json().arrindex(f"api_key:{api_key}", '$.active_tokens', session_token)
    redis_client.json().arrpop(f"api_key:{api_key}", '$.active_tokens', token_index)
    return redis_client.delete(f"session_token:{session_token}")


def register_session_token(api_key: str) -> str:
    """
    Registers a new session token for the specified API key.

    :param api_key: The API key for which to register a session token.

    :return: The generated session token.
    """
    session_token: str = secrets.token_hex(16)
    try:
        try:
            active_tokens = redis_client.json().get(f"api_key:{api_key}", '$.active_tokens')
            for token in active_tokens:
                redis_client.delete(f"session_token:{token}")
        except RedisError as e:
            logging.error(f"Failed to delete existing session tokens for API key {api_key}: {e}")

        redis_client.json().set(f"session_token:{session_token}", '$', {
            'active': True,
            'apikey': api_key,
            'time_created': int(time.time())
        })
        redis_client.json().arrappend(f"api_key:{api_key}", '$.active_tokens', session_token)

        redis_client.expire(f"session_token:{session_token}", SESSION_TOKEN_TTL)
    except RedisError as e:
        logging.error(f"Failed to register session token for API key {api_key} on Redis: {e}")

    return session_token, api_key


def is_valid_session_token(session_token: str) -> bool:
    """
    Checks if the provided session token is valid and active.

    :param session_token: The session token to validate.

    :return: True if the session token is valid; False otherwise.
    """
    logging.info("QUERYing for session token provided by request")
    result: any = None

    try:
        result = redis_client.json().get("session_token:{}".format(session_token), '$')[0]
    except RedisError as e:
        logging.error(f"Failed to verify session token {session_token} from Redis: {e}")
    except TypeError:
        return False

    if result is None:
        return False

    if not is_valid_api_key(result['apikey']):
        return False

    if result['active'] is False:
        return False
    else:
        return True


def is_valid_auth(api_key: str, session_token: str) -> bool:
    """
    Checks if the provided session token is valid and active.

    :param api_key: The API key to validate.
    :param session_token: The session token to validate.

    :return: True if the session token or api-key is valid; False otherwise. Additionally, returns the api-key if the session token is valid.
    returns error if both are invalid.
    """
    if not is_valid_api_key(api_key):
        if session_token is not None:
            logging.info('API KEY is invalid, checking for session token')
            if not is_valid_session_token(session_token):
                return None, (None, None), ({'error': 'Invalid Session Token'}, 401)
            api_key = redis_client.json().get(f"session_token:{session_token}", '$.apikey')[0]
            logging.info('Session Token verified')
        else:
            return None, (None, None), ({'error': 'Invalid API Key'}, 401)

    return True, (api_key, redis_client.json().get(f'api_key:{api_key}')), None


def require_api_key(view_func: Callable) -> Callable:
    """
    Decorator function that requires an API key for accessing a view.

    :param view_func: The view function to wrap with API key authentication.

    :return: The wrapped view function with API key authentication.
    """

    @wraps(view_func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        session_token = request.cookies.get('session')
        api_key = request.headers.get('authkey', None)

        validity, api_key, error = is_valid_auth(api_key, session_token)

        if not validity:
            return error

        api_key, metadata = api_key

        logging.info('API KEY/Session Token verified')
        try:
            params = inspect.signature(view_func).parameters
            if 'api_key' in params:
                kwargs['api_key'] = api_key
            if 'metadata' in params:
                kwargs['metadata'] = metadata

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
