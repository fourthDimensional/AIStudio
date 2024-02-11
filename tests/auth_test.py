import pytest
from unittest.mock import patch
from flask import Flask
import routes.helpers.auth as auth


@pytest.fixture
def mock_redis():
    """
    Provides a mocked Redis client that simulates Redis JSON operations.

    This fixture patches `routes.helpers.auth.redis_client` to return
    mocked responses for JSON-related operations such as get, set, delete,
    and numincrby. It yields the mock object to be used in tests.

    :return: Yields a mock Redis client object with predefined return values.
    """
    with patch('routes.helpers.auth.redis_client') as mock:
        mock.json.return_value.get.return_value = [True]  # Mimics returning a list of matching objects
        mock.json.return_value.set.return_value = True  # Mimics simple boolean success
        mock.json.return_value.delete.return_value = True  # Mimics boolean success
        mock.json.return_value.numincrby.return_value = "new_value"  # Mimics returning a new string
        yield mock


@pytest.fixture
def app():
    """
    Creates and returns a Flask application for testing.

    This fixture is used to create a Flask app context for route testing,
    ensuring that the Flask testing client can be used to simulate
    requests to the application.

    :return: A Flask application instance.
    """
    app = Flask(__name__)
    return app


def test_generate_api_key():
    """
    Tests the generation of a secure, random API key.

    Asserts that the generated API key is a string of length 32,
    indicating a 16-byte hexadecimal representation.
    """
    api_key = auth.generate_api_key()
    assert len(api_key) == 32


def test_save_api_key(mock_redis):
    """
    Tests saving an API key along with associated metadata to Redis.

    This test checks that the save_api_key function attempts to save the
    correct structure to Redis and asserts that the operation is successful.

    :param mock_redis: The mock Redis client fixture.
    """
    api_key = auth.save_api_key("testkey", "dev", "John", "Doe", "john.doe@example.com")
    assert api_key == "testkey"
    mock_redis.json.return_value.set.assert_called_once()


def test_is_valid_api_key(mock_redis):
    """
    Tests the validation of an API key's existence and active status.

    Asserts that the function returns True when the API key is found
    and marked as active in Redis.

    :param mock_redis: The mock Redis client fixture.
    """
    valid = auth.is_valid_api_key("existingkey")
    assert valid is True
    mock_redis.json.return_value.get.assert_called_with("api_key:existingkey", '$.active')


def test_delete_api_key(mock_redis):
    """
    Tests the deletion of a specified API key from Redis.

    Asserts that the delete operation is called with the correct key
    and returns a successful result.

    :param mock_redis: The mock Redis client fixture.
    """
    result = auth.delete_api_key("keytodelete")
    assert result is True
    mock_redis.json.return_value.delete.assert_called_with("api_key:keytodelete", '$')


def test_require_api_key_decorator_valid(app, mock_redis):
    """
    Tests the API key requirement decorator with a valid API key.

    Simulates a request to a protected route with a valid API key and
    asserts that access is granted and the correct response is returned.

    :param app: The Flask application fixture.
    :param mock_redis: The mock Redis client fixture.
    """
    mock_redis.json.return_value.get.return_value = [True]  # API key exists and is active

    @app.route('/test')
    @auth.require_api_key
    def test_route():
        return "Success", 200

    with app.test_client() as client:
        response = client.get('/test', headers={'authkey': 'validkey'})
        assert response.status_code == 200
        assert response.data.decode() == "Success"


def test_require_api_key_decorator_invalid(app, mock_redis):
    """
    Tests the API key requirement decorator with an invalid API key.

    Simulates a request to a protected route with an invalid API key and
    asserts that access is denied with the correct error response.

    :param app: The Flask application fixture.
    :param mock_redis: The mock Redis client fixture.
    """
    mock_redis.json.return_value.get.return_value = None  # API key does not exist or is not active

    @app.route('/test')
    @auth.require_api_key
    def test_route():
        return "Unauthorized", 401

    with app.test_client() as client:
        response = client.get('/test', headers={'authkey': 'invalidkey'})
        assert response.status_code == 401
        assert response.json == {"error": "Invalid API Key"}


def test_update_api_key_metadata_success(mock_redis):
    """
    Tests updating API key metadata upon a successful request.

    This test checks that the successful_requests and total_requests
    counters are incremented correctly in Redis.

    :param mock_redis: The mock Redis client fixture.
    """
    auth.update_api_key_metadata("keywithsuccess", True)
    calls = [call for call in mock_redis.json.return_value.numincrby.mock_calls]
    assert len(calls) == 2  # Check if it incremented both successful and total requests
    assert calls[0][1][2] == 1  # Success increment by 1
    assert calls[1][1][2] == 1  # Total increment by 1


def test_update_api_key_metadata_failure(mock_redis):
    """
    Tests updating API key metadata upon a failed request.

    This test checks that only the total_requests counter is incremented
    in Redis when a request fails.

    :param mock_redis: The mock Redis client fixture.
    """
    auth.update_api_key_metadata("keywithfailure", False)
    mock_redis.json.return_value.numincrby.assert_called_with("api_key:keywithfailure", '$.usage.total_requests', 1)
