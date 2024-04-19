import json
import os
from pathlib import Path
import pytest
import jsonpickle
import hmac
import hashlib
from routes.helpers.submodules import utils as utils
import fakeredis
from redis.commands.json.path import Path

# Update these variables as needed
SECRET_KEY = 'cheese'
utils.SECRET_KEY = SECRET_KEY

# Sample data for use in tests
VALID_API_KEY = "1c9962737975ab0fc0cfd3dad6028e30"
UUID = "8a662bfe"
MODEL_ID = "1"
MODEL = {"name": "test_model"}
API_KEY_STRUCTURE = {'uuid': UUID}
SERIALIZED_MODEL = jsonpickle.encode(MODEL, unpicklable=False)  # Ensuring it's a string
SIGNATURE = hmac.new(SECRET_KEY.encode(), SERIALIZED_MODEL.encode(), hashlib.sha256).hexdigest()


@pytest.fixture
def redis_setup():
    """
    Set up a fake Redis instance using fakeredis for testing purposes.

    This fixture initializes a fakeredis instance and populates it with
    predefined API key and model data, including the serialized model and its signature.
    After the test, it flushes the fake Redis database to ensure cleanliness for subsequent tests.
    """
    fake_redis = fakeredis.FakeRedis(decode_responses=True)
    fake_redis.json().set(f"api_key:{VALID_API_KEY}", Path.root_path(), API_KEY_STRUCTURE)

    model_data_key = f"model:{UUID}:{MODEL_ID}:data"
    model_signature_key = f"model:{UUID}:{MODEL_ID}:signature"
    fake_redis.json().set(model_data_key, Path.root_path(), json.loads(SERIALIZED_MODEL))
    fake_redis.set(model_signature_key, SIGNATURE)

    yield fake_redis

    fake_redis.flushall()


@pytest.mark.parametrize("input,expected", [
    ("valid_naming_convention", True),
    ("InvalidNaming", False),
])
def test_check_naming_convention(input, expected):
    """
    Test the naming convention checker with various inputs.

    Verifies that the function correctly identifies valid and invalid naming conventions.

    :param input: The string to be checked for naming convention adherence.
    :param expected: The expected boolean result indicating naming convention adherence.
    """
    assert utils.check_naming_convention(input) == expected


def test_get_uuid(redis_setup):
    """
    Verify fetching of UUID using a valid API key from Redis.

    Checks that the UUID associated with a valid API key is correctly retrieved from the fake Redis instance.

    :param redis_setup: The fake Redis setup fixture used for testing.
    """
    assert utils.get_uuid(VALID_API_KEY, redis_setup) == UUID


def test_verify_signature():
    """
    Test the signature verification process for both valid and invalid signatures.

    This test ensures that the utility function accurately verifies the integrity of a given signature
    against a known secret key and serialized model.
    """
    assert utils.verify_signature(SERIALIZED_MODEL, SIGNATURE.encode(), SECRET_KEY) is True
    incorrect_signature = "incorrect_signature".encode()  # Correctly encode the incorrect signature as bytes
    assert not utils.verify_signature(SERIALIZED_MODEL, incorrect_signature, SECRET_KEY)


def test_fetch_model_model_exists(redis_setup):
    """
    Test the fetching of a model from Redis when the model exists.

    Verifies that the correct model and status code are returned when the model is present in Redis.

    :param redis_setup: The fake Redis setup fixture used for testing.
    """
    model, status = utils.fetch_model(VALID_API_KEY, MODEL_ID, redis_setup, SECRET_KEY)
    assert status == 1
    assert model == MODEL


def test_fetch_model_model_not_exist(redis_setup):
    """
    Test the fetching of a model from Redis when the model does not exist.

    Ensures that the appropriate status code is returned when the requested model ID does not correspond to any stored model.

    :param redis_setup: The fake Redis setup fixture used for testing.
    """
    _, status = utils.fetch_model(VALID_API_KEY, "non_existing_model_id", redis_setup, SECRET_KEY)
    assert status == 0


def test_store_model(redis_setup):
    """
    Test the storage of a new model in Redis.

    Validates that a new model can be successfully serialized, stored, and then accurately retrieved from Redis.

    :param redis_setup: The fake Redis setup fixture used for testing.
    """
    new_model = {"name": "new_model"}
    new_model_id = "new_model_123"
    utils.store_model(VALID_API_KEY, new_model_id, new_model, redis_setup, SECRET_KEY)
    stored_model_key = f"model:{UUID}:{new_model_id}:data"
    # Ensure correct retrieval and comparison
    stored_model = redis_setup.json().get(stored_model_key, Path.root_path())
    assert stored_model == new_model


def test_delete_model_exists(redis_setup):
    """
    Test deletion of an existing model from Redis.

    Confirms that a model, once stored in Redis, can be successfully deleted and is no longer retrievable.

    :param redis_setup: The fake Redis setup fixture used for testing.
    """
    delete_model_key = f"model:{UUID}:{MODEL_ID}:data"
    # Ensure setup for deletion
    redis_setup.json().set(delete_model_key, Path.root_path(), json.loads(SERIALIZED_MODEL))
    assert utils.delete_model(VALID_API_KEY, MODEL_ID, redis_setup) == 1
    assert not redis_setup.exists(delete_model_key)


def test_delete_model_not_exist(redis_setup):
    """
    Test the deletion operation for a model that does not exist in Redis.

    Ensures that attempting to delete a non-existent model correctly returns a status indicating failure.

    :param redis_setup: The fake Redis setup fixture used for testing.
    """
    non_existing_model_id = "non_existing_model_id"
    assert utils.delete_model(VALID_API_KEY, non_existing_model_id, redis_setup) == 0


def test_find_index_of_specific_class_found():
    """
    Test the utility function for finding the index of a specific class instance in a list.

    This test verifies that when the specified class instance is present in the list, its index is correctly returned.
    """
    class TestClass:
        pass

    class OtherClass:
        pass

    given_list = [OtherClass(), TestClass(), OtherClass()]
    index = utils.find_index_of_specific_class(given_list, TestClass)
    assert index == 1


def test_find_index_of_specific_class_not_found():
    """
    Test the utility function for finding the index of a specific class instance in a list when it is not present.

    Verifies that the function returns None when the specified class instance is not in the list.
    """
    class TestClass:
        pass

    class OtherClass:
        pass

    given_list = [OtherClass(), OtherClass()]
    index = utils.find_index_of_specific_class(given_list, TestClass)
    assert index is None


@pytest.mark.parametrize("given_id,expected", [
    ("10", True),
    ("31", False),  # Assuming MAX_NETWORKS_PER_PERSON is 30
    ("notadigit", False),
    ("-1", False),
])
def test_check_id(given_id, expected):
    """
    Test the ID checker with various ID values.

    Ensures that the function correctly identifies valid and invalid IDs based on predefined criteria.

    :param given_id: The ID to be checked.
    :param expected: The expected boolean outcome of the check.
    """
    assert utils.check_id(given_id) == expected


def test_merge_lists_with_lists():
    """
    Test the list merging function with two lists.

    Verifies that two lists can be merged into a single list containing all elements from both.
    """
    a = [1, 2]
    b = [3, 4]
    result = utils.merge_lists(a, b)
    assert result == [1, 2, 3, 4]


def test_merge_lists_with_string():
    """
    Test the list merging function with a list and a string.

    Checks that a list and a string can be combined into a single list with the string treated as a single element.
    """
    a = [1, 2]
    b = "3"
    result = utils.merge_lists(a, b)
    assert result == [1, 2, "3"]


def test_fetch_model_security_warning(redis_setup):
    """
    Test the model fetching function with a security compromise scenario.

    Ensures that when the signature does not match due to tampering, the function returns an appropriate failure status.
    """
    model_key = f"model:{UUID}:{MODEL_ID}:data"
    signature_key = f"model:{UUID}:{MODEL_ID}:signature"

    # Setting up valid model data and an intentionally incorrect signature
    redis_setup.json().set(model_key, '$', ['{"name": "test_model"}'])
    redis_setup.set(signature_key, "incorrect_signature")

    # Attempting to fetch the model should fail due to signature verification failure
    model, status = utils.fetch_model(VALID_API_KEY, MODEL_ID, redis_setup, SECRET_KEY)
    assert model is None
    assert status == -2  # Assuming -1 indicates a failure due to security reasons


def test_convert_to_dataframe(tmp_path):
    """
    Test the conversion of CSV data into a pandas DataFrame.

    Validates that a CSV file with missing values is correctly converted into a DataFrame, dropping rows with missing values.
    """
    data_file = tmp_path / "test_data.csv"
    data_file.write_text("col1,col2\n1,2\n3,\n,5", encoding="utf-8")

    # Convert the dataset file to a pandas DataFrame
    dataframe = utils.convert_to_dataframe(str(data_file))

    # Check if the DataFrame is correctly created and rows with missing values are dropped
    assert len(dataframe) == 1  # Only one row should remain after dropping rows with missing values
    assert list(dataframe.columns) == ["col1", "col2"]
    assert dataframe.iloc[0]["col1"] == 1
    assert dataframe.iloc[0]["col2"] == 2


@pytest.fixture
def setup_environment(tmp_path):
    """
    Set up a temporary testing environment including a profiler directory and Flask environment settings.

    This fixture creates a temporary directory for profiler logs and sets the FLASK_ENV to development for the duration of the test.
    """
    profiler_dir = tmp_path / "profiles"
    profiler_dir.mkdir()
    for i in range(20):  # Assuming MAX_LOGS is less than 20
        log_file = profiler_dir / f"log{i}.prof"
        log_file.touch()

    # Set FLASK_ENV to "development" temporarily
    original_flask_env = os.environ.get("FLASK_ENV")
    os.environ["FLASK_ENV"] = "development"
    original_profiler_dir = utils.PROFILER_DIR  # Backup the original PROFILER_DIR
    utils.PROFILER_DIR = profiler_dir  # Set the PROFILER_DIR to the temporary directory

    yield profiler_dir  # This allows the test to use the temporary PROFILER_DIR

    # Cleanup
    os.environ["FLASK_ENV"] = original_flask_env if original_flask_env is not None else ""
    utils.PROFILER_DIR = original_profiler_dir  # Restore the original PROFILER_DIR


def test_verify_signature_development_mode(setup_environment):
    """
    Test the signature verification process in development mode.

    Ensures that in development mode, the signature verification always passes and old logs are cleaned up as expected.
    """
    secret_key = "test_secret"
    data = "test_data"
    signature = "test_signature".encode()  # Example signature. Adjust as needed.

    # Call verify_signature in development mode, which should also trigger cleanup_old_logs
    result = utils.verify_signature(data, signature, secret_key)
    assert result is True  # In development mode, it should return True unconditionally

    # Verify that logs were cleaned up (assuming MAX_LOGS < 20)
    log_files = list(setup_environment.iterdir())
    assert len(log_files) <= utils.MAX_LOGS  # Adjust the assertion based on your MAX_LOGS value
