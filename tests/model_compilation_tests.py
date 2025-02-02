import pytest
from unittest.mock import patch, MagicMock
from routes.helpers.compiler import ModelCompiler
from routes.helpers.model import ModelWrapper, DataProcessingEngine, LayerManipulator

@pytest.fixture
def mock_redis():
    with patch('routes.helpers.compiler.Redis') as mock:
        mock_instance = mock.return_value
        mock_instance.exists.return_value = False
        mock_instance.json.return_value.set.return_value = True
        mock_instance.json.return_value.get.return_value = None
        yield mock_instance

@pytest.fixture
def model_wrapper():
    data_processing_engine = DataProcessingEngine()
    layer_manipulator = LayerManipulator()
    return ModelWrapper(data_processing_engine, layer_manipulator)

def test_compile_model(mock_redis, model_wrapper):
    compiler = ModelCompiler()
    redis_connection = {'host': 'localhost', 'port': 6379, 'db': 0}

    compiled_model = compiler.compile_model(model_wrapper, redis_connection)

    assert compiled_model is not None
    mock_redis.json.return_value.set.assert_called_once()
    mock_redis.expire.assert_called_once()

def test_compile_model_with_existing_cache(mock_redis, model_wrapper):
    mock_redis.exists.return_value = True
    mock_redis.json.return_value.get.return_value = model_wrapper.serialize()

    compiler = ModelCompiler()
    redis_connection = {'host': 'localhost', 'port': 6379, 'db': 0}

    compiled_model = compiler.compile_model(model_wrapper, redis_connection)

    assert compiled_model is not None
    mock_redis.json.return_value.get.assert_called_once()

def test_compile_model_with_dense_layers(mock_redis, model_wrapper):
    layer_manipulator = model_wrapper.layer_manipulator

    # Add layers to the model
    input_layer = InputLayer(input_size=5)
    dense_layer = DenseLayer(units=10)
    layer_manipulator.add_layer(input_layer, 0, 0)
    layer_manipulator.forward_layer(0, 0)
    layer_manipulator.add_layer(dense_layer, 1, 0)

    compiler = ModelCompiler()
    redis_connection = {'host': 'localhost', 'port': 6379, 'db': 0}

    compiled_model = compiler.compile_model(model_wrapper, redis_connection)

    assert compiled_model is not None
    mock_redis.json.return_value.set.assert_called_once()
    mock_redis.expire.assert_called_once()

def test_compile_model_with_gru_layers(mock_redis, model_wrapper):
    layer_manipulator = model_wrapper.layer_manipulator

    # Add layers to the model
    input_layer = InputLayer(input_size=5)
    gru_layer = GRULayer(units=5)
    layer_manipulator.add_layer(input_layer, 0, 0)
    layer_manipulator.forward_layer(0, 0)
    layer_manipulator.add_layer(gru_layer, 1, 0)

    compiler = ModelCompiler()
    redis_connection = {'host': 'localhost', 'port': 6379, 'db': 0}

    compiled_model = compiler.compile_model(model_wrapper, redis_connection)

    assert compiled_model is not None
    mock_redis.json.return_value.set.assert_called_once()
    mock_redis.expire.assert_called_once()

def test_compile_model_with_complex_layers(mock_redis, model_wrapper):
    layer_manipulator = model_wrapper.layer_manipulator

    # Add layers to the model
    input_layer = InputLayer(input_size=5)
    dense_layer = DenseLayer(units=10)
    batch_norm_layer = BatchNormalizationLayer()
    reshape_layer = ReshapeLayer(target_shape=(1, 5))
    gru_layer = GRULayer(units=5)
    flatten_layer = FlattenLayer()

    layer_manipulator.add_layer(input_layer, 0, 0)
    layer_manipulator.forward_layer(0, 0)
    layer_manipulator.add_layer(batch_norm_layer, 1, 0)
    layer_manipulator.forward_layer(1, 0)
    layer_manipulator.add_layer(reshape_layer, 2, 0)
    layer_manipulator.forward_layer(2, 0)
    layer_manipulator.add_layer(gru_layer, 3, 0)
    layer_manipulator.forward_layer(3, 0)
    layer_manipulator.add_layer(flatten_layer, 4, 0)
    layer_manipulator.forward_layer(4, 0)
    layer_manipulator.add_layer(dense_layer, 5, 0)

    compiler = ModelCompiler()
    redis_connection = {'host': 'localhost', 'port': 6379, 'db': 0}

    compiled_model = compiler.compile_model(model_wrapper, redis_connection)

    assert compiled_model is not None
    mock_redis.json.return_value.set.assert_called_once()
    mock_redis.expire.assert_called_once()