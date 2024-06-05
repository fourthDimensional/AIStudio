import secrets
from routes.helpers.compiler import ModelCompiler
from routes.helpers.training import TrainingConfigPackager


class Project:
    def __init__(self, dataset_key: str, model_compiler: ModelCompiler, training_config_packager: TrainingConfigPackager):
        self.model_registry: dict = {}
        self.feature_registry: dict = {}

        self.dataset_key: str = dataset
        self.project_key: str = hex(secrets.token_hex(4)) # Generate a random project key

        self.model_compiler: ModelCompiler = model_compiler
        self.training_config_packager: TrainingConfigPackager = training_config_packager

