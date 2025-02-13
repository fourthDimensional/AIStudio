import secrets
import os
from routes.helpers.compiler import ModelCompiler
from routes.helpers.jobs import TrainingConfigPackager

from redis import Redis

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

"""
WIP Up-to-date Project Class Code

Currently being written.

Needs to be integrated into the project creation process across the codebase.

Will be used to store and manage projects and their associated models and features.
"""

class Project:
    def __init__(self, dataset_key: str, model_compiler: ModelCompiler, training_config_packager: TrainingConfigPackager,
                 title: str = None, description: str = None, project_key: str = str(secrets.token_hex(nbytes=4)),
                 training_history_key: str = str(secrets.token_hex(nbytes=4))):
        self.model_registry: dict = {}
        self.feature_registry: dict = {}

        self.title = title
        self.description = description

        self.dataset_key: str = dataset_key
        self.project_key: str = project_key
        self.training_history_key = training_history_key

        self.model_compiler: ModelCompiler = model_compiler
        self.training_config_packager: TrainingConfigPackager = training_config_packager

    def init_training_history(self):
        redis_client.json().set(f"training_history:{self.training_history_key}", '$', [])

    def serialize(self):
        return {
            's_ver': 1,
            'title': self.title,
            'description': self.description,
            'dataset_key': self.dataset_key,
            'project_key': self.project_key,
            'training_history_key': self.training_history_key,
            'models': {model_key: model.serialize() for model_key, model in self.model_registry.items()},
            'features': {feature_key: feature.serialize() for feature_key, feature in self.feature_registry.items()}
        }

    @staticmethod
    def deserialize(cls):
        project = cls()

        match cls['s_ver']:
            case 1:
                project.dataset_key = cls['dataset_key']
                project.project_key = cls['project_key']
                project.title = cls['title']
                project.description = cls['description']
                project.training_history_key = cls['training_history_key']
                project.model_registry = {model_key: Model.deserialize(model) for model_key, model in cls['models'].items()}
                project.feature_registry = {feature_key: Feature.deserialize(feature) for feature_key, feature in cls['features'].items()}
                return project
            case _:
                logging.error('Unknown serialization version')
                return None





