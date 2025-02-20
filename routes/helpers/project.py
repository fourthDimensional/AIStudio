import secrets
import os
from routes.helpers.compiler import ModelCompiler
from routes.helpers.model import ModelWrapper
from routes.helpers.jobs import TrainingConfigPackager

import pandas as pd

from io import BytesIO

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
    def __init__(self, dataset_key: str, title: str = None, description: str = None, project_key: str = str(secrets.token_hex(nbytes=4)),
                 training_history_key: str = str(secrets.token_hex(nbytes=4))):
        self.model_registry: dict[str, ModelWrapper] = {}
        self.features: list[str] = []
        self.dataset_fields: list[str] = []

        self.title = title
        self.description = description

        self.dataset_key: str = dataset_key
        self.project_key: str = project_key
        self.training_history_key = training_history_key

    def get_dataset_fields(self, api_key: str):
        self.dataset_fields = redis_client.json().get(f"file:{api_key}:{self.dataset_key}:meta")['columns']

    def initialize_training_history(self):
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
            'features': self.features,
            'dataset_fields': self.dataset_fields
        }

    @staticmethod
    def deserialize(cls):
        project = Project(
            dataset_key=cls['dataset_key'],
            title=cls['title'],
            description=cls['description'],
            project_key=cls['project_key'],
            training_history_key=cls['training_history_key'],
        )

        for model_key, model in cls['models'].items():
            project.model_registry[model_key] = ModelWrapper.deserialize(model)

        project.features = cls['features']
        project.dataset_fields = cls['dataset_fields']

        return project





