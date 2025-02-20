import pandas as pd
import json
from redis import Redis
import socket
from rq import get_current_job

def load_dataset(name: str) -> pd.DataFrame:
    """
    Load a dataset from the datasets folder.

    :param name: The name of the dataset to load.
    :return: The loaded dataset.
    """
    return pd.read_csv(f'static/datasets/{name}.csv')


def generate_profile_report(name: str, apikey: str, redis_connection_info: dict) -> None:
    """
    Generate a profile report for a dataset and save it to Redis.

    :param redis_connection_info:
    :param apikey: Owner of the dataset
    :param name: The name of the dataset to generate a profile report for.
    """
    redis_client = Redis(**redis_connection_info)
    job = get_current_job()

    df = load_dataset(name)
    metadata = df.describe(include='all').to_dict()

    job.meta['handled_by'] = socket.gethostname()
    job.meta.update(metadata)
    job.save_meta()

    # profile = ProfileReport(df, title=f"Profile Report for {name}")
    json_data = profile.to_json()
    redis_client.json().set(f"profile_report:{apikey}:{name}", '$', json.loads(json_data)) # TODO ADD CONSISTENT STRING CLEANING

    return f"profile_report:{apikey}:{name}"
    app.register_blueprint(project)