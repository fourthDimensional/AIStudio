import rq.command
from flask import Blueprint, current_app, request, send_file
from routes.helpers.submodules.auth import require_api_key

from redis import Redis
from rq import Queue, Worker

import logging

import os

# Configuration for Redis connection
redis_host: str = 'localhost'
redis_port: int = 6379
redis_db: int = 0

redis_queue = Queue(connection=Redis())

REDIS_CONNECTION_INFO = {
    'host': os.getenv('REDIS_HOST', redis_host),
    'port': int(os.getenv('REDIS_PORT', str(redis_port))),
    'decode_responses': True
}

workers = Blueprint('workers', __name__)

REQUEST_SUCCEEDED = 200
REQUEST_CREATED = 201

BAD_REQUEST = 400
UNAUTHENTICATED_REQUEST = 401
FORBIDDEN_REQUEST = 403
PAGE_NOT_FOUND = 404
REQUEST_CONFLICT = 409

REQUEST_NOT_IMPLEMENTED = 501

AUTHKEY_HEADER = 'authkey'

logger = logging.getLogger(__name__)

@workers.route('/workers', methods=['GET'])
def get_all_workers():
    worker_list = Worker.all(connection=redis_queue.connection)
    workers_info = []
    for worker in worker_list:
        worker_info = {'name': worker.name, 'state': worker.get_state(),
                       'successful_jobs': worker.successful_job_count, 'failed_jobs': worker.failed_job_count,
                       'current_job': worker.get_current_job_id(),
                       'queues': [queue.name for queue in worker.queues], 'last_heartbeat': worker.last_heartbeat}

        workers_info.append(worker_info)

    return workers_info, REQUEST_SUCCEEDED

# route that counts all workers
@workers.route('/workers/count', methods=['GET'])
def get_worker_count():
    worker_list = Worker.all(connection=redis_queue.connection)
    return {'count': len(worker_list)}, REQUEST_SUCCEEDED

# route that deletes worker from worker name
@workers.route('/worker/<worker_name>', methods=['DELETE'])
def delete_worker(worker_name):
    rq.command.send_shutdown_command(connection=redis_queue.connection, worker_name=worker_name)
    return {'info': 'Successfully deleted worker'}, REQUEST_SUCCEEDED

# route that gets ALL information on a specific job from an ID
@workers.route('/job/<job_id>', methods=['GET'])
def get_job(job_id):
    try:
        job = rq.job.Job.fetch(job_id, connection=redis_queue.connection)
    except rq.exceptions.NoSuchJobError:
        return {'error': 'Job not found'}, REQUEST_CONFLICT

    job_info = {'id': job.id, 'created_at': job.created_at, 'enqueued_at': job.enqueued_at,
                'ended_at': job.ended_at, 'origin': job.origin, 'result': job.return_value(),
                'description': job.description, 'timeout': job.timeout,
                'status': job.get_status(), 'meta': job.meta,
                'position': job.get_position()}

    return job_info, REQUEST_SUCCEEDED
