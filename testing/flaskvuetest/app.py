from flask import Flask, jsonify, request
from flask_cors import CORS

BOOKS = [
    {
        'title': 'On the Road',
        'author': 'Jack Kerouac',
        'read': True
    },
    {
        'title': 'Harry Potter and the Philosopher\'s Stone',
        'author': 'J. K. Rowling',
        'read': False
    },
    {
        'title': 'Green Eggs and Ham',
        'author': 'Dr. Seuss',
        'read': True
    }
]

# instantiate the app
app = Flask(__name__)
app.config.from_object(__name__)

# enable CORS
CORS(app, resources={r'/*': {'origins': '*'}})


# sanity check route
@app.route('/ping', methods=['GET'])
def ping_pong():
    return jsonify('L bozo')

@app.route('/books', methods=['GET', 'POST'])
def all_books():
    response_object = {'status': 'success'}
    if request.method == 'POST':
        post_data = request.get_json()
        BOOKS.append({
            'title': post_data.get('title'),
            'author': post_data.get('author'),
            'read': post_data.get('read')
        })
        response_object['message'] = 'Book added!'
    else:
        response_object['books'] = BOOKS
    return jsonify(response_object)

api_keys = ['test1', 'test2', 'test3']

@app.route('/add', methods=['GET'])
def add():
    api_key = request.headers.get('API-Key')
    if api_key not in api_keys:
        return {'error': 'Invalid API Key'}, 401
    param1 = request.args.get('one')
    param2 = request.args.get('two')   
    return jsonify(int(param1) + int(param2)), 200

if __name__ == '__main__':
    app.run(port=5001)