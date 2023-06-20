from flask import Flask, jsonify, request
from flask_cors import CORS

# instantiate the app
app = Flask(__name__)
app.config.from_object(__name__)

def convert_text_file_to_list(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        lines = [line.strip() for line in lines]
    return lines
text_file_path = 'apikeys.txt'
api_keys = convert_text_file_to_list(text_file_path)

# enable CORS
CORS(app, resources={r'/*': {'origins': '*'}})

@app.route('/add', methods=['GET'])
def add():
    api_key = request.headers.get('API-Key')
    if api_key not in api_keys:
        return {'error': 'Invalid API Key'}, 401
    param1 = request.args.get('one')
    param2 = request.args.get('two')   
    return jsonify(int(param1) + int(param2)), 200

@app.route('/add/three', methods=['GET'])
def addthree():
    api_key = request.headers.get('API-Key')
    if api_key not in api_keys:
        return {'error': 'Invalid API Key'}, 401
    param1 = request.args.get('one')
    return jsonify(int(param1) + 3), 200



if __name__ == '__main__':
    app.run(port=5001)