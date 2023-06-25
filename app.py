from flask import Flask, jsonify, request
from flask_cors import CORS
import os
import data_proc as dp

# instantiate the app
app = Flask(__name__)
app.config.from_object(__name__)
app.config['MAX_CONTENT_LENGTH'] = 300 + (1024 * 1024) # Basic request size + large dataset limit

def convert_text_file_to_list(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        lines = [line.strip() for line in lines]
    return lines
text_file_path = 'apikeys.txt'
api_keys = convert_text_file_to_list(text_file_path)
print(api_keys)

# enable CORS
CORS(app, resources={r'/*': {'origins': '*'}})

UPLOAD_FOLDER = 'static/files'
app.config['UPLOAD_FOLDER'] =  UPLOAD_FOLDER



@app.route('/data/upload', methods=['POST'])
def upload():
    api_key = request.headers.get('API-Key')
    if api_key not in api_keys:
        return {'error': 'Invalid API Key'}, 401
    
    uploaded_file = request.files['file']
    id = request.form.get('id')
    
    if id == None:
        return {'error': 'No Dataset ID provided'}, 400
    
    if os.path.exists(os.path.join(app.config['UPLOAD_FOLDER'], "dataset_{}_{}".format(id, api_key))):
        return {'error': 'Dataset already exists; delete existing set before trying again'}, 409
    
    if uploaded_file.filename != '':
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], "dataset_{}_{}".format(id, api_key))
        uploaded_file.save(file_path)
        return {'info': 'Dataset uploaded'}, 200
    return {'error': 'Dataset not uploaded or given a file name'}, 412



@app.route('/data/delete', methods=['DELETE'])
def deleteData():
    api_key = request.headers.get('API-Key')
    if api_key not in api_keys:
        return {'error': 'Invalid API Key'}, 401
    
    id = request.form.get('id')
    
    if id == None or id == '': 
        return {'error': 'No Dataset ID provided'}, 400
    
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], "dataset_{}_{}".format(id, api_key))
    
    if os.path.exists(file_path) == False:
        return {'error': 'Dataset does not exist; create one before trying again'}, 409
    
    os.remove(file_path)
    return {'info': 'Dataset deleted'}, 200


 
if __name__ == '__main__':
    app.run(port=5001)