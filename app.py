from flask import Flask, jsonify, request
from flask_cors import CORS
import os
import model as md
import pickle
import utils

# instantiate the app
app = Flask(__name__)
app.config.from_object(__name__)
app.config['MAX_CONTENT_LENGTH'] = 300 + (1024 * 1024) # Basic request size + large dataset limit

text_file_path = 'apikeys.txt'
api_keys = utils.convert_text_file_to_list(text_file_path)

# enable CORS
CORS(app, resources={r'/*': {'origins': '*'}})

UPLOAD_FOLDER = 'static/files'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER



@app.route('/data/upload', methods=['POST'])
def upload():
    api_key = request.headers.get('API-Key')
    if api_key not in api_keys:
        return {'error': 'Invalid API Key'}, 401
    
    uploaded_file = request.files['file']
    
    # TODO verify safe file ID
    
    id = request.form.get('id')
    
    if id == None:
        return {'error': 'No Dataset ID provided'}, 400
    
    if os.path.exists(os.path.join(app.config['UPLOAD_FOLDER'], "dataset_{}_{}.csv".format(id, api_key))):
        return {'error': 'Dataset already exists; delete existing set before trying again'}, 409
    
    if uploaded_file.filename != '':
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], "dataset_{}_{}.csv".format(id, api_key))
        uploaded_file.save(file_path)
        return {'info': 'Dataset uploaded'}, 200
    
    return {'error': 'Dataset not uploaded or given a file name'}, 412



@app.route('/data/delete', methods=['DELETE'])
def delete_data():
    api_key = request.headers.get('API-Key')
    if api_key not in api_keys:
        return {'error': 'Invalid API Key'}, 401
    
    id = request.form.get('id')
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], "dataset_{}_{}.csv".format(id, api_key))
    
    if id == None or id == '': 
        return {'error': 'No Dataset ID provided'}, 400
    
    if os.path.exists(file_path) == False:
        return {'error': 'Dataset does not exist; create one before trying again'}, 409
    
    os.remove(file_path)
    return {'info': 'Dataset deleted'}, 200



@app.route('/model/create', methods=['POST'])
def create_model():
    api_key = request.headers.get('API-Key')
    if api_key not in api_keys:
        return {'error': 'Invalid API Key'}, 401
    
    id = request.form.get('id')
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], "dataset_{}_{}.csv".format(id, api_key))
    
    if os.path.exists(file_path) == False:
        return {'error': 'Dataset does not exist; create one before trying again'}, 409
    
    model_name = request.form.get('model_name')
    type = request.form.get('type')
    visual_name = request.form.get('visual_name')
    model_path = os.path.join(app.config['UPLOAD_FOLDER'], "model_{}_{}.pk1".format(id, api_key))
    
    model = md.create_model(file_path, model_name, visual_name, type, model_path)
    return_value = model[0]
    model = model[1]
    
    utils.save(model, model_path)
    
    print(model.train())

    return return_value



@app.route('/model/name', methods=['GET'])
def return_model_name():
    api_key = request.headers.get('API-Key')
    if api_key not in api_keys:
        return {'error': 'Invalid API Key'}, 
    
    id = request.form.get('id')
    
    if id == None or id == '': 
        return {'error': 'No Dataset ID provided'}, 400
    
    if os.path.exists(os.path.join(app.config['UPLOAD_FOLDER'], "model_{}_{}.pk1".format(id, api_key))) == False:
        return {'error': 'Model does not exist; create one before trying again'}, 409
    
    model = utils.load_model_from_file(id, api_key)
    
    return model.name
 
 
 
if __name__ == '__main__':
    app.run(port=5001)