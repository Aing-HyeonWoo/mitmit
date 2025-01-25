from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from imagec import forward_img
from nlp import return_answer
from audio import return_audio_label

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TORCH_USE_CUDA_DSA"] = '1'

app = Flask(__name__)

CORS(app)


UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'wav'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "http://localhost:5500"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return response

@app.route('/upload', methods=['POST'])
def upload_files():
    if 'image' not in request.files or 'audio' not in request.files:
        return jsonify({'error': 'No image or audio file part'}), 400
    
    image_file = request.files['image']
    audio_file = request.files['audio']

    if image_file.filename == '' or audio_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if not allowed_file(image_file.filename) or not allowed_file(audio_file.filename):
        return jsonify({'error': 'Invalid file type. Allowed types are png, jpg, jpeg, gif, wav.'}), 400
    
    image_filename = os.path.join(UPLOAD_FOLDER, image_file.filename)
    image_file.save(image_filename)

    audio_filename = os.path.join(UPLOAD_FOLDER, audio_file.filename)
    audio_file.save(audio_filename)

    response_data = {
        'image': forward_img(image_filename),
        'audio': return_audio_label(audio_filename)
    }

    return jsonify(response_data)

@app.route('/chat', methods=['POST'])
def conversation():
    question = request.json.get("q")
    ans = return_answer(question)
    return jsonify({"a": ans})

if __name__ == '__main__':
    app.run(debug=True)
