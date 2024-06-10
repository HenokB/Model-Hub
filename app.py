from flask import Flask, request, render_template, jsonify, send_file
import requests
import os
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


LLM_API_URL = "https://api-inference.huggingface.co/models/EthioNLP/EthioLLM-l-250K"
ASR_API_URL = "https://api-inference.huggingface.co/models/agkphysics/wav2vec2-large-xlsr-53-amharic"
AMHARIC_ENGLISH_MT_URL = "https://api-inference.huggingface.co/models/Atnafu/Amharic-English-MT"
ENGLISH_AMHARIC_MT_URL = "https://api-inference.huggingface.co/models/Atnafu/English-Amharic-MT"
TTS_API_URL = "https://api-inference.huggingface.co/models/Walelign/SpeechT5_Amharic_TTS_V1"
headers = {"Authorization": f"Bearer {os.getenv('HF_API_TOKEN')}"}



def query_tts(payload):
    response = requests.post(TTS_API_URL, headers=headers, json=payload)
    return response.content

@app.route('/text_to_speech', methods=['POST'])
def text_to_speech():
    data = request.json
    text = data.get('text')
    audio_bytes = query_tts({"inputs": text})
    audio_filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'tts_audio.wav')
    with open(audio_filepath, 'wb') as f:
        f.write(audio_bytes)
    return send_file(audio_filepath, as_attachment=True, download_name='tts_audio.wav')

headers = {"Authorization": "Bearer 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"}

def query_llm(payload):
    response = requests.post(LLM_API_URL, headers=headers, json=payload)
    return response.json()

def query_asr(filename):
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(ASR_API_URL, headers=headers, data=data)
    return response.json()

def query_translation(api_url, text, src_lang, tgt_lang):
    payload = {
        "inputs": text,
        "parameters": {
            "src_lang": src_lang,
            "tgt_lang": tgt_lang
        }
    }
    response = requests.post(api_url, headers=headers, json=payload)
    return response.json()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_suggestions', methods=['POST'])
def get_suggestions():
    data = request.json
    prompt = data.get('prompt')
    output = query_llm({"inputs": prompt})

    if output and isinstance(output, list):
        suggestions = [suggestion['sequence'] for suggestion in output]
        return jsonify({'suggestions': suggestions})
    else:
        return jsonify({'error': "Sorry, I couldn't understand that."})

@app.route('/upload_audio', methods=['POST'])
def upload_audio():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        output = query_asr(filepath)
        return jsonify({'transcription': output})
    return jsonify({'error': 'File upload failed'})

@app.route('/translate_amharic_to_english', methods=['POST'])
def translate_amharic_to_english():
    data = request.json
    text = data.get('text')
    output = query_translation(AMHARIC_ENGLISH_MT_URL, text, src_lang="amh", tgt_lang="en")
    return jsonify({'translation': output})

@app.route('/translate_english_to_amharic', methods=['POST'])
def translate_english_to_amharic():
    data = request.json
    text = data.get('text')
    output = query_translation(ENGLISH_AMHARIC_MT_URL, text, src_lang="en", tgt_lang="amh")
    return jsonify({'translation': output})

@app.route('/translate_amharic_file', methods=['POST'])
def translate_amharic_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        translation = query_translation(AMHARIC_ENGLISH_MT_URL, text, src_lang="amh", tgt_lang="en")
        translated_text = translation.get('translation', '')
        translated_filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'translated_amharic_to_english.txt')
        with open(translated_filepath, 'w', encoding='utf-8') as f:
            f.write(translated_text)
        return send_file(translated_filepath, as_attachment=True)
    return jsonify({'error': 'File upload failed'})

@app.route('/translate_english_file', methods=['POST'])
def translate_english_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        translation = query_translation(ENGLISH_AMHARIC_MT_URL, text, src_lang="en", tgt_lang="amh")
        translated_text = translation.get('translation', '')
        translated_filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'translated_english_to_amharic.txt')
        with open(translated_filepath, 'w', encoding='utf-8') as f:
            f.write(translated_text)
        return send_file(translated_filepath, as_attachment=True)
    return jsonify({'error': 'File upload failed'})

@app.route('/feedback', methods=['POST'])
def feedback():
    data = request.json
    index = data.get('index')
    feedback_type = data.get('type')
    print(f"Received {feedback_type} feedback for suggestion {index}")
    return jsonify({'message': 'Thank you for your feedback!'})

if __name__ == "__main__":
    #app.run(debug=True)
    app.run(host='0.0.0.0', port=5000)

