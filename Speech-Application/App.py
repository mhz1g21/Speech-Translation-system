import webbrowser
import os
import sounddevice as sd
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from SpeechEngine import SpeechEngine

app = Flask(__name__)
app.secret_key = os.urandom(24)
speech_engine = SpeechEngine()

@app.route('/')
def home():
    model_name = session.get('model_name', 'default model')
    language = session.get('language', 'es')
    # Get the list of output devices
    devices = sd.query_devices()
    output_devices = [device['name'] for device in devices if device['max_output_channels'] > 0]

    # Get the default output device
    default_output_device = sd.default.device[1]

    # Set the speech engine's output device to the default
    speech_engine.change_output_device(default_output_device)

    # Run the speech engine
    speech_engine.change_model(model_name)
    #speech_engine.change_language(language)
    speech_engine.run()

    #create empty transcript files
    with open('transcript.txt', 'w+') as f:
        f.write('')
    with open('transcript-translated.txt', 'w+') as f:
        f.write('')

    models = ['Pre-Trained','Experimental']

    return render_template('home.html', output_devices=output_devices, models=models, default_device=default_output_device)

@app.route('/switch_device', methods=['POST'])
def switch_device():
    device_name = request.form.get('device_name')
    device_index = next(i for i, device in enumerate(sd.query_devices()) if device['name'] == device_name)
    speech_engine.change_output_device(device_index)
    return jsonify(message='Switched to device: {}'.format(device_name))

@app.route('/switch_model', methods=['POST'])
def switch_model():
    model_name = request.form.get('model_name')

    # Store the selected model in the session
    session['model_name'] = model_name

    # Reload the app
    return redirect(url_for('home'))

@app.route('/get_transcribed_text')
def get_transcribed_text():
    with open('transcript.txt', 'r') as f:
        transcript = f.read()
    return jsonify(transcript)

@app.route('/get_translated_text')
def get_translated_text():
    with open('transcript-translated.txt', 'r') as f:
        transcript = f.read()
    return jsonify(transcript)

@app.route('/switch_language', methods=['POST'])
def switch_language():
    language = request.form.get('language')

    # Store the selected language in the session
    session['language'] = language

    # Reload the app
    return redirect(url_for('home'))

@app.route('/shutdown', methods=['POST'])
def shutdown():
    os._exit(0)

if __name__ == '__main__':
    webbrowser.open_new('http://127.0.0.1:3000/')
    app.run(port=3000)