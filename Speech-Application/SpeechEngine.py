import pyaudio
import threading
import time
import argparse
import wave
import torchaudio
import torch
import sys
import wave
import numpy as np
from Decoder import Decoder
from threading import Event
from Models import STT_experimental, STT_pretrained, TTS_pretrained
import deepl
import time
import sounddevice as sd
import soundfile as sf

class Listener:
    # Initialize Listener with default sample rate and record seconds
    def __init__(self, sample_rate=16000, record_seconds=2):
        self.chunk = 1024  # Size of audio data read in one iteration
        self.sample_rate = sample_rate  # Sample rate of the audio input
        self.record_seconds = record_seconds  # Duration of recording
        self.audio = pyaudio.PyAudio()  # Create a PyAudio instance
        self.stream = self._create_stream()  # Create an audio stream

    # Create an audio stream with the specified format, channels, rate, and buffer size
    def _create_stream(self):
        return self.audio.open(format=pyaudio.paInt16,
                               channels=1,
                               rate=self.sample_rate,
                               input=True,
                               output=True,
                               frames_per_buffer=self.chunk)

    # Continuously read data from the audio stream and append it to the queue
    def _listen(self, queue):
        while True:
            data = self.stream.read(self.chunk, exception_on_overflow=False)
            queue.append(data)
            time.sleep(0.01)  # Sleep for a short duration to prevent CPU overuse

    # Start a new thread that listens to the audio stream and appends data to the queue
    def run(self, queue):
        thread = threading.Thread(target=self._listen, args=(queue,), daemon=True)
        thread.start()
        print("Speech Recognition engine is now listening...\n")



class SpeechEngine:
    def __init__(self, stt_system='Pre-Trained'):
        self.auth_key = "ce8a44c1-908c-b3f8-47e0-36282300e5a5:fx"
        self.translator = deepl.Translator(self.auth_key)
        self.queue = []
        self.listener = Listener()
        self.stt_system = stt_system
        self.stt1 = STT_experimental()
        self.stt2 = STT_pretrained()
        self.tts = TTS_pretrained()
        self.speaker = Speaker()
        self.model_lock = threading.Lock()
        self.target_lang = 'es'

    def choose_stt(self):
        with self.model_lock:
            if self.stt_system == 'Experimental':
                return self.stt1
            else:
                return self.stt2
        
    def change_model(self, stt_system):
        self.stt_system = stt_system
    
    def change_language(self, language):
        if language in ['es', 'de', 'fr']:
            self.target_lang = language
            self.tts.change_language(language)
        else:
            print(f"Unsupported language: {language}")
            

    def translate_text(self, text):
        result = self.translator.translate_text(text,target_lang=self.target_lang)
        return result

    def text_to_speech(self, text):
        # Use the TTS system to convert the text to speech
        return self.tts.speak(text)
    
    def change_output_device(self, device):
        # Change the output device for the speaker
        self.speaker.device = device
    
    def save_temp(self,audio,filenmame='temp.wav'):
        audiof = wave.open(filenmame, 'wb')
        audiof.setnchannels(1)
        audiof.setsampwidth(self.listener.audio.get_sample_size(pyaudio.paInt16))
        audiof.setframerate(16000)
        audiof.writeframes(b''.join(audio))
        audiof.close()
        return filenmame
    
    def save_transcript(self,transcript):
        with open('transcript.txt', 'a+') as f:
            f.write(" " + transcript)
    
    def save_translated_transcript(self,transcript):
        with open('transcript-translated.txt', 'a+') as f:
            f.write(transcript)


    def speech_loop(self):
        # Initialize an empty string to hold the concatenated text
        concatenated_text = ""
        # Record the current time
        last_print_time = time.time()

        # Start an infinite loop
        while True:
            # If the queue has less than 5 items, skip the rest of the loop and start over
            if len(self.queue) < 5:
                continue
            else:
                # Copy the queue and clear the original
                pred_q = self.queue.copy()
                self.queue.clear()
                # Save the copied queue as audio
                audio = self.save_temp(pred_q)
                # Choose a speech-to-text service
                stt = self.choose_stt()
                # Transcribe the audio to text
                text = stt.transcribe(audio)
                # Save the transcribed text
                self.save_transcript(text)
                # Add the transcribed text to the concatenated text
                concatenated_text += " " + text
                # Count the number of words in the concatenated text
                word_count = len(concatenated_text.split(" "))
                # Calculate the time elapsed since the last print
                current_time = time.time()
                time_elapsed = current_time - last_print_time
                # If the word count is 20 or more, or 10 or more seconds have passed since the last print
                if word_count >= 20 or time_elapsed >= 10:
                    # Print the concatenated text
                    print("concat text " + concatenated_text)
                    # Translate the concatenated text
                    trans_text = self.translate_text(concatenated_text)
                    trans_text = str(trans_text)
                    # Save the translated text
                    self.save_translated_transcript(trans_text)
                    # Print the translated text
                    print("translated text: " + trans_text)
                    # Speak the translated text
                    self.speaker.speak(trans_text)
                    # Reset the concatenated text and the last print time
                    concatenated_text = ""
                    last_print_time = current_time

            # Pause for 0.05 seconds before starting the loop again
            time.sleep(0.05)
        

    def run(self):
        self.listener.run(self.queue)
        thread = threading.Thread(target=self.speech_loop, daemon=True)
        thread.start()
    
class Speaker:
    def __init__(self):
        self.tts = TTS_pretrained()
        self.device = 7

    def speak(self,text):
        audio_path = self.tts.synthesize(text)
        # Load the audio file
        data, samplerate = sf.read(audio_path)

        # Convert the data to float32
        data = data.astype(np.float32)

        # Play the audio file on the specified device
        with sd.OutputStream(device=self.device, channels=1, samplerate=samplerate) as stream:
            stream.write(data)
        

                
    