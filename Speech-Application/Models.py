import torch
import torch.nn as nn
import torch.nn.functional as F
from Decoder import Decoder
import torchaudio
import io
import soundfile as sf

class STTModel(nn.Module):
    def __init__(self, cnn_layer, rnn_layers, rnn_dim,n_class, n_feats, stride=2, dropout=0,device='cuda'):
        super(STTModel, self).__init__()
        n_feats = n_feats//2
        # cnn for extracting heirachal features
        self.cnn = nn.Conv2d(1,32,3, stride=stride, padding=3//2)
        # residual cnn for extracting heirachal features and reducing the sequence length
        self.resnetlayer = []
        for i in range(cnn_layer):
            self.resnetlayer.append({
                'cnn1':nn.Conv2d(32, 32, 3, 1, 1).to(device),
                'cnn2':nn.Conv2d(32, 32, 3, 1, 1).to(device),
                'ln1':nn.LayerNorm(n_feats).to(device),
                'ln2':nn.LayerNorm(n_feats).to(device),
                'actF':nn.ReLU(inplace=True).to(device),
                'dp':nn.Dropout(dropout).to(device)}
            )
        self.fully_connected = nn.Linear(32*n_feats, rnn_dim)
        #LSTM for sequence modeling
        self.LSTM = []
        for i in range(rnn_layers):
            self.LSTM.append({
                'lstm':nn.LSTM(input_size=rnn_dim if i==0 else rnn_dim*2, hidden_size=rnn_dim,
                        num_layers=1, batch_first=i==0, bidirectional=True).to(device),
                'ln':nn.LayerNorm(rnn_dim if i==0 else rnn_dim*2).to(device),
                'dp':nn.Dropout(dropout).to(device),
                'actF':nn.ReLU(inplace=True).to(device)}
            )
        #classifier
        self.classifier = nn.Sequential(
            nn.Linear(rnn_dim*2, rnn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(rnn_dim, n_class)
        )

    def forward(self, x):
        x = self.cnn(x)
        #
        for res in self.resnetlayer:
            x = self.res_forward(x,res)
        #
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])
        x = x.transpose(1, 2)
        x = self.fully_connected(x)
        #
        for lstm in self.LSTM:
            x = self.lstm_foward(x,lstm)
        #
        x = self.classifier(x)
        return x
    
    def res_forward(self, x,layer):
        residual = x
        out = layer['cnn1'](x)#cnn1
        out = out.transpose(2, 3).contiguous() # (batch, channel, time, feature)
        out = layer['ln1'](out)#layernorm1
        out = out.transpose(2, 3).contiguous() # (batch, channel, feature, time)
        out = layer['actF'](out)#relu
        out = layer['dp'](out)#dropout
        out = layer['cnn2'](out)#cnn2
        out = out.transpose(2, 3).contiguous() # (batch, channel, time, feature)
        out = layer['ln2'](out)#layernorm2
        out= out.transpose(2, 3).contiguous() # (batch, channel, feature, time)
        out += residual
        out = layer['actF'](out)#relu
        out = layer['dp'](out)#dropout
        return out
    
    def lstm_foward(self ,x ,layer):
        x = layer['ln'](x)#layernorm
        x = layer['actF'](x)#relu
        x, _ = layer['lstm'](x)#lstm
        x = layer['dp'](x)#dropout
        return x

class TTS_pretrained:
    def __init__(self,target_language='es'):
        self.language = 'es'
        self.model_id = 'v3_es'
        self.sample_rate = 48000
        self.speaker = 'es_1'
        self.device = torch.device('cpu')
        self.audio_path = 'temp_audio.wav'

        self.model, example_text = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                            model='silero_tts',
                                            language=self.language,
                                            speaker=self.model_id)
        self.model.to(self.device)  # gpu or cpu
    
    def change_language(self, language):
        if language == 'es':
            self.language = 'es'
            self.model_id = 'v3_es'
            self.speaker = 'es_1'
        elif language == 'de':
            self.language = 'de'
            self.model_id = 'v3_de'
            self.speaker = 'karlsson'
        elif language == 'fr':
            self.language = 'fr'
            self.model_id = 'v3_fr'
            self.speaker = 'fr_0'
    
    def synthesize(self, text):
        audio = self.model.apply_tts(text=text,
                        speaker=self.speaker,
                        sample_rate=self.sample_rate)
        print(type(audio))
        # Get the tensor from the list
        audio_tensor = audio

        # Convert the tensor to a numpy array
        audio_numpy = audio_tensor.numpy()

        # Write the numpy array to a WAV file
        sf.write(self.audio_path, audio_numpy, self.sample_rate)
        return self.audio_path
        
    
class STT_pretrained:
    def __init__(self):
        self.device = torch.device('cpu')  # gpu also works, but our models are fast enough for CPU
        self.model, self.decoder, self.utils = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                            model='silero_stt',
                                            language='en', # also available 'de', 'es'
                                            device=self.device)
        (self.read_batch, self.split_into_batches,
        self.read_audio, self.prepare_model_input) = self.utils  # see function signature for details
    
    def transcribe(self, audio_file):
        batches = self.split_into_batches([audio_file], batch_size=10)
        input = self.prepare_model_input(self.read_batch(batches[0]), device=self.device)
        output = self.model(input)
        for example in output:
           text = self.decoder(example.cpu())
        return text
    
class STT_experimental:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        params = {
            "cnnLayers" : 3,
            "lstmLayers" : 5,
            "rnnDim" : 256,
            "nClass" : 29,
            "nFeats" : 128,
            "stride" : 2,
            "dropout" : 0}
        self.model = STTModel(params['cnnLayers'], 
                                params['lstmLayers'], 
                                params['rnnDim'], 
                                params['nClass'], 
                                params['nFeats'], 
                                params['stride'], 
                                params['dropout'],device=self.device).to(self.device)
        self.model.load_state_dict(torch.load('modelv2_epoch_90.pt'))
        self.model.eval()
        self.decoder = Decoder()
    
    def transcribe(self, audio_file):
        waveform, sample_rate = torchaudio.load(audio_file)
        inputTransforms = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=128)
        input = inputTransforms(waveform).squeeze(0).transpose(0, 1)
        input = input.view(128,-1)
        input = input.unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            output = self.model(input.to(self.device))
        text = self.decoder.decode(output)
        text = ' '.join(text)
        return text
        