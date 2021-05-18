import matplotlib
matplotlib.use("agg")
import matplotlib.pylab as plt
import IPython.display as ipd
import librosa
import scipy.io.wavfile
#import Soundfile as sf

import sys
sys.path.append('waveglow/')
import numpy as np
import torch

from hparams import create_hparams
from model import Tacotron2
from layers import TacotronSTFT, STFT
from audio_processing import griffin_lim
from train import load_model
from text import text_to_sequence
from waveglow.denoiser import Denoiser


def plot_data(data, figsize=(16, 4)):
    fig, axes = plt.subplots(1, len(data), figsize=figsize)
    for i in range(len(data)):
        axes[i].imshow(data[i], aspect='auto', origin='bottom', 
                       interpolation='none')

hparams = create_hparams()
hparams.sampling_rate = 22050


checkpoint_path = "../../../data/checkpoint/tacotron2_statedict.pt"
model = load_model(hparams)

model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu'))['state_dict'])
# model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
# _ = model.cuda().eval().half()

waveglow_path = '../../../data/checkpoint/waveglow_256channels_universal_v5.pt'
# waveglow = torch.load(waveglow_path)['model']
waveglow = torch.load(waveglow_path, map_location=torch.device('cpu'))['model']

# waveglow.cuda().eval().half()
for k in waveglow.convinv:
    k.float()
denoiser = Denoiser(waveglow)

text = "Blessed is the man who does not walk in the counsel of the wicked or stand in the way of sinners or sit in the seat of mockers."
#text = '하나님은 사람이 아니시니 거짓말을 하지 않으시고 인생이 아니시니 후회가 없으시도다.'
#sequence = np.array(text_to_sequence(text, ['english_cleaners']))[None, :]
sequence = np.array(text_to_sequence(text, ['english_cleaners']))[None, :]
if torch.cuda.is_available():
    sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()
else:
    sequence = torch.autograd.Variable(torch.from_numpy(sequence)).long()

mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)
plot_data((mel_outputs.float().data.cpu().numpy()[0],
           mel_outputs_postnet.float().data.cpu().numpy()[0],
           alignments.float().data.cpu().numpy()[0].T))

with torch.no_grad():
    audio = waveglow.infer(mel_outputs_postnet, sigma=0.666)

ipd.Audio(audio[0].data.cpu().numpy(), rate=hparams.sampling_rate)
librosa.output.write_wav('sample2_librosa.wav', audio[0].data.cpu().numpy(), hparams.sampling_rate)
#scipy.io.wavfile.write('sample2_scipy.wav', hparams.sampling_rate, audio[0].data.cpu().numpy())
#sf.write('sample1_sf.wav', hparams.sampling_rate, audio)

#audio_denoised = denoiser(audio, strength=0.01)[:, 0]
#ipd.Audio(audio_denoised.cpu().numpy(), rate=hparams.sampling_rate)
#librosa.output.write_wav('sample2_librosa.wav', audio_denoised.cpu().numpy()[0], hparams.sampling_rate)
#scipy.io.wavfile.write('sample2_scipy.wav', hparams.sampling_rate, audio_denoised.cpu().numpy()[0])
#sf.write('sample2_sf.wav', hparams.sampling_rate, audio_denoised)





