import numpy as np
from scipy.io.wavfile import read
import torch
import librosa


def get_mask_from_lengths(lengths):
    max_len = torch.max(lengths).item()
    if torch.cuda.is_available():
        ids = torch.arange(0, max_len, out=torch.cuda.LongTensor(max_len))
    else:
        ids = torch.arange(0, max_len, out=torch.LongTensor(max_len))
    mask = (ids < lengths.unsqueeze(1)).bool()
    return mask

# add 
def read_wav_np(path):
    try:
        sr, wav = read(path, mmap=True)
    except Exception as e1:
        #print('scipy.io.wavfile :' + str(e1) + path)
        try:
            wav, sr = librosa.load(path) 
        except Exception as e2:
            print('scipy.io.wavfile :' + str(e1) + path)
            print('librosa :' +str(e2) + path)
            return Exception

    if len(wav.shape) == 2:
        wav = wav[:, 0]

    if wav.dtype == np.int16:
        wav = wav / 32768.0
    elif wav.dtype == np.int32:
        wav = wav / 2147483648.0
    elif wav.dtype == np.uint8:
        wav = (wav - 128) / 128.0

    wav = wav.astype(np.float32)

    return sr, wav

def load_wav_to_torch(full_path):
    # scipy.wavefil.read does not take care of the case where wav is int or uint.
    # Thus, scipy.read is replaced with read_wav_np    
    # sampling_rate, data = read(full_path)
    sampling_rate, data = read_wav_np(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def load_filepaths_and_text(filename, split="|"):
    with open(filename, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
    return filepaths_and_text

# add
def files_to_list(filename):
    """
    Takes a text file of filenames and makes a list of filenames
    """
    with open(filename, encoding='utf-8') as f:
        files = f.readlines()

    files = [f.rstrip() for f in files]
    return files


def to_gpu(x):
    x = x.contiguous()

    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return torch.autograd.Variable(x)
