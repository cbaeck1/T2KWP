# from configs.hparams import create_hparams
from hparams import create_hparams
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import numpy as np
import librosa
from utils import read_wav_np
import os
hparams = create_hparams()
from scipy.io.wavfile import write
import torch
import glob
from scipy import interpolate
import pandas as pd

MOD_NUMBER = 400 # 전체데이터 400, sample 20

def build_from_path(in_dir, out_dir, meta_dir, meta_path, filelist_names, num_workers, tqdm=lambda x: x):
    wav_paths = []
    # for all speakers, count index and either add to train_list/eval_list/test_list
    # Create wav path list
    wav_paths = glob.glob(os.path.join(in_dir, 'wav_trimmed_22050', '*', '*.wav'))
    # texts2d = [[] for i in range(len(wav_paths))]
    texts2d = [[]]
    #for i in range(len(books)):
    with open(meta_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        print("파일 {}, 문서 {}".format(len(wav_paths), len(lines)))
        texts2d = [[] for i in range(len(lines))]
        for i in range(len(lines)):
            text = lines[i].split('|')
            for j in range(len(text)):
                texts2d[i].append(text[j])

    out_dir = os.path.join(out_dir, 'wav_22050') 
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    executor = ProcessPoolExecutor(max_workers=num_workers)
    futures = []
    futures_val = []
    futures_test = []
    indexP = 1
    for i, wav_path in enumerate(wav_paths):
        wav_filename = os.path.basename(wav_path)        

        try:
            # 2차원 리스트(list)의 특정 열(column) 골라 취하기
            # wav_name = wav_path.replace(in_dir+'/','')
            #print(list(zip(*texts2d))[0], wav_name)
            #sentence = list(zip(*texts2d))[0].index(wav_name)
            textslist = pd.DataFrame(texts2d)[0].to_list()
            sentence = textslist.index(wav_path)
            text = texts2d[sentence][1]
            speaker = texts2d[sentence][2]

            if not os.path.exists(os.path.join(out_dir, speaker)):
                os.makedirs(os.path.join(out_dir, speaker))
            out_path = os.path.join(out_dir, speaker, wav_filename)

            if int(indexP) % MOD_NUMBER == 0:
                # print(wav_path, out_path, speaker, text)
                futures_val.append(executor.submit(partial(_process_utterance, wav_path, out_path, speaker, text)))
            elif int(indexP) % MOD_NUMBER == 1:
                futures_test.append(executor.submit(partial(_process_utterance, wav_path, out_path, speaker, text)))
            else:
                futures.append(executor.submit(partial(_process_utterance, wav_path, out_path, speaker, text)))

        except:
            print('ERROR! OUT OF RANGE: {} {} {}'.format(i, wav_path, wav_filename))

        indexP += 1

    write_metadata([future.result() for future in tqdm(futures)], meta_dir, filelist_names[0])
    write_metadata([future.result() for future in tqdm(futures_val)], meta_dir, filelist_names[1])
    write_metadata([future.result() for future in tqdm(futures_test)], meta_dir, filelist_names[2])
'''
1. Read each file
2. Down sample to 22050Hz
3. Create meta file with the format 'path|phonemes|speaker'
["g",  "n",  "d",  "l",  "m",  "b",  "s",  "-", "j",   "q",
                "k", "t", "p", "h", "x", "w", "f", "c", "z", "A",
                "o", "O", "U", "u", "E", "a", "e", "1", "2", "3",
                "4", "5", "6", "7", "8", "9", "[", "]", "<", ">",
                "G", "N", "D", "L", "M", "B", "0", "K", ";;",";", "sp", "*",
                "$", "?", "!","#"]
'''
# change sampling rate to 22050
# trim
# Find matching text
# I have not decided whether to separate dierectories for train/eval/test
def _process_utterance(in_path, out_path, speaker, text):
    # Change sampling rate
    try:
        old_samplerate, old_audio = read_wav_np(in_path)
    except:
        return
    new_samplerate = hparams.sampling_rate

    if old_samplerate != new_samplerate:
        duration = old_audio.shape[0] / old_samplerate
        time_old = np.linspace(0, duration, old_audio.shape[0])
        time_new = np.linspace(0, duration, int(old_audio.shape[0] * new_samplerate / old_samplerate))

        interpolator = interpolate.interp1d(time_old, old_audio.T)
        new_audio = interpolator(time_new).T.astype(np.float32)
    else:
        new_audio = old_audio

    # Trim
    wav, _ = librosa.effects.trim(new_audio, top_db=25, frame_length=2048, hop_length=512)
    wav = torch.from_numpy(wav).unsqueeze(0)
    wav = wav.squeeze(0).numpy()
    write(out_path, 22050, wav)

    line = text.rstrip('\n')
    return (out_path, line, speaker)

def write_metadata(metadata, out_dir, out_file):
    with open(os.path.join(out_dir, out_file), 'w', encoding='utf-8') as f:
        for m in metadata:
            if m is None:
                continue
            f.write('|'.join([str(x) for x in m]) + '|1\n')