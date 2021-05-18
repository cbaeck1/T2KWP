import os
import glob

def load_filepaths_and_text(filename, split="|"):
    with open(filename, encoding='utf-8-sig') as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
    return filepaths_and_text

# 전체
datapath = '/mnt/d/data'
# sample
#datapath = ''

filepaths = [
'korean_public_wav/korean_public_wav_orgin.txt'
]
wavpaths = [
'selvas_wav'
]
filepathsCnt = [0,0,0]
wavpathsCnt = [0,0,0]

'''
filepaths = [
'korean_public_wav/korean_public_wav.txt', 'korean_public_wav/korean_public_wav_orgin.txt',
'kss_wav/kss_wav.txt', 'kss/kss_wav_origin.txt', 
'selvas_wav/selvas_wav.txt', 'selvas_wav/selvas_wav_origin.txt'
]
# 전체, 존재, 존재하지 않음
filepathsCnt = [0,0,0, 0,0,0,
0,0,0, 0,0,0,
0,0,0, 0,0,0]
wavpathsCnt = [0,0,0, 0,0,0,
0,0,0, 0,0,0,
0,0,0, 0,0,0]


'''

# 문서기준으로 파일 존재여부 확인
iPosition = 0
for filepath in filepaths:
    filepaths_and_texts = load_filepaths_and_text(os.path.join(datapath, filepath))
    with open(os.path.join(datapath, filepath + '.new'), 'w', encoding='utf-8') as f:
        for file_text in filepaths_and_texts: 
            filepathsCnt[iPosition] += 1
            if os.path.isfile(file_text[0]):
                filepathsCnt[iPosition+1] += 1
                #print(file_text[0])
                listStr = '|'.join(file_text)    
                #f.write('{}|{}|{}|{}\n'.format(file_text[0], file_text[1], file_text[2], file_text[3]))
                f.write('{}\n'.format(listStr))
            else:
                filepathsCnt[iPosition+2] += 1
                print(file_text[0] + ' not exists!')

    iPosition += 3
print(filepathsCnt)

# 파일기준으로 문서에 있는지 확인
# iPosition = 0
# for wavpath in wavpaths:
#     wav_paths = glob.glob(os.path.join(wavpath, 'wav_16000', '*', '*.wav'))
#     with open(os.path.join(datapath, filepath + '.wavnew'), 'w', encoding='utf-8') as f:
#         for wav_path in wav_paths:
#             wav_filename = os.path.basename(wav_path)

