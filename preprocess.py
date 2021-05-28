import argparse
import os
from tqdm import tqdm
from datasets import kss_wav, public_korean_wav, selvas_wav, check_file_integrity, generate_mel_f0, f0_mean
from multiprocessing import cpu_count
from hparams import create_hparams
import torch

# TODO: lang code is written in this procedure. Langcode==1 for korean-only case is hard-coded for now.
# TODO: This must be fixed to support english and other languages as well.
def _integrate(meta_dir, train_file_lists, target_file, lang_code):
    sources = [[] for i in range(len(train_file_lists))]
    i = 0
    for file_list in train_file_lists:
        path_list = os.path.join(meta_dir, file_list)
        with open(path_list, 'r', encoding='utf-8') as f:
            sources[i] = f.readlines()
        i += 1

    # integrate meta file
    target_path = os.path.join(meta_dir, target_file)
    with open(target_path, 'w', encoding='utf-8') as f:
        for i in range(len(sources)):
            for j in range(len(sources[i])):
                sources[i][j] = sources[i][j].rstrip() + '\n'
            #    sources[i][j] = sources[i][j].rstrip() + '|{}\n'.format(str(lang_code))  # add language code

        for i in range(1, len(sources)):
            sources[0] += sources[i]

        # shuffle or not
        f.writelines(sources[0])


# This better not be done multithread because meta file is going to be locked and it will be inefficient.
def integrate_dataset(args):
    # sample 20개 
    # train_file_lists = ['sample_kss_wav_train.txt',
    #                     'sample_public_korean_wav_train.txt',
    #                     'sample_selvas_wav_train.txt'
    # ]
    # eval_file_lists = ['sample_kss_wav_valid.txt',
    #                    'sample_public_korean_wav_valid.txt',
    #                    'sample_selvas_wav_valid.txt'
    # ]
    # test_file_lists = ['sample_kss_wav_test.txt',
    #                    'sample_public_korean_wav_test.txt',
    #                    'sample_selvas_wav_test.txt'
    # ]
    # 전체데이터
    train_file_lists = ['kss_wav_train.txt',
                        'public_korean_wav_train.txt',
                        'selvas_wav_train.txt'
    ]
    eval_file_lists = ['kss_wav_valid.txt',
                       'public_korean_wav_valid.txt',
                       'selvas_wav_valid.txt'
    ]
    test_file_lists = ['kss_wav_test.txt',
                       'public_korean_wav_test.txt',
                       'selvas_wav_test.txt'
    ]

    target_train_file = 'merge_train.txt'
    target_eval_file = 'merge_valid.txt'
    target_test_file = 'merge_test.txt'

    # merge train lists Langcode==1 for korean-only
    _integrate(args.meta_dir, train_file_lists, target_train_file, 1)

    # merge eval lists Langcode==1 for korean-only
    _integrate(args.meta_dir, eval_file_lists, target_eval_file, 1)

    # merge test lists Langcode==1 for korean-only
    _integrate(args.meta_dir, test_file_lists, target_test_file, 1)

    print('Dataset integration has been complete')

# Try opening files on the filelist and write down the files with io error.
def check_for_file_integrity(args):
    lists = ['merge_train.txt', 'merge_valid.txt', 'merge_test.txt']
    check_file_integrity.check_paths(lists, args.meta_dir, args.num_workers, tqdm=tqdm)

def gen_mel_f0(args):
    lists = ['merge_train.txt', 'merge_valid.txt', 'merge_test.txt']
    generate_mel_f0.build_from_path(lists, args.meta_dir, args.hparams, args.num_workers, tqdm=tqdm)

def preprocess_cal_f0_scale_per_training_speaker(args):
    root_list = ['kss_wav/wav_22050','korean_public_wav/wav_22050','selvas_wav/wav_22050']
    f0_mean.build_from_path(root_list, args.hparams, args.num_workers, tqdm=tqdm)

def preprocess_kss_wav(args, isSample):
    # sample 20개
    if isSample:
        meta_path = 'kss/metadata.csv'
        in_dir = 'kss' 
        out_dir = 'kss_wav'
        filelists_name = [
            'sample_kss_wav_train.txt',
            'sample_kss_wav_valid.txt',
            'sample_kss_wav_test.txt'
        ]
    # 전체데이터
    else:
        meta_path = '/mnt/d/data/kss/kss_wav_origin.txt.new'
        in_dir = '/mnt/d/data/kss' 
        out_dir = '/mnt/d/data/kss_wav' 
        filelists_name = [
            'kss_wav_train.txt',
            'kss_wav_valid.txt',
            'kss_wav_test.txt'
        ]

    kss_wav.build_from_path(in_dir, out_dir, args.meta_dir, meta_path, filelists_name, 1, tqdm=tqdm)

def preprocess_public_korean_wav(args, isSample):
    # sample 20개
    if isSample:
        meta_path = 'public_korean_wav/metadata.txt'
        in_dir = 'public_korean_wav' 
        out_dir = 'public_korean_wav'
        filelists_name = [
            'sample_public_korean_wav_train.txt',
            'sample_public_korean_wav_valid.txt',
            'sample_public_korean_wav_test.txt'
        ]
    # 전체데이터
    else:
        meta_path = '/mnt/d/data/public_korean_wav/public_korean_wav_orgin.txt'
        in_dir = '/mnt/d/data/public_korean_wav' 
        out_dir = '/mnt/d/data/public_korean_wav' 
        filelists_name = [
            'public_korean_wav_train.txt',
            'public_korean_wav_valid.txt',
            'public_korean_wav_test.txt'
        ]

    public_korean_wav.build_from_path(in_dir, out_dir, args.meta_dir, meta_path, filelists_name, args.num_workers, tqdm=tqdm)

def preprocess_selvas_wav(args, isSample):
    # sample 20개
    if isSample:
        meta_path = 'selvas_wav/metadata.txt'
        in_dir = 'selvas_wav' 
        out_dir = 'selvas_wav'
        filelists_name = [
            'sample_selvas_wav_train.txt',
            'sample_selvas_wav_valid.txt',
            'sample_selvas_wav_test.txt'
        ]
    # 전체데이터
    else:
        meta_path = '/mnt/d/data/selvas_wav/selvas_wav_origin.txt.new'
        in_dir = '/mnt/d/data/selvas_wav' 
        out_dir = '/mnt/d/data/selvas_wav' 
        filelists_name = [
            'selvas_wav_train.txt',
            'selvas_wav_valid.txt',
            'selvas_wav_test.txt'
        ]

    selvas_wav.build_from_path(in_dir, out_dir, args.meta_dir, meta_path, filelists_name, args.num_workers, tqdm=tqdm)

def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--base_dir', default=os.path.expanduser('/past_projects/DB'))
    # parser.add_argument('--output', default='sitec')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--dataset', required=False, default='kss_wav',
                        choices=['integrate_dataset', 'public_korean_wav', 'kss_wav', 'selvas_wav', 
                                 'check_file_integrity', 'generate_mel_f0', 'cal_f0_scale_per_training_speaker'])
    parser.add_argument('--hparams', default='',
                        help='Hyperparameter overrides as a comma-separated list of name=value pairs')
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--meta_dir', type=str, default='filelists')
    parser.add_argument('--sample', type=bool, default=True)
    args = parser.parse_args()
    args.num_workers = cpu_count() if args.num_workers is None else int(args.num_workers)  # cpu_count() = process 갯수
    args.hparams = create_hparams()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.dataset == 'integrate_dataset':
        integrate_dataset(args)
    elif args.dataset == 'kss_wav':
        preprocess_kss_wav(args, args.sample)
    elif args.dataset == 'public_korean_wav':
        preprocess_public_korean_wav(args, args.sample)
    elif args.dataset == 'selvas_wav':
        preprocess_selvas_wav(args, args.sample)
    elif args.dataset == 'check_file_integrity':
        check_for_file_integrity(args)
    elif args.dataset == 'generate_mel_f0':
        gen_mel_f0(args)
    elif args.dataset == 'cal_f0_scale_per_training_speaker':
        preprocess_cal_f0_scale_per_training_speaker(args)


if __name__ == "__main__":
    main()
