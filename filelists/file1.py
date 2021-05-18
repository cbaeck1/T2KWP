import os

def load_filepaths_and_text(filename, split="|"):
    with open(filename, encoding='utf-8-sig') as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
    return filepaths_and_text


#filepaths_and_texts = load_filepaths_and_text('filelists/selvas_main_train.txt')
filepaths_and_texts = load_filepaths_and_text('filelists/selvas_main_valid.txt')
# print(filepaths_and_texts[0][0])


#with open('filelists/new_selvas_main_train.txt', 'w', encoding='utf-8') as f:
with open('filelists/new_selvas_main_valid.txt', 'w', encoding='utf-8') as f:
    for file_text in filepaths_and_texts: 
        #print(file_text[0][47:52])
        # 01001 ~ 01049, 02001 ~ 02275
        if 1001 <= int(file_text[0][47:52]) <= 1049 or 2001 <= int(file_text[0][47:52]) <= 2275:
            print(file_text[0][47:52])
            f.write('{}|{}|{}|{}\n'.format(file_text[0], file_text[1], file_text[2], file_text[3]))

