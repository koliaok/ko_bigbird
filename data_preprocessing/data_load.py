from tqdm import tqdm
import pandas as pd

def load_pretraining():
    total_data = []
    with open('preprocessing_dataset/all.train', 'r', encoding='utf-8') as rd:
        with open('../bigbird/datasource/pretrained_data/pretraining_data_v1.train', 'w', encoding='utf-8') as wr:
            cnt = 0
            total_cnt = 0
            for read in tqdm(rd.readlines()):
                data_len = len(read.split())
                if len(read.split()) > 30:
                    wr.write(read)
                    wr.write('\n')
                    wr.write('\n')
                    total_data.append(data_len)
                    cnt += 1
                total_cnt += 1

            print('total : ', total_cnt)
            print('train file cnt : ', cnt)


    res = sorted(total_data, key=lambda item: item, reverse=True)

    for i in res[:10]:
        print(i)

def add_pertraining():
    total_data = []

    with open('../bigbird/datasource/pretrained_data/pretraining_data_v1.train', 'a', encoding='utf-8') as wr:
        with open('preprocessing_dataset/news_data_v1.train', 'r', encoding='utf-8') as rd:
            cnt = 0
            total_cnt = 0
            for read in tqdm(rd.readlines()):
                data_len = len(read.split())
                if len(read.split()) > 30:
                    wr.write(read)
                    wr.write('\n')
                    wr.write('\n')
                    total_data.append(data_len)
                    cnt += 1
                total_cnt += 1

            print('total : ', total_cnt)
            print('train file cnt : ', cnt)


    res = sorted(total_data, key=lambda item: item, reverse=True)

    for i in res[:10]:
        print(i)

#load_pretraining()
add_pertraining()