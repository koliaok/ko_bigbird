from tqdm import tqdm
import pandas as pd

total_data = []
with open('preprocessing_dataset/all.train', 'r', encoding='utf-8') as rd:
    with open('preprocessing_dataset/pretraining_data.train', 'w', encoding='utf-8') as wr:
        cnt = 0
        total_cnt = 0
        for read in tqdm(rd.readlines()):
            data_len = len(read.split())
            if len(read.split()) > 10:
                wr.write(read)
                wr.write('\n')
                wr.write('\n')
                total_data.append(data_len)
                cnt + 1
            total_cnt += 1

        print('total : ', total_cnt)
        print('train file cnt : ', cnt)


res = sorted(total_data, key=lambda item: item, reverse=True)

for i in res[:10]:
    print(i)
