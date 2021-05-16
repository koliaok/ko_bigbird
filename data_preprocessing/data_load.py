
total_data = []
with open('preprocessing_dataset/all.train', 'r', encoding='utf-8') as rd:
    for i, read in enumerate(rd.readlines()):
        print(read)
        data_len = len(read.split())
        total_data.append(data_len)
res = sorted(total_data, key=lambda item: item, reverse=True)

for i in res:
    print(i)
