import big_theme_stock_system.setup_script_django

from big_theme_classifier.models import (
    NewsDataInfo
)

from data_preprocessing.tokenizer import BasicTokenizer

import argparse
import pendulum

parser = argparse.ArgumentParser(description="데이터처리")
parser.add_argument('--save_file', type=str)

added_parser = parser.parse_args()


class DataProcessing():

    def __init__(self):
        self.basic_tokenizer = BasicTokenizer()

    def write_process(self, wr, text):
        wr.write(text)
        wr.write('\n')
        wr.write('\n')

    def save_training_text_data(self):
        total_cnt = 0
        with open(added_parser.save_file, 'w', encoding='utf-8') as wr:
            for (news, original_token) in zip(self.get_news_data_gen(), self.get_original_data()):
                news_data = news.text
                original_data = original_token.preprocessing_data

                if total_cnt == 20:
                    break
                print(news_data)
                self.write_process(wr, news_data)
                self.write_process(wr, original_data)

                total_cnt += 1

    def row_data_save(self):
        preprocessing_text = ''
        total_cnt = 0
        with open('E:\original_database/text_data/bpe_total_pretraining_data.txt', 'r', encoding='utf-8') as file:
            with open(added_parser.save_file, 'w', encoding='utf-8') as wr:

                for i, read in enumerate(file.readlines()):

                    if i == 0:
                        continue

                    if read == '\n':
                        if len(preprocessing_text.split(' ')) < 10:
                            preprocessing_text = ''
                            continue
                        self.write_process(wr, preprocessing_text)

                        preprocessing_text = ''
                        total_cnt += 1

                    preprocessing_text += read.replace('\n', ' ').lstrip()

                for data in self.get_news_data_gen():

                    text = data.text
                    text = text.split('\n')
                    text = ' '.join(text).strip()
                    text = self.basic_tokenizer.regex_replace(text).strip()

                    self.write_process(wr, text)

    def news_data_save(self):
        with open(added_parser.save_file, 'w', encoding='utf-8') as wr:
            for data in self.get_test_news_data_gen():

                text = data.text
                text = text.split('\n')
                text = ' '.join(text).strip()
                text = self.basic_tokenizer.regex_replace(text).strip()
                self.write_process(wr, text)


    def data_preprocessing(self):
        #self.get_original_bpe()
        self.basic_tokenizer = BasicTokenizer()
        self.get_financial_data()

    def get_financial_data(self):
        for data in self.get_news_data_gen():
            text = data.text
            #regex_text = self.basic_tokenizer.regex_replace(text)
            #output_text = self.basic_tokenizer.tokenize(regex_text)
            #output_text = ' '.join(output_text)
            print(text)


    def get_news_data_gen(self):
        """
        뉴스 데이터 가져오기
        """
        for news_data in NewsDataInfo.objects.all():
            yield news_data

    def get_test_news_data_gen(self):
        """
        뉴스 데이터 가져오기
        """
        befor_one_year = pendulum.now().subtract(months=1)

        for news_data in NewsDataInfo.objects.filter(time__gte=befor_one_year):
            yield news_data


    #E:\original_database\text_data
    def get_original_bpe(self):
        preprocessing_text = ''
        original_text = ''
        with open('./original_dataset/bpe_total_pretraining_data.txt', 'r', encoding='utf-8') as file:
            for i, read in enumerate(file.readlines()):

                if i == 0:
                    continue

                if read == '\n':
                    if len(preprocessing_text.split(' ')) < 10:
                        preprocessing_text = ''
                        original_text = ''
                        continue

                    print(preprocessing_text)

                    try:
                        PreTrainingData.objects.create(preprocessing_data=preprocessing_text, original_data=original_text)
                    except Exception as ex:
                        print(ex)

                    preprocessing_text = ''
                    original_text = ''
                    continue

                preprocessing_text += read.replace('\n', ' ')
                original_text += read


def main():
    data_processing = DataProcessing()
    data_processing.news_data_save()
    #data_processing.row_data_save() -> 전체 wiki data + news data
    #data_processing.data_preprocessing()
    #data_processing.row_data_save()

if __name__ == "__main__":
    main()