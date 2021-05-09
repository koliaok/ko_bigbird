import big_theme_stock_system.setup_script_django
import json
import argparse
import pendulum

from django.db.models import Q
from data_preprocessing.tokenizer import BasicTokenizer
from big_theme_classifier.models import (
    NewsDataInfo
)

parser = argparse.ArgumentParser(description="Summary extractor argument")
parser.add_argument('--input_dir', type=str)
parser.add_argument('--save_dir', type=str)
added_parser = parser.parse_args()

class SummaryExtractor(object):

    def __init__(self):
        self.input_dir = added_parser.input_dir
        self.save_json_data()

    def save_json_data(self):
        loaded_summary_file = self.load_json_file(added_parser.input_dir)
        self.preprocessing_json_data(loaded_summary_file)

    def preprocessing_json_data(self, json_data):

        res_data = self.preprocessing_summary_from_database()
        res_data = self.preprocessing_summary_from_json(res_data)

        with open(added_parser.save_dir, 'w', encoding='utf-8') as wr:
            json.dump(res_data, wr, indent='\t')


    def preprocessing_summary_from_database(self):
        res_data = {}

        for data in self.get_summary_data_gen():
            update_summary = data.update_summary.replace('\n\n', ' ')
            print(update_summary)
            print(data.title)
            print(data.text)

        return res_data

    def preprocessing_summary_from_json(self, json_data):

        res_data = {}
        for i, data in enumerate(json_data['data']):

            original_sentence = self.data_preprocessing(" ".join(data["topic_sentences"]))
            summary_sentence = self.data_preprocessing(" ".join(data["summary_sentences"]))
            res_data[i] = {
                'document': original_sentence,
                'summary': summary_sentence
            }

        return res_data

    def load_json_file(self, input_dir):
        """
        json data load
        """
        with open(input_dir, 'r', encoding='utf-8') as rd:
            json_data = json.load(rd)

        return json_data

    def data_preprocessing(self, text):

        basic_tokenizer = BasicTokenizer()
        regex_text = basic_tokenizer.regex_replace(text)
        output_text = basic_tokenizer.tokenize(regex_text)
        output_text = ' '.join(output_text)

        return output_text

    def get_news_data_gen(self):
        """
        뉴스 데이터 가져오기
        """
        for news_data in NewsDataInfo.objects.all():
            yield news_data

    def get_summary_data_gen(self):
        """
        뉴스 데이터 가져오기
        """
        now = pendulum.now()
        one_month = pendulum.now().subtract(weeks=1)

        for news_data in NewsDataInfo.objects.filter(
            Q(time__gte=one_month, time__lte=now) &
            Q(update_summary__isnull=False)):
            yield news_data

    def write_process(self, wr, text):
        wr.write(text)
        wr.write('\n')
        wr.write('\n')


def main():
    summary_extractor = SummaryExtractor()

if __name__=="__main__":
    main()

